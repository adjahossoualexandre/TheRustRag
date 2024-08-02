#### Custom TransformerEmbedding
from abc import ABC, abstractclassmethod
from typing import Any, Dict, Union, List, Optional
from functools import lru_cache
from lightrag.core.types import ModelType, Embedding, EmbedderOutput
import torch.nn.functional as F
import torch
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig
from lightrag.core import ModelClient
import os

def load_from_HuggingFace(model_name, tokenizer_name):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)

    return model, tokenizer

def load_local_model(model_name: str, persist_dir) -> tuple:
    model_path = os.path.join(persist_dir, model_name)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_config(config)
    return model, config

def save_model(model, tokenizer, persist_dir):
    tokenizer.save_pretrained(os.path.join(persist_dir, tokenizer.name_or_path))
    model.save_pretrained(os.path.join(persist_dir, model.name_or_path))


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

log = logging.getLogger(__name__)

class TransformerEmbedder(ABC):

    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        self.models = dict()
        if model_name is not None:
            self.model_name = model_name
            """Lazy intialisation of the model in TransformerClient.init_sync_client()"""
            #self.init_model(model_name=self.model_name)
            
    @lru_cache(None)
    def init_model(self, model_name: str, auto_model: Optional[type] = AutoModel, auto_tokenizer: Optional[type] = AutoTokenizer):
        try:
            self.tokenizer = auto_tokenizer.from_pretrained(model_name)
            self.model = auto_model.from_pretrained(model_name)
            # register the model
            self.models[model_name] = self.model
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

    @abstractclassmethod
    def infer_embedding(
        self,
        input=Union[str, List[str]],
        tolist: bool = True,
    ):
        pass

    @abstractclassmethod
    def handle_input(self, input):
        pass

    @abstractclassmethod
    def tokenize_inputs(self, input):
        pass

    @abstractclassmethod
    def compute_model_outputs(self, batch_dict, model):
        pass
    @abstractclassmethod
    def compute_embeddings(self, outputs, batch_dict):
        pass

    def __call__(self, **kwargs):
        if "model" not in kwargs:
            raise ValueError("model is required")

        if "mock" in kwargs and kwargs["mock"]:
            import numpy as np

            embeddings = np.array([np.random.rand(768).tolist()])
            return embeddings

        # inference the model
        return self.infer_embedding(kwargs["input"])

class CustomModelClient(ModelClient):


    def __init__(self, transformer_embedder: TransformerEmbedder) -> None:
        super().__init__()
        self.transformer_embedder = transformer_embedder
        self.sync_client = self.init_sync_client()
        self.async_client = None

    def init_sync_client(self):
        model_name = self.transformer_embedder.model_name
        self.transformer_embedder.init_model(model_name)
        """The transformerEmbedder is initialised by the user so I removed the parenthesis from the return statement to avoid executing self.transformer_embedder.call()"""
        return self.transformer_embedder

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        embeddings: List[Embedding] = []
        for idx, emb in enumerate(response):
            embeddings.append(Embedding(index=idx, embedding=emb))
        response = EmbedderOutput(data=embeddings)
        return response

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        if "model" not in api_kwargs:
            raise ValueError("model must be specified in api_kwargs")
        if (
            model_type == ModelType.EMBEDDER
            and "model" in api_kwargs
            ############ No need for this anymore
            #and api_kwargs["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        ):
            if self.sync_client is None:
                self.sync_client = self.init_sync_client()
            return self.sync_client(**api_kwargs)

    def convert_inputs_to_api_kwargs(
        self,
        input: Any,  # for retriever, it is a single query,
        model_kwargs: dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> dict:
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            final_model_kwargs["input"] = input
            return final_model_kwargs

class AllMiniLML6V2Embedder(TransformerEmbedder):

    def infer_embedding(
        self,
        input=Union[str, List[str]],
        tolist: bool = True,
    ):
        model = self.models.get(self.model_name, None)
        if model is None:
            # initialize the model
            self.init_model(self.model_name)

        self.handle_input(input)
        batch_dict = self.tokenize_inputs(input)
        outputs = self.compute_model_outputs(batch_dict, model)
        embeddings = self.compute_embeddings(outputs, batch_dict)

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if tolist:
            embeddings = embeddings.tolist()
        return embeddings


    def handle_input(self, input):
        if isinstance(input, str):
            input = [input]
        return input
     
    def tokenize_inputs(self, input):
        batch_dict = self.tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch_dict

    def compute_model_outputs(self, batch_dict, model):
        with torch.no_grad():
            outputs = model(**batch_dict)
        return outputs

    def compute_embeddings(self, outputs, batch_dict):
        embeddings = mean_pooling(
            outputs, batch_dict["attention_mask"]
        )
        return embeddings