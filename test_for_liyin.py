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
from lightrag.core import Embedder



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
            self.init_model(model_name=self.model_name)
            
    @lru_cache(None)
    def init_model(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
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

        # load files and models, cache it for the next inference
        model_name = kwargs["model"]

        # inference the model
        return self.infer_embedding(kwargs["input"])

class ConcreteTransformerEmbedder(TransformerEmbedder):

    def __init__(self, model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.models = dict()
        if model_name is not None:
            self.model_name = model_name
            self.init_model(model_name=self.model_name)

    @lru_cache(None)
    def init_model(self, model_name: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            # register the model
            self.models[model_name] = self.model
            log.info(f"Done loading model {model_name}")

        except Exception as e:
            log.error(f"Error loading model {model_name}: {e}")
            raise e

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

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if tolist:
            embeddings = embeddings.tolist()
        return embeddings
    
    def handle_input(self, input):
        if isinstance(input, str):
            input = [input]
        return input
    
    def tokenize_inputs(self, input):
        batch_dict = self.tokenizer(
            input, padding=True, truncation=True, return_tensors='pt'
        )
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

    def __call__(self, **kwargs):
        if "model" not in kwargs:
            raise ValueError("model is required")

        if "mock" in kwargs and kwargs["mock"]:
            import numpy as np

            embeddings = np.array([np.random.rand(768).tolist()])
            return embeddings

        # load files and models, cache it for the next inference
        model_name = kwargs["model"]

        # inference the model
        return self.infer_embedding(kwargs["input"])

class CustomModelClient(ModelClient):


    def __init__(self, transformer_embedder: TransformerEmbedder) -> None:
        super().__init__()
        self.transformer_embedder = transformer_embedder
        self.sync_client = self.init_sync_client()
        self.async_client = None

    def init_sync_client(self):
        return self.transformer_embedder()

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

# reference: https://huggingface.co/intfloat/multilingual-e5-large
from torch import Tensor
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class MultiLanguageE5LargeEmbedder(TransformerEmbedder):

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
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings


MODEL_PATH = "intfloat/multilingual-e5-large"
TOKENIZER_PATH = "intfloat/multilingual-e5-large"
PERSIST_DIR = "persist_temp/"

sentence_ref = "That is a happy person"

# load embedding model
model_kwargs = {
    "model": MODEL_PATH
}

transformer_embedder = MultiLanguageE5LargeEmbedder(MODEL_PATH)
model_client = CustomModelClient(transformer_embedder)
local_embedder = Embedder(
    model_client=model_client,
    model_kwargs=model_kwargs
    )

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

load_from_HuggingFace(MODEL_PATH, MODEL_PATH)