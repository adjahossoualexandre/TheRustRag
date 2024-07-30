""""Model: malay-huggingface/t5-small-bahasa-cased"""
from abc import ABC, abstractclassmethod
from typing import Any, Dict, Union, List, Optional
from functools import lru_cache
from lightrag.core.types import ModelType, Embedding, EmbedderOutput
import torch.nn.functional as F
import torch
import logging
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from lightrag.core import ModelClient
from lightrag.core import Embedder
from torch import Tensor

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

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

class T5SmallBahasaCased(TransformerEmbedder):

    @lru_cache(None)
    def init_model(
        self,
        model_name: str,
        auto_model: Optional[type] = AutoModel,
        ):
        """For this example, we need to use the class T5Model"""
        auto_model = T5EncoderModel
        super().init_model(model_name, auto_model)


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
        input_ids, attention_mask = self.tokenize_inputs(input)
        outputs = self.compute_model_outputs(input_ids, model)
        embeddings = self.compute_embeddings(outputs, attention_mask)

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
        input_ids = batch_dict["input_ids"]
        attention_mask = batch_dict["attention_mask"]
        return input_ids, attention_mask

    def compute_model_outputs(self, input_ids, model):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        return outputs

    def compute_embeddings(self, outputs, attention_mask):
        embeddings = average_pool(
            outputs.last_hidden_state, attention_mask
        )
        return embeddings


MODEL_PATH = "malay-huggingface/t5-small-bahasa-cased" 
TOKENIZER_PATH = "malay-huggingface/t5-small-bahasa-cased"
PERSIST_DIR = "../persist_temp/"

sentence_ref = "soalan: siapakah perdana menteri malaysia?"

# load embedding model
model_kwargs = {
    "model": MODEL_PATH
}

transformer_embedder = T5SmallBahasaCased(MODEL_PATH)
model_client = CustomModelClient(transformer_embedder)
local_embedder = Embedder(
    model_client=model_client,
    model_kwargs=model_kwargs
    )
local_embedder(sentence_ref)
