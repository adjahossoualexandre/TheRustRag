from lightrag.components.model_client import TransformersClient
from transformers import AutoTokenizer, AutoModelForTextEncoding, AutoConfig
from lightrag.core import Embedder
import os

def load_from_HuggingFace(model_name, tokenizer_name, auto_model):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = auto_model.from_pretrained(model_name)

    return model, tokenizer

def load_local_model(model_path: str, auto_model):
    config = AutoConfig.from_pretrained(model_path)
    model = auto_model.from_config(config)
    return model, config

def save_model(model, tokenizer, persist_dir):
    tokenizer.save_pretrained(os.path.join(persist_dir, tokenizer.name_or_path))
    model.save_pretrained(os.path.join(persist_dir, model.name_or_path))


MODEL_PATH = f"sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_PATH = f"sentence-transformers/all-MiniLM-L6-v2"

model = load_from_HuggingFace(MODEL_PATH, TOKENIZER_PATH, AutoModelForTextEncoding)

save_model(model[0], model[1], "persist_temp")

a,b = load_local_model("persist_temp/" + MODEL_PATH, AutoModelForTextEncoding)

model_kwargs = {
    "model": "persist_temp/" + MODEL_PATH
}

local_embedder = Embedder(model_client=TransformersClient(), model_kwargs=model_kwargs)

output = local_embedder("Explain the borrow-checker to a five years old.")