from transformers import AutoTokenizer, AutoModel, AutoConfig
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
