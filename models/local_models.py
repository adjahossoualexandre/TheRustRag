# Download model from HF
from model_utils import load_from_HuggingFace, save_model

if __name__ == "__main__":

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATION_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_STORE = "model_store"
    LOCAL_EMBEDDING_MODEL = MODEL_STORE + "/" + EMBEDDING_MODEL 
    LOCAL_GENERATRION_MODEL = MODEL_STORE + "/" + GENERATION_MODEL 
    
    # load and save embedding model
    model, tokenizer = load_from_HuggingFace(EMBEDDING_MODEL, EMBEDDING_MODEL)
    save_model(model, tokenizer, LOCAL_EMBEDDING_MODEL)

    # load and save generation model
    model, tokenizer = load_from_HuggingFace(GENERATION_MODEL, GENERATION_MODEL)
    save_model(model, tokenizer, LOCAL_GENERATRION_MODEL)

