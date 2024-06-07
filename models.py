from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def load_and_cache_embedding_model(model_name: str, cache_folder: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=cache_folder
        )
def load_local_embedding_model(model_path: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=model_path
        )