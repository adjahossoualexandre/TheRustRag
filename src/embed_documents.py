## Configure the splitter settings
from lightrag.components.data_process.text_splitter import TextSplitter
from lightrag.components.data_process import text_splitter


# Generate embeddings
from src.models.embedding_model import CustomEmbeddingModelClient, AllMiniLML6V2Embedder
from lightrag.core import Embedder, Sequential
from lightrag.components.data_process import ToEmbeddings

from lightrag.core.db import LocalDB
import os

 
def embed(
        embedding_model: str,
        model_store: str,
        doc_store: str,
        key: str,
        batch_size: int,
        split_by: str,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer_kwargs: dict,
        custom_separator: dict[str, str] = None

) -> int:
    model_path= model_store + "/" + embedding_model

    try:
        db = LocalDB().load_state(doc_store)
    except FileNotFoundError as e:
        print("Cannot embedd documents: you need to create a document store first")
    except Exception as e:
        print("Cannot embedd documents:", e)

    if custom_separator:
        text_splitter.SEPARATORS = custom_separator
    splitter = TextSplitter(
        split_by=split_by,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    model_kwargs = {
        "model": model_path 
    }

    transformer_embedder = AllMiniLML6V2Embedder(embedding_model, tokenizer_kwargs)
    model_client = CustomEmbeddingModelClient(transformer_embedder)
    local_embedder = Embedder(model_client=model_client,
        model_kwargs=model_kwargs
        )
    embedder_transformer = ToEmbeddings(local_embedder, batch_size=batch_size)
    data_transformer = Sequential(splitter, embedder_transformer)

    n_items = len(db.items)
    
    ## Generate and store embedding
    print(f'{"-"*10}{f"Generate embeddings for {n_items} documents."}{" "}{"-"*10}')

    key = key 
    db.transform(data_transformer, map_fn= None, key=key)
    print(f'{"-"*10}{f"Embeddings generation: done."}{" "}{"-"*10}')
    db.save_state(doc_store)
    print(f'{"-"*10}{f"Document store updated with new embeddings."}{" "}{"-"*10}')

    return n_items

if  __name__ == "__main__":

    # Files and folders
    DOC_STORE = "doc_store_002.pkl"
    MODEL_STORE = "model_store"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH = MODEL_STORE + "/" + EMBEDDING_MODEL
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"model {EMBEDDING_MODEL} is not in the model store.")

    # Embedding parameters
    BATCH_SIZE = 10
    SPLIT_BY = "passage"
    CHUNK_SIZE = 1
    CHUNK_OVERLAP = 0
    ## Tokenizer kwargs (for embedding model)
    TOKENIZER_KWARGS =  {
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "return_tensors": 'pt'
    }

    # Document Store
    KEY = "split_and_embed"

    n_items = embed(
        embedding_model=EMBEDDING_MODEL,
        model_store=MODEL_STORE,
        doc_store=DOC_STORE,
        key=KEY,
        batch_size=BATCH_SIZE,
        split_by=SPLIT_BY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer_kwargs=TOKENIZER_KWARGS
    )