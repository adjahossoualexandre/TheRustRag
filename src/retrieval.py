from typing import Any
from lightrag.core.db import LocalDB
from lightrag.core import Embedder, Document
from lightrag.core.types import RetrieverOutput
from src.models.embedding_model import CustomEmbeddingModelClient, AllMiniLML6V2Embedder
from lightrag.components.retriever.faiss_retriever import FAISSRetriever, Retriever
from lightrag.components.data_process.data_components import RetrieverOutputToContextStr
import os

def build_index(db: LocalDB, key: str, retriever: type[Retriever], retriever_kwargs=dict) -> tuple[Retriever, list[Document]]:
    transformed_documents = db.get_transformed_data(key)

    retriever = retriever(**retriever_kwargs)
    retriever.build_index_from_documents(documents=transformed_documents, document_map_func=lambda doc: doc.vector)
    return retriever, transformed_documents

def retrieve_documents(user_query: str, transformed_documents: list[Document], retriever: Retriever) -> list[RetrieverOutput]:
    # retrieve documents
    user_query = user_query
    retrieved_documents: list[RetrieverOutput] = retriever(input=user_query)

    ## fill in the outputs with texts
    for i, retriever_output in enumerate(retrieved_documents):
        retrieved_documents[i].documents = [
            transformed_documents[doc_index]
            for doc_index in retriever_output.doc_indices
        ]
    return retrieved_documents

def retrieve(
        embedding_model: str,
        model_store: str,
        doc_store: str,
        key: str,

        user_query: str,
        retriever_strategy: type, 

        model_kwargs: dict,
        retriever_kwargs: dict,
        tokenizer_kwargs: dict,
    ) -> list[RetrieverOutput]:

    model_path = model_store + "/" + embedding_model
    if not os.path.isfile(doc_store):
        print("Need to create a document store.")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model {embedding_model} is not in the model store.")

    transformer_embedder = AllMiniLML6V2Embedder(embedding_model, tokenizer_kwargs)
    model_client = CustomEmbeddingModelClient(transformer_embedder)
    local_embedder = Embedder(
        model_client=model_client,
        model_kwargs=model_kwargs
        )

    print("?"*24)
    db = LocalDB().load_state(doc_store)
    print("?"*24)

    retriever_kwargs["embedder"] = local_embedder
    retriever, transformed_documents = build_index(db, key, retriever_strategy, retriever_kwargs)

    # retrieve documents
    retrieved_documents = retrieve_documents(user_query,transformed_documents, retriever)

    return retrieved_documents

def build_context_str(retrieved_documents: list[RetrieverOutput]) -> str:
    builder = RetrieverOutputToContextStr(deduplicate=True)
    context_str = builder(retrieved_documents)
    return context_str


if __name__ == "__main__":

    # Docuemnt store
    DOC_STORE = "doc_store_002.pkl"
    KEY = "split_and_embed"

    # Embedding model
    MODEL_STORE = "model_store"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH = MODEL_STORE + "/" + EMBEDDING_MODEL

    # Query
    USER_QUERY = "explain the borrow-checker as if I was 5"

    # Retriever
    RETRIEVER_STRATEGY = FAISSRetriever

    # Keyword arguments
    model_kwargs = {
        "model": MODEL_PATH
    }
    retriever_kwargs = {
        "top_k": 2,
    }
    TOKENIZER_KWARGS =  {
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "return_tensors": 'pt'
    }

    retrieved_documents = retrieve(
        embedding_model= EMBEDDING_MODEL,
        model_store = MODEL_STORE,
        doc_store = DOC_STORE,
        key = KEY,
        user_query = USER_QUERY,
        retriever_strategy = RETRIEVER_STRATEGY,
        model_kwargs = model_kwargs,
        retriever_kwargs = retriever_kwargs,
        tokenizer_kwargs = TOKENIZER_KWARGS
    )
    context_str = build_context_str(retrieved_documents)
