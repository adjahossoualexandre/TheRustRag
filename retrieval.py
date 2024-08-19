from typing import Any
from lightrag.core.db import LocalDB
from lightrag.core import Embedder, Document
from lightrag.core.types import RetrieverOutput
from models.embedding_model import CustomEmbeddingModelClient, AllMiniLML6V2Embedder
from lightrag.components.retriever.faiss_retriever import FAISSRetriever, Retriever
from lightrag.components.data_process.data_components import RetrieverOutputToContextStr
import os

# MLFLow
import mlflow
from mlflow import log_artifact, log_param 
from mlflow_utils import create_mlflow_experiment, get_mlflow_experiment
from dotenv import load_dotenv


def build_index(db: LocalDB, key: str, retriever: type[Retriever], retriever_kwargs=dict) -> tuple[Retriever, list[Document]]:
    transformed_documents = db.get_transformed_data(key)

    retriever = retriever(**retriever_kwargs)
    retriever.build_index_from_documents(documents=transformed_documents, document_map_func=lambda doc: doc.vector)
    return retriever, transformed_documents

def retrieve_documents(user_query: str, retriever: Retriever):
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

if __name__ == "__main__":

    RUN_EXPERIMENT = False 

    # Docuemnt store
    DOC_STORE = "doc_store_001.pkl"
    KEY = "split_and_embed"

    # Embedding model
    MODEL_STORE = "model_store"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH = MODEL_STORE + "/" + EMBEDDING_MODEL

    # Query
    USER_QUERY = "explain the borrow-checker as if I was 5"

    # Keyword arguments
    model_kwargs = {
        "model": MODEL_PATH
    }
    retriever_kwargs = {
        "top_k": 2,
    }

    if not os.path.isfile(DOC_STORE):
        print("Need to create a document store.")
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"model {EMBEDDING_MODEL} is not in the model store.")


    transformer_embedder = AllMiniLML6V2Embedder(EMBEDDING_MODEL)
    model_client = CustomEmbeddingModelClient(transformer_embedder)
    local_embedder = Embedder(
        model_client=model_client,
        model_kwargs=model_kwargs
        )
    retriever_strategy: type = FAISSRetriever
    build_context_str = RetrieverOutputToContextStr(deduplicate=True)



    print("?"*24)
    db = LocalDB().load_state(DOC_STORE)
    print("?"*24)

    retriever_kwargs["embedder"] = local_embedder
    retriever, transformed_documents = build_index(db, KEY, retriever_strategy, retriever_kwargs)

    # retrieve documents
    retrieved_documents = retrieve_documents(USER_QUERY, retriever)
    context_str = build_context_str(retrieved_documents)
    print(context_str)


    if RUN_EXPERIMENT:

        # Set tracking URI to your Heroku application
        load_dotenv()

        MLFLOW_URI = os.environ["MLFLOW_URI"]

        EXPERIMENT_NAME="Debug chunking"
        mlflow.set_tracking_uri(MLFLOW_URI)


        run_descrition = "this is a test."
        ml_flow_experiment = get_mlflow_experiment(experiment_name=EXPERIMENT_NAME)

        from utils import ManualExperiment

        experiment_metadata = {
            "llm": None,
            "storage_path": "manual_tracking/first AdalFlow (lightrag) retrieval",
            "chunk_size(nb of sentence)": 1,
            "chunk_overlap(nb of sentence)": 1,
            "user_query": USER_QUERY
        }
        experiment = ManualExperiment("first AdalFlow (lightrag) retrieval", "manual_tracking", experiment_metadata)

        #retrieve documents metadata
        docS = retrieved_documents[0]
        for doc_pos in range(len(docS.doc_indices)):
            doc = docS.documents[doc_pos]
            output = dict(    
                id_ = doc.id,
                scores = docS.doc_scores[doc_pos],
                file_name = doc.meta_data["file_name"],
                chapter_number = doc.meta_data["chapter_number"],
                subsection_name = doc.meta_data["subsection_name"],
                subsection_number = doc.meta_data["subsection_number"],
                parent_doc_id = doc.parent_doc_id,
                text = doc.text
            )
            experiment.track(output)

        filter_str = """
        tags.parent = "True"
        """
        runs_in_exp = mlflow.search_runs(
            filter_string=filter_str,
            experiment_ids=ml_flow_experiment.experiment_id,
            order_by=["attributes.created DESC"],
            output_format="list"      
            )
        parent_run = runs_in_exp[0]

        with mlflow.start_run(
            run_name = "retrieve documents",
            experiment_id = ml_flow_experiment.experiment_id,
            nested=True,
            parent_run_id=parent_run.info.run_id,
            ) as run:
            log_param("user_query", USER_QUERY)
            log_param("retriever", retriever_strategy)
            for key, val in retriever_kwargs.items():
                param = "retr" + "_" + key
                log_param(param, retriever_kwargs[key])

            log_artifact(experiment.file_path)