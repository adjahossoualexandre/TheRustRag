from typing import Any
from src.retrieval import retrieve, build_context_str
from src.embed_documents import embed
from lightrag.components.retriever.faiss_retriever import FAISSRetriever
import os

from src.utils import ManualExperiment

# MLFLow
import mlflow
from mlflow import log_artifact, log_param, set_experiment
from src.mlflow_utils import get_mlflow_experiment
from dotenv import load_dotenv



if __name__ == "__main__":

    load_dotenv()
    # Metadata file constante
    FILE_NAME = "second AdalFlow (lightrag) retrieval.txt"

    # Experiment constants
    MLFLOW_URI = os.environ["MLFLOW_URI"]
    EXPERIMENT_NAME="Debug chunking"
    PARENT_RUN_NAME="custom separator: \\n\\n\\n"
    EMBEDDING_RUN_DESCRIPTION = ""
    RETRIEVAL_RUN_DESCRIPTION = ""

    # Shared constants
    DOC_STORE = "doc_store_001.pkl"
    KEY = "split_and_embed"
    MODEL_STORE = "model_store"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOKENIZER_KWARGS =  {
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "return_tensors": 'pt'
    }

    # Embedding constants
    BATCH_SIZE = 10
    SPLIT_BY = "custom_passage_separator"
    CUSTOM_SEPARATOR = {"custom_passage_separator": "\n\n\n"}
    CHUNK_SIZE = 1
    CHUNK_OVERLAP = 0

    # Retrieval constants
    #### Files and folders
    MODEL_PATH = MODEL_STORE + "/" + EMBEDDING_MODEL
    USER_QUERY = "explain the borrow-checker as if I was 5"
    RETRIEVER_STRATEGY = FAISSRetriever
    model_kwargs = {
        "model": MODEL_PATH
    }
    retriever_kwargs = {
        "top_k": 2,
    }
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"model {EMBEDDING_MODEL} is not in the model store.")



    n_items = embed(
        embedding_model=EMBEDDING_MODEL,
        model_store=MODEL_STORE,
        doc_store=DOC_STORE,
        key=KEY,
        batch_size=BATCH_SIZE,
        split_by=SPLIT_BY,
        custom_separator=CUSTOM_SEPARATOR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer_kwargs=TOKENIZER_KWARGS
    )

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

    # Set tracking URI to your Heroku application
    mlflow.set_tracking_uri(MLFLOW_URI)
    ml_flow_experiment = get_mlflow_experiment(experiment_name=EXPERIMENT_NAME)

    set_experiment(experiment_name=EXPERIMENT_NAME)
    # PARENT RUN
    with mlflow.start_run(
        run_name = PARENT_RUN_NAME,
        experiment_id = ml_flow_experiment.experiment_id,
        description= EMBEDDING_RUN_DESCRIPTION,
        tags={"parent": "True"}
        ) as parent_run:

    # CHILD RUN 1: Embed documents
        with mlflow.start_run(
            run_name= "embed documents",
            nested=True,
            parent_run_id=parent_run.info.run_id,
            tags={"parent": "False"}
            ):

            log_param("doc_store", DOC_STORE)
            log_param("doc_store_key", KEY)
            log_param("nb_items", n_items)
            log_param("emb_model", EMBEDDING_MODEL)
            log_param("emb_batch_size", BATCH_SIZE)
            log_param("splitter_split_by", SPLIT_BY)
            log_param("split_chunk_size", CHUNK_SIZE)
            log_param("split_chunk_overlap", CHUNK_OVERLAP)
            for key, val in TOKENIZER_KWARGS.items():
                name = "emb_tok" + "_" + key
                log_param(name, val)
    
    # CHILD RUN 2: retrieval
        with mlflow.start_run(
            run_name = "retrieve documents",
            experiment_id = ml_flow_experiment.experiment_id,
            nested=True,
            parent_run_id=parent_run.info.run_id,
            description=RETRIEVAL_RUN_DESCRIPTION
            ) as run:
                
            retrieval_run_metadata = {
                "llm": None,
                "local_storage_path": f"manual_tracking/{FILE_NAME}",
                "chunk_size(nb of sentence)": 1,
                "chunk_overlap(nb of sentence)": 1,
                "user_query": USER_QUERY
            }
            experiment_results = ManualExperiment(FILE_NAME, "manual_tracking", retrieval_run_metadata)

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
                experiment_results.track(output)


                log_param("user_query", USER_QUERY)
                log_param("retriever", RETRIEVER_STRATEGY.__name__)
                for key, val in retriever_kwargs.items():
                    param = "retr" + "_" + key
                    log_param(param, retriever_kwargs[key])

                log_artifact(experiment_results.file_path)