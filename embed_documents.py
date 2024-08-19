## Configure the splitter settings
from lightrag.components.data_process.text_splitter import TextSplitter


# Generate embeddings
from models.embedding_model import CustomEmbeddingModelClient, AllMiniLML6V2Embedder
from lightrag.core import Embedder, Sequential
from lightrag.components.data_process import ToEmbeddings

from lightrag.core.db import LocalDB
import os

# MLFLow
import mlflow
from mlflow import log_param 
from mlflow_utils import create_mlflow_experiment, get_mlflow_experiment
from dotenv import load_dotenv



if __name__ == "__main__":
    # Set tracking URI to your Heroku application
    load_dotenv()

    MLFLOW_URI = os.environ["MLFLOW_URI"]

    EXPERIMENT_NAME="Debug chunking"
    PARENT_RUN_NAME="embed + retrieve"
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Experiment settings
    run_descrition = ""
    experiment_id = create_mlflow_experiment(EXPERIMENT_NAME)
    experiment = get_mlflow_experiment(experiment_id)

    # Files and folders
    DOC_STORE = "doc_store_001.pkl"
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
    tokenizer_kwargs =  {
        "max_length": 512,
        "padding": True,
        "truncation": True,
        "return_tensors": 'pt'
    }

    # Document Store
    KEY = "split_and_embed"


    try:
        db = LocalDB().load_state(DOC_STORE)
    except FileNotFoundError as e:
        print("Cannot embedd documents: you need to create a document store first")
    except Exception as e:
        print("Cannot embedd documents:", e)

    text_splitter = TextSplitter(
        split_by=SPLIT_BY,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    model_kwargs = {
        "model": EMBEDDING_MODEL
    }

    transformer_embedder = AllMiniLML6V2Embedder(EMBEDDING_MODEL, tokenizer_kwargs)
    model_client = CustomEmbeddingModelClient(transformer_embedder)
    local_embedder = Embedder(model_client=model_client,
        model_kwargs=model_kwargs
        )
    embedder_transformer = ToEmbeddings(local_embedder, batch_size=BATCH_SIZE)
    data_transformer = Sequential(text_splitter, embedder_transformer)

    n_items = len(db.items)
    
    ## Generate and store embedding
    print(f'{"-"*10}{f"Generate embeddings for {n_items} documents."}{" "}{"-"*10}')

    key = KEY
    db.transform(data_transformer, map_fn= None, key=key)
    print(f'{"-"*10}{f"Embeddings generation: done."}{" "}{"-"*10}')
    db.save_state(DOC_STORE)
    print(f'{"-"*10}{f"Document store updated with new embeddings."}{" "}{"-"*10}')

    with mlflow.start_run(
        run_name = PARENT_RUN_NAME,
        experiment_id = experiment.experiment_id,
        description= run_descrition,
        tags={"parent": "True"}
        ) as run:
        run.info.run_id
        with mlflow.start_run(run_name= "embed documents", nested=True):

            log_param("doc_store", DOC_STORE)
            log_param("doc_store_key", KEY)
            log_param("nb_items", n_items)
            log_param("emb_model", EMBEDDING_MODEL)
            log_param("emb_batch_size", BATCH_SIZE)
            log_param("splitter_split_by", SPLIT_BY)
            log_param("split_chunk_size", CHUNK_SIZE)
            log_param("split_chunk_overlap", CHUNK_OVERLAP)
            for key, val in tokenizer_kwargs.items():
                name = "emb_tok" + "_" + key
                log_param(name, val)
