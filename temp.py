import os
from lightrag.core.types import Document

# Import docs
## Make this a function that ingest multiple files into Documents objects.

## Open Json files

def read_file(file_path: str, mode: str="r") -> str:
    with open(file_path, mode=mode) as f:
        file = f.read()
    return file


def list_files_in_folder(folder_path, file_extension=None):
    """
    Returns a list of file names in the specified folder, optionally filtered by file extension.

    Args:
    folder_path (str): Path to the folder.
    file_extension (str, optional): File extension to filter by (e.g., '.txt'). Defaults to None.

    Returns:
    list: List of file names.
    """
    # Check if the folder path exists and is a directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a valid directory.")
    
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print("-"*24, len(files))
    # If file_extension is specified, filter the files by that extension
    if file_extension:
        files = [f for f in files if f.endswith(file_extension)]
    
    return files

def read_all_files(folder, file_extension) -> list[dict]:
    files = []
    file_names = list_files_in_folder(folder_path=folder, file_extension=file_extension)
    for file_name in file_names:
        file_path = os.path.join(folder, file_name)
        content = read_file(file_path)

        files.append(
           dict(
                text=content,
                meta_data={"file_name":file_name}
                )
        )
    return files

def convert_file_to_documents(files: list) -> list:

    documents = [
        Document(
            text=file["text"],
            meta_data=file["meta_data"]
            ) for file in files]
    
    return documents

folder = "chapters/parsed"
files = read_all_files(folder, ".md")
docs = convert_file_to_documents(files)

from document_metadata import set_metadata
set_metadata(docs)

# Download model from HF
from models import load_from_HuggingFace, save_model

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "persist_temp"

model, tokenizer = load_from_HuggingFace(MODEL_NAME, MODEL_NAME)
save_model(model, tokenizer, PERSIST_DIR)

## Configure the splitter settings
from lightrag.components.data_process.text_splitter import TextSplitter

text_splitter = TextSplitter(
    split_by="passage",
    chunk_size=1,
    chunk_overlap=0
)

# Generate embeddings
from models import CustomModelClient, AllMiniLML6V2Embedder
from lightrag.core import Embedder, Sequential
from lightrag.components.data_process import ToEmbeddings

MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "persist_temp/"
BATCH_SIZE = 10

model_kwargs = {
    "model": MODEL_PATH
}

transformer_embedder = AllMiniLML6V2Embedder(MODEL_PATH)
model_client = CustomModelClient(transformer_embedder)
local_embedder = Embedder(model_client=model_client,
    model_kwargs=model_kwargs
    )
embedder_transformer = ToEmbeddings(local_embedder, batch_size=BATCH_SIZE)
data_transformer = Sequential(text_splitter, embedder_transformer)

## store preprocessed documents
from lightrag.core.db import LocalDB
db = LocalDB()
db.load(docs)
#d = data_transformer(docs)

## Generate and store embedding
key = "split_and_embed"
db.transform(data_transformer, map_fn= None, key=key)
transformed_documents = db.get_transformed_data(key)

# Retriever
#from lightrag.components.retriever import FAISSRetriever <- for some reason it doesn't work. see with Li Yin bc it is the code from documentation https://lightrag.sylph.ai/tutorials/retriever.html#faissretriever
from lightrag.components.retriever.faiss_retriever import FAISSRetriever
retriever = FAISSRetriever(top_k=2, embedder=local_embedder)
retriever.build_index_from_documents([doc.vector for doc in transformed_documents])

# retrieve documents
retrieved_documents = retriever(input="explain the borrow-checker as if I was 5")
## fill in the document
for i, retriever_output in enumerate(retrieved_documents):
    retrieved_documents[i].documents = [
        transformed_documents[doc_index]
        for doc_index in retriever_output.doc_indices
    ]
# generate response
## convert all the documents to context string
from lightrag.components.data_process.data_components import RetrieverOutputToContextStr

retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)
context_str = retriever_output_processors(retrieved_documents)
print(context_str)

## Generator