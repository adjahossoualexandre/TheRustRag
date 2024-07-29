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

from metadata import set_metadata
set_metadata(docs)

# upload model from HF

# save model locally

# Load local model

# preprocess documents

## split texts into chunks

## manage chunks/docs

## store preprocessed documents

# Generate embeddings

# Store embeddings


# Preprocess user query

# load local model

# embed user query

# retrieve documents

# generate response