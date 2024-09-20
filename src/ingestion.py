import os
from lightrag.core.types import Document


def read_file(file_path: str, mode: str="r") -> str:
    with open(file_path, mode=mode) as f:
        file = f.read()
    return file

def list_files_in_folder(folder_path: str, file_extension: str =None) -> list[str | bytes]:
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

def read_all_files(folder: str, file_extension: str, exclude: list[str] = None) -> list[dict[str, str]]:
    files = []
    file_names = list_files_in_folder(folder_path=folder, file_extension=file_extension)
    if exclude:
        print("-"*10, "exclude files:", ", ".join(exclude), "-"*10)
        file_names = [file_name for file_name in file_names if file_name not in exclude]
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

def convert_file_to_documents(files: list[dict]) -> list[Document]:

    documents = [
        Document(
            text=file["text"],
            meta_data=file["meta_data"]
            ) for file in files]
    
    return documents

def ingest_documents(folder_path: str, file_extension: str = None, exclude: list[str] = None) -> list[Document]:
    files = read_all_files(folder_path, file_extension, exclude=exclude)
    documents = convert_file_to_documents(files)
    return documents

if __name__ == "__main__":
    from lightrag.core.db import LocalDB
    from document_metadata import set_metadata

    PARSED_DATA_FOLDER = "chapters/parsed"
    EXTENSION = ".md"
    DOC_STORE = "doc_store_002.pkl"
    EXCLUDE = ["SUMMARY.md"]

    db = LocalDB().load_state(DOC_STORE) if os.path.isfile(DOC_STORE) else LocalDB()
    db_init_size = len(db.items)
    
    print(f'{"-"*10}{" "}{db_init_size} documents in the document store {DOC_STORE}.{" "}{"-"*10}')
    print(f'Ingest new documents.')

    documents = ingest_documents(PARSED_DATA_FOLDER, EXTENSION, EXCLUDE)
    set_metadata(documents)
    n_new_docs = len(documents)
    db.load(documents)
    db.save_state(DOC_STORE)
    print(f'{"-"*10}{" "}Ingested {n_new_docs} new documents.{" "}{"-"*10}')

