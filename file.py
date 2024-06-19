from markdown import markdown
from bs4 import BeautifulSoup
import os

############################## Parse raw data

def read_file(file_path: str, mode: str="r") -> str:
    with open(file_path, mode=mode) as f:
        file = f.read()
    return file

def write_text_to_file(file_path, text):
    """
    Writes the given text to a new file.

    Args:
    file_path (str): Path to the new file.
    text (str): Text to write into the file.
    """
    try:
        with open(file_path, 'w') as file:
            file.write(text)
        print(f"Text successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def parse_markdown_to_text(file: str) -> str:
    html = markdown(file)
    page_elements = BeautifulSoup(html, "html.parser").findAll(text=True)
    text = ''.join(page_elements)
    return text


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

def parse_all_pages(raw_folder, parsed_folder, file_extension=None):
    pages = list_files_in_folder(raw_folder, file_extension)
    for file_name in pages:
        page_path = raw_folder + file_name
        page = read_file(page_path, "r")
        page_parsed = parse_markdown_to_text(page)
        destination = parsed_folder + file_name
        write_text_to_file(destination, page_parsed)
        print("-"*24, file_name, "parsed.")

if __name__ == "__main__":
    RAW_FOLDER = "chapters/raw/"
    PARSED_FOLDER = "chapters/parsed/" 
    FILE_EXTENSION = ".md"
    parse_all_pages(RAW_FOLDER, PARSED_FOLDER, FILE_EXTENSION)