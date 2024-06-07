from markdown import markdown
from bs4 import BeautifulSoup
import os
import re

############################## Parse raw data

def read_file(file_path: str, mode: str="r") -> str:
    with open(file_path, mode=mode) as f:
        file = f.read()
    return file

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


############################## extract metadata from Documents for indexing

def get_toc_info(document, first_n_char=7, chapter_name_pattern="^ch\d+"):
    file_name: str = document.metadata["file_name"]

    # retrieve substring with section/subsection info
    chapter_info = "-".join(file_name.split("-")[:2])
    chapter_info = chapter_info.rstrip("md").rstrip(".")
    chapter_number, subsection_number = chapter_info.split("-") if "-" in chapter_info else (chapter_info.split(".")[0], None)
    if bool(re.match(chapter_name_pattern, file_name)):
        chapter_number = int(chapter_number[2:4])
        subsection_name = file_name[first_n_char + 1:].rstrip("md").rstrip(".")
    else:
        subsection_name = file_name.split("-")[-1].rstrip("md").rstrip(".") if subsection_number != None else None

    return {
        "chapter_number": chapter_number,
        "subsection_name": subsection_name,
        "subsection_number": int(subsection_number) if subsection_number != None and subsection_number.isnumeric() else None
    }

def map_chapnum_to_chapname(documents) -> dict:
    chapters = dict()
    for doc in documents:
        chapter_number: str = doc.metadata["chapter_number"]
        subsection_number = doc.metadata["subsection_number"]

        if isinstance(chapter_number, str):
            chapters[chapter_number] = chapter_number
        if subsection_number == 0:
            chapters[chapter_number] =  doc.metadata["subsection_name"]

    return chapters

def get_chapter_name(document, chapters: dict):
    chapter_number: str | int = document.metadata["chapter_number"]
    chapter_name = chapters[chapter_number]

    return chapter_name