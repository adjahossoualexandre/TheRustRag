import re

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