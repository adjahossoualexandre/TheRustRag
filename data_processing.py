from module import (
    parse_markdown_to_text,
    read_file,
    list_files_in_folder,
    write_text_to_file
    )

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