from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from module import (
    get_toc_info,
    get_chapter_name,
    map_chapnum_to_chapname
    )
import re

documents = SimpleDirectoryReader("chapters/parsed/").load_data()

for doc in documents:
    infos = get_toc_info(doc)
    for key,value in infos.items():
        doc.metadata[key] = value
chapters = map_chapnum_to_chapname(documents)

for doc in documents:
    doc.metadata["chapter_name"] = get_chapter_name(doc, chapters)

def set_metadata(documents):

    # Get chapters number and subsections names & numbers
    for document in documents:
        infos = get_toc_info(document)
        for key,value in infos.items():
            document.metadata[key] = value
    # Get chapters names    
    chapters = map_chapnum_to_chapname(documents)
    for doc in documents:
        doc.metadata["chapter_name"] = get_chapter_name(doc, chapters)


index = VectorStoreIndex.from_documents(
    documents, transformations=[SentenceSplitter(chunk_size=512)]
)