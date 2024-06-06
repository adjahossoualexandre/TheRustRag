from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from module import (
    get_toc_info,
    get_chapter_name,
    map_chapnum_to_chapname
    )
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

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



documents = SimpleDirectoryReader("chapters/parsed/").load_data()

set_metadata(documents)

