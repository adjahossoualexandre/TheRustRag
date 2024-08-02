from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from document_metadata import (
    get_toc_info,
    get_chapter_name,
    map_chapnum_to_chapname
    )
from models import load_and_cache_embedding_model

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


if __name__ == "__main__":
    PARSED_FOLDER = "chapters/parsed/" 
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_CACHE_FOLDER = "embedding_model"
    CHUNK_SIZE = 256

    documents = SimpleDirectoryReader(PARSED_FOLDER).load_data()
    set_metadata(documents)

    embed_model = load_and_cache_embedding_model(EMBEDDING_MODEL, EMBEDDING_CACHE_FOLDER)
    service_context = ServiceContext.from_defaults(
        chunk_size=CHUNK_SIZE,
        embed_model=embed_model
        )
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[SentenceSplitter(chunk_size=CHUNK_SIZE)], # 256 is the size limit imposed by the model https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        service_context=service_context
    )
    index.storage_context.persist()