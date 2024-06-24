from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from metadata import (
    get_toc_info,
    get_chapter_name,
    map_chapnum_to_chapname
    )
from models import load_and_cache_embedding_model
import pickle

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
    
    nodes = SentenceSplitter(chunk_size=CHUNK_SIZE) \
        .get_nodes_from_documents(documents)
    
    # Load model
    embed_model = load_and_cache_embedding_model(EMBEDDING_MODEL, EMBEDDING_CACHE_FOLDER)

    # Embed chunks and store in SimpleVectorStore
    vector_store = SimpleVectorStore()
    for node in nodes:
        node.embedding = embed_model.get_text_embedding(node.get_content())
        vector_store.add(node.embedding, node.get_doc_id(), node.get_content())

    # Save vector store
    vector_store.persist()
