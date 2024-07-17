from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.llms.huggingface import HuggingFaceLLM
import pickle

# Constants
PARSED_FOLDER = "chapters/parsed/" 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_FOLDER = "embedding_model"
CHUNK_SIZE = 256

# Load and split documents
documents = SimpleDirectoryReader(PARSED_FOLDER ).load_data()
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE)
nodes = splitter.get_nodes_from_documents(documents)

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# Persistence
## Embed chunks and store in SimpleVectorStore
vector_store = SimpleVectorStore()
for node in nodes:
    node.embedding = embed_model.get_text_embedding(node.get_content())
vector_store.add(nodes)
vector_store.data
vector_store.persist("vector_store.json") # try that

## Doc store
doc_store = SimpleDocumentStore()
doc_store.add_documents(nodes)
doc_store.persist("doc_store.json")

## Index index_store
index_store = SimpleIndexStore()
index_store.index_structs()
index_store.persist("storage_bis/index_store.json")


# Need to save docstore as well
# Need to save indexstore as well


# Initialize LLM
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    model_name="path/to/your/local/llm_model",
    tokenizer_name="path/to/your/local/llm_tokenizer",
    device_map="auto",
)

def retrieve_similar_chunks(query, vector_store, embed_model, top_k=5):
    query_embedding = embed_model.get_text_embedding(query)
    results = vector_store.similarity_search(query_embedding, top_k=top_k)
    return results

def generate_response(query, similar_chunks, llm):
    context = "\n".join([chunk[1] for chunk in similar_chunks])  # chunk[1] is the content
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.complete(prompt)
    return response.text

# Function to handle queries
def process_query(query, vector_store, embed_model, llm):
    similar_chunks = retrieve_similar_chunks(query, vector_store, embed_model)
    response = generate_response(query, similar_chunks, llm)
    return response

# Example usage
query = "Your query here"
answer = process_query(query, vector_store, embed_model, llm)
print(answer)