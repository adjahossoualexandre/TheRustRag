from ingestion import ingest_documents
from document_metadata import set_metadata

folder = "chapters/parsed"
extension = ".md"
docs = ingest_documents(folder, extension)

set_metadata(docs)

# Download model from HF
from models.model_utils import load_from_HuggingFace, save_model

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "persist_temp"

model, tokenizer = load_from_HuggingFace(MODEL_NAME, MODEL_NAME)
save_model(model, tokenizer, PERSIST_DIR)

## Configure the splitter settings
from lightrag.components.data_process.text_splitter import TextSplitter

text_splitter = TextSplitter(
    split_by="passage",
    chunk_size=1,
    chunk_overlap=0
)

# Generate embeddings
from models.embedding_model import CustomEmbeddingModelClient, AllMiniLML6V2Embedder
from lightrag.core import Embedder, Sequential
from lightrag.components.data_process import ToEmbeddings

MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "persist_temp/"
BATCH_SIZE = 10

model_kwargs = {
    "model": MODEL_PATH
}

transformer_embedder = AllMiniLML6V2Embedder(MODEL_PATH)
model_client = CustomEmbeddingModelClient(transformer_embedder)
local_embedder = Embedder(model_client=model_client,
    model_kwargs=model_kwargs
    )
embedder_transformer = ToEmbeddings(local_embedder, batch_size=BATCH_SIZE)
data_transformer = Sequential(text_splitter, embedder_transformer)

## store preprocessed documents
from lightrag.core.db import LocalDB
db = LocalDB()
db.load(docs)
#d = data_transformer(docs)

## Generate and store embedding
key = "split_and_embed"
db.transform(data_transformer, map_fn= None, key=key)
transformed_documents = db.get_transformed_data(key)

# Retriever
#from lightrag.components.retriever import FAISSRetriever <- for some reason it doesn't work. see with Li Yin bc it is the code from documentation https://lightrag.sylph.ai/tutorials/retriever.html#faissretriever
from lightrag.components.retriever.faiss_retriever import FAISSRetriever
retriever = FAISSRetriever(top_k=2, embedder=local_embedder)
retriever.build_index_from_documents([doc.vector for doc in transformed_documents])

# retrieve documents
user_query = "explain the borrow-checker as if I was 5"
retrieved_documents = retriever(input=user_query)

## fill in the document
for i, retriever_output in enumerate(retrieved_documents):
    retrieved_documents[i].documents = [
        transformed_documents[doc_index]
        for doc_index in retriever_output.doc_indices
    ]
# generate response
## convert all the documents to context string
from lightrag.components.data_process.data_components import RetrieverOutputToContextStr

retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)
context_str = retriever_output_processors(retrieved_documents)
print(context_str)

from utils import ManualExperiment

experiment_metadata = {
    "llm": None,
    "storage_path": "manual_tracking/first AdalFlow (lightrag) retrieval",
    "chunk_size(nb of sentence)": 1,
    "chunk_overlap(nb of sentence)": 1,
    "user_query": user_query
}
experiment = ManualExperiment("first AdalFlow (lightrag) retrieval", "manual_tracking")

#retrieve documents metadata
docS = retrieved_documents[0]
for doc_pos in range(len(docS.doc_indices)):
    doc = docS.documents[doc_pos]
    output = dict(    
        id_ = doc.id,
        scores = docS.doc_scores[doc_pos],
        file_name = doc.meta_data["file_name"],
        chapter_number = doc.meta_data["chapter_number"],
        subsection_name = doc.meta_data["subsection_name"],
        subsection_number = doc.meta_data["subsection_number"],
        parent_doc_id = doc.parent_doc_id,
        text = doc.text
    )
    experiment.track(output)
## Generator