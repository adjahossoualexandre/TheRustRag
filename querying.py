from llama_index.core import load_index_from_storage, StorageContext, ServiceContext
from llama_index.core import Settings
from models import load_local_embedding_model

class DocumentRetriever():

    def __init__(self, embed_model, index_folder, chunk_size, llm=None) -> None:
        self.embed_model = embed_model
        self.index_folder = index_folder
        self.chunk_size = chunk_size
        self.llm = llm
        self.index = None
        
    def load_model_and_index(self) -> None:

        embed_model = load_local_embedding_model(self.embed_model)
        storage_context = StorageContext.from_defaults(persist_dir=self.index_folder)
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            chunk_size=self.chunk_size,
            embed_model=embed_model
            )
        self.index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

    def retrieve_top_k(self, user_query, similarity_top_k=1):
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)

        nodes = retriever.retrieve(user_query)
        return nodes

if __name__ == "__main__":
    LLM = None
    EMBEDDING_MODEL = "embedding_model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a"
    STORAGE_PATH = "storage"
    CHUNK_SIZE = 256

    user_query = "explain the borrow checker like I was 5"

    retriever = DocumentRetriever(EMBEDDING_MODEL, STORAGE_PATH, CHUNK_SIZE, LLM)
    retriever.load_model_and_index()
    nodes = retriever.retrieve_top_k(user_query, 3)


    from utils import ManualExperiment
    metadata = {
        "llm": LLM,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "STORAGE_PATH": STORAGE_PATH,
        "CHUNK_SIZE": CHUNK_SIZE,
        "user_query": user_query
    }
    experiment = ManualExperiment(
        file_name="very_first_retrieval_try.txt",
        dir="/workspaces/TheRustRag/manual_tracking",
        exp_metadata=metadata
    )
    experiment.file_path

    for n in nodes:
        output = {
            "chapter_number":  n.metadata["chapter_number"],
            "chapter_name":  n.metadata["chapter_name"],
            "subsection_number":  n.metadata["subsection_number"],
            "subsection_name":  n.metadata["subsection_name"],
            "text": n.text
        }
        experiment.track(output)
