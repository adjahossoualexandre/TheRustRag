from models.embedding_model import CustomEmbeddingModelClient
from lightrag.core import Embedder
from sentence_transformers.util import pytorch_cos_sim

def parse_embedding(embedding):
    return embedding.data[0].embedding

def get_scalar(similarity_score) -> float:
    return similarity_score.numpy()[0][0].tolist()

def get_sentence_similarities(sentence_ref: str, sentence_to_compare_to: dict, embedder: Embedder) -> None:
    embedding_ref = embedder(sentence_ref)

    parsed_embedding_ref = parse_embedding(embedding_ref)
    for sentence in sentence_to_compare_to.keys():
        embedding = embedder(sentence_to_compare_to[sentence]["text"])
        parsed_embedding = parse_embedding(embedding)
        sim_to_sentence_ref = pytorch_cos_sim(parsed_embedding_ref, parsed_embedding)
        sentence_to_compare_to[sentence]["similarity_score"] = get_scalar(sim_to_sentence_ref)

def test_CustomModelClient(sentence_to_compare_to: dict) -> None:
    for name, dic in sentence_to_compare_to.items():
        print("-"*24, name, "-"*24)
        rounded_score = round(dic["similarity_score"], 3)
        assert rounded_score == dic["ground_truth"]
        print("test passed.")
        

if __name__ =="__main__":
    MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
    TOKENIZER_PATH = "sentence-transformers/all-MiniLM-L6-v2"
    PERSIST_DIR = "persist_temp/"

    # test data. reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    sentence_ref = "That is a happy person"
    sentence_to_compare_to = {
        "sentence_1": {
            "ground_truth": 0.695,
            "text": "That is a happy dog"
        },
        "sentence_2": {
            "ground_truth": 0.943,
            "text": "That is a very happy person"
        },
        "sentence_3": {
            "ground_truth": 0.257,
            "text": "Today is a sunny day"
        }
    }

    # load embedding model
    model_kwargs = {
        "model": MODEL_PATH
    }
    local_embedder = Embedder(model_client=CustomEmbeddingModelClient(), model_kwargs=model_kwargs)

    # run test
    get_sentence_similarities(sentence_ref, sentence_to_compare_to, local_embedder)
    test_CustomModelClient(sentence_to_compare_to)
