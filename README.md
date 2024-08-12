# Desccription

A Retrieval-Augmented Generation (RAG) chatbot based on [the Rust book](https://github.com/rust-lang/book)

# Instruction

execution order:

    `pip install -r requirements.txt`
    `python file_processing.py`
    `python ingestion.py`
    `python models/local_models.py`
    `python embed_documents.py`
    `python retrieval.py`

# Reproducibility
The raw data can be found [here](https://github.com/rust-lang/book/tree/main/src). Please ignore the `ìmg/` folder. The data should be downloaded inside `chapters/raw`.

