import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTORSTORE_DIR = "data/vectorstore"


def build_or_load_index(ticker: str, documents):
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    path = os.path.join(VECTORSTORE_DIR, ticker.upper())

    if os.path.exists(path):
        return FAISS.load_local(path, EMBEDDINGS, allow_dangerous_deserialization=True)

    vectorstore = FAISS.from_documents(documents, EMBEDDINGS)
    vectorstore.save_local(path)
    return vectorstore
