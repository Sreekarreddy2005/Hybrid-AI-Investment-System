from langchain_core.tools import tool
from src.rag.loader import load_and_split_docs
from src.rag.vectorstore import create_vectorstore

FILE_PATH = "data/documents/TSLA_10K_2024.html"


@tool
def retrieve_company_documents(query: str) -> str:
    """
    Retrieves relevant context from company financial documents
    using vector-based semantic search (RAG).
    """
    docs = load_and_split_docs(FILE_PATH)
    vectorstore = create_vectorstore(docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = retriever.get_relevant_documents(query)

    return "\n\n".join(doc.page_content for doc in results)
