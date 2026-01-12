from langchain_core.tools import tool
from src.ingestion.sec_downloader import download_latest_10k
from src.ingestion.processor import process_filing
from src.ingestion.indexer import build_or_load_index

@tool
def retrieve_company_documents(ticker: str, query: str) -> str:
    """
    Automatically retrieves and indexes the latest SEC 10-K filing for a given company ticker 
    and returns relevant document context using vector-based semantic search (RAG).
    """
    try:
        filepath = download_latest_10k(ticker)
        docs = process_filing(filepath)
        vectorstore = build_or_load_index(ticker, docs)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        results = retriever.get_relevant_documents(query)
        if not results:
            return "No relevant information found in company filings."
        
        return "\n\n".join([f"From 10-K: {doc.page_content[:800]}" for doc in results])
    except Exception as e:
        return f"RAG_UNAVAILABLE: {str(e)}"
