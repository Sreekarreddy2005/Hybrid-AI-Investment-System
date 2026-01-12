import os
import requests
import time


SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {
    "User-Agent": "Hybrid-AI-Investment-System (academic project, contact: student@example.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}


def safe_get_json(url: str):
    response = requests.get(url, headers=HEADERS, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(f"SEC request failed ({response.status_code}) for {url}")

    try:
        return response.json()
    except Exception:
        raise RuntimeError("SEC returned non-JSON response (rate limited or blocked)")


def get_cik_for_ticker(ticker: str) -> str:
    data = safe_get_json(SEC_TICKER_CIK_URL)

    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)

    raise RuntimeError(f"CIK not found for ticker {ticker}")


def download_latest_10k(ticker: str, save_dir="data/filings") -> str:
    os.makedirs(save_dir, exist_ok=True)

    cik = get_cik_for_ticker(ticker)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    submissions = safe_get_json(submissions_url)
    filings = submissions.get("filings", {}).get("recent", {})

    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accession_numbers[i].replace("-", "")
            primary_doc = primary_docs[i]

            file_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession}/{primary_doc}"
            )

            time.sleep(0.5)  # SEC rate-limit safety
            response = requests.get(file_url, headers=HEADERS, timeout=10)

            if response.status_code != 200:
                raise RuntimeError("Failed to download 10-K document")

            file_path = os.path.join(save_dir, f"{ticker.upper()}_10K.html")
            with open(file_path, "wb") as f:
                f.write(response.content)

            return file_path

    raise RuntimeError(f"No 10-K filing found for {ticker}")
