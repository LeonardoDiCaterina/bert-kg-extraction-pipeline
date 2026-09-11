from typing import Dict, Optional


def extract_html_from_sgml(content: str) -> Optional[str]:
    """
    Extracts the primary HTML document text from an SEC EDGAR full-submission.txt SGML file.
    Avoids XBRL, UUEncoded images, and auxiliary exhibits that cause document parsers to crash.

    Returns:
        Optional[str]: Extracted HTML string, or None if no valid document text block is found.
    """
    doc_start = content.find("<DOCUMENT>")
    doc_end = content.find("</DOCUMENT>", doc_start)
    if doc_start == -1 or doc_end == -1:
        return None

    doc_block = content[doc_start:doc_end]
    text_start = doc_block.find("<TEXT>")
    text_end = doc_block.find("</TEXT>", text_start)

    if text_start == -1 or text_end == -1:
        return None

    return doc_block[text_start + 6:text_end].strip()


def resolve_company_name(
    doc_id: str,
    company_map: Optional[Dict[str, str]] = None,
    default_name: str = "The Corporation"
) -> str:
    """
    Resolves an SEC filing document ID / ticker to its canonical corporate name.

    Args:
        doc_id: Document filename or identifier (e.g. 'AAPL_2024.pdf' or 'MSFT').
        company_map: Dictionary of uppercase ticker symbol to official name.
        default_name: Fallback name if ticker is not recognized.

    Returns:
        str: Canonical company name.
    """
    if not company_map:
        company_map = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corp.",
            "AMZN": "Amazon.com, Inc.",
            "NVDA": "NVIDIA Corporation",
            "GOOGL": "Alphabet Inc.",
            "META": "Meta Platforms, Inc.",
            "TSLA": "Tesla, Inc.",
            "JPM": "JPMorgan Chase & Co.",
        }

    doc_upper = doc_id.upper()
    for ticker, full_name in company_map.items():
        if ticker in doc_upper:
            return full_name

    return default_name
