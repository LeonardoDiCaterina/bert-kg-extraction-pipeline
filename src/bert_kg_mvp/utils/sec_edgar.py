import re
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

    return doc_block[text_start + 6 : text_end].strip()


def resolve_company_name(
    doc_id: str,
    company_map: Optional[Dict[str, str]] = None,
    default_name: str = "The Corporation",
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
            "CSCO": "Cisco Systems, Inc.",
            "INTC": "Intel Corporation",
            "ORCL": "Oracle Corporation",
            "CRM": "Salesforce, Inc.",
            "AMD": "Advanced Micro Devices, Inc.",
            "QCOM": "QUALCOMM Incorporated",
        }

    doc_upper = doc_id.upper().strip()
    for ticker, full_name in company_map.items():
        if ticker in doc_upper:
            return full_name

    # If doc_id itself is a bare ticker symbol (e.g. "CSCO", "PSKY", "MDT")
    if doc_upper.isalpha() and len(doc_upper) <= 5:
        return doc_upper

    return default_name


# Canonical display names for the five targeted SEC sections
_SECTION_LABELS: Dict[str, str] = {
    "1": "Item 1 – Business",
    "1a": "Item 1A – Risk Factors",
    "7": "Item 7 – MD&A",
    "7a": "Item 7A – Market Risk",
    "8": "Item 8 – Financial Statements",
}

_SECTION_RE = re.compile(
    r"item\s+(1a|7a|1|7|8)",
    re.IGNORECASE,
)


def infer_section_label(section_hint: str) -> str:
    """
    Maps a raw section-header string (e.g. 'Item 1A. Risk Factors') to a
    canonical label such as 'Item 1A – Risk Factors'.
    Returns 'Unknown' when the string does not match any targeted section.
    """
    m = _SECTION_RE.search(section_hint)
    if m:
        key = m.group(1).lower()
        return _SECTION_LABELS.get(key, "Unknown")
    return "Unknown"


def parse_doc_metadata(doc_id: str) -> Dict[str, str]:
    """
    Extracts coarse provenance metadata from a document identifier / filepath.

    Returns a dict with keys:
        ``ticker``  – uppercased ticker symbol (e.g. 'AAPL'), or '' if unknown.
        ``year``    – 4-digit filing year string (e.g. '2024'), or '' if absent.
        ``section`` – canonical section label (e.g. 'Item 7 – MD&A'), or ''
                      when no section hint is embedded in the id.

    The ``section`` field is intentionally left empty here because the section
    is discovered *during* document parsing (not from the id alone).  The
    caller (``parse_sec_filings``) is expected to fill it in after detection.
    """
    doc_upper = doc_id.upper()

    # --- Ticker ---
    ticker = ""
    # Try sec-edgar-filings/<TICKER>/... path structure first
    if "SEC-EDGAR-FILINGS/" in doc_upper:
        parts = doc_upper.split("SEC-EDGAR-FILINGS/")[1].split("/")
        if parts and parts[0]:
            ticker = parts[0]
    else:
        # Fall back: first all-caps word segment up to 5 characters
        m = re.match(
            r"([A-Z]{1,5})", doc_upper.replace("-", "").replace("_", " ").strip()
        )
        if m:
            ticker = m.group(1)

    # --- Year ---
    year = ""
    # Check for SEC EDGAR accession number format: CIK-YY-Seq (e.g. 0000320193-24-000106)
    acc_m = re.search(r"\d{10}-(\d{2})-\d{6}", doc_id)
    if acc_m:
        yy = int(acc_m.group(1))
        year = str(2000 + yy if yy < 50 else 1900 + yy)
    else:
        # Match standalone 4-digit year bounded by non-digits
        year_m = re.search(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)", doc_id)
        if year_m:
            year = year_m.group(1)

    return {"ticker": ticker, "year": year, "section": ""}
