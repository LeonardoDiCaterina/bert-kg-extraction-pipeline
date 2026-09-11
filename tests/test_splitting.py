import pandas as pd

from bert_kg_mvp.utils.splitting import extract_company_identifier, split_by_company


def test_extract_company_identifier():
    assert extract_company_identifier("AAPL_2025.pdf") == "AAPL"
    assert extract_company_identifier("sec-edgar-filings/MSFT/10-K/0001/doc.html") == "MSFT"
    assert extract_company_identifier("data/01_raw/NVDA-10K.txt") == "NVDA"
    assert extract_company_identifier("TSLA") == "TSLA"


def test_split_by_company_no_leakage():
    # Construct synthetic dataset with 10 companies
    rows = []
    companies = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "JPM", "WMT", "V"]
    for comp in companies:
        for i in range(10):
            rows.append({
                "doc_id": f"{comp}_2025_{i}.pdf",
                "text": f"Sample text for {comp}",
                "triples": "[]"
            })
    df = pd.DataFrame(rows)

    train_df, val_df, test_df = split_by_company(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    # 1. Total rows match
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # 2. Extract company sets
    train_comps = set(train_df["company_id"])
    val_comps = set(val_df["company_id"])
    test_comps = set(test_df["company_id"])

    # 3. Check mutual exclusion (ZERO leakage between splits!)
    assert len(train_comps.intersection(val_comps)) == 0
    assert len(train_comps.intersection(test_comps)) == 0
    assert len(val_comps.intersection(test_comps)) == 0

    # 4. Check coverage
    assert train_comps.union(val_comps).union(test_comps) == set(companies)


def test_split_by_company_edge_cases():
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["doc_id", "text"])
    t, v, te = split_by_company(empty_df)
    assert t.empty and v.empty and te.empty

    # Less than 3 companies fallback
    df_small = pd.DataFrame([
        {"doc_id": "AAPL_1.pdf", "text": "a"},
        {"doc_id": "AAPL_2.pdf", "text": "b"},
        {"doc_id": "MSFT_1.pdf", "text": "c"},
        {"doc_id": "MSFT_2.pdf", "text": "d"},
    ])
    t_s, v_s, te_s = split_by_company(df_small, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert len(t_s) + len(v_s) + len(te_s) == 4
