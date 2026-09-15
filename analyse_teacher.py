"""
analyse_teacher.py
──────────────────
Four-part deep analysis of the LLM Teacher distillation dataset.

Produces a single Markdown report covering:
  1. Teacher Self-Consistency  (Agent 1 vs Agent 2 correction rate)
  2. Schema Compliance & Class Imbalance  (+ auto focal-loss alpha)
  3. Teacher-as-Ceiling Evaluation  (upper-bound F1 via teacher agreement)
  4. Entity Verbosity / Token-Slot Analysis  (guides num_token_slots tuning)

Usage (run from project root on the cluster):
    python analyse_teacher.py

    # or with explicit paths / options:
    python analyse_teacher.py \
        --triplets   data/02_intermediate/teacher_extracted_triplets.csv \
        --agent1     data/02_intermediate/teacher_agent1_extractions.parquet \
        --agent2     data/02_intermediate/teacher_agent2_critiques.parquet \
        --tokenizer  nlpaueb/sec-bert-base \
        --token_slots 8 \
        --output     docs/teacher_analysis.md \
        --seed       42
"""

import argparse
import ast
import json
import math
import os
import re
import random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ─── Schema ──────────────────────────────────────────────────────────────────

KNOWN_RELATIONS = [
    "has_metric", "produces", "operates_in", "reports_risk", "led_by",
    "has_stake_in", "impacts", "involved_in", "impacted_by", "discloses",
    "complies_with", "supplies", "partners_with",
]
KNOWN_ENTITY_TYPES = [
    "org", "person", "comp", "product", "segment",
    "fin_metric", "risk_factor", "event", "regulatory_requirement", "esg_topic",
]
TYPE_SYNONYMS = {
    "company": "org", "corporation": "org", "firm": "org", "enterprise": "org",
    "metric": "fin_metric", "financial_metric": "fin_metric", "kpi": "fin_metric",
    "risk": "risk_factor", "program": "product", "service": "product",
    "division": "segment", "unit": "segment", "business_segment": "segment",
}

REL_SET = set(KNOWN_RELATIONS)
ENT_SET = set(KNOWN_ENTITY_TYPES)


# ─── Parsing helpers ──────────────────────────────────────────────────────────

def robust_parse(raw_val):
    """Mirrors the robust parsing logic from data_prep/nodes.py."""
    if not isinstance(raw_val, str):
        return []
    s = raw_val.strip()
    s = re.sub(r"\}\s*\{", "}, {", s)
    for fn in [ast.literal_eval, json.loads]:
        try:
            r = fn(s)
            if isinstance(r, list):
                return r
        except Exception:
            pass
    try:
        fixed = re.sub(r"(?<=[\{\s,])'([a-zA-Z0-9_]+)':", r'"\1":', s)
        r = json.loads(fixed)
        if isinstance(r, list):
            return r
    except Exception:
        pass
    return []


def normalise_type(t):
    t = str(t).strip().lower()
    return TYPE_SYNONYMS.get(t, t)


def triple_to_tuple(t):
    """Convert a dict triple to a canonical comparable tuple."""
    rel = str(t.get("relation", "")).strip().lower()
    head = str(t.get("head", "")).strip().lower()
    tail = str(t.get("tail", "")).strip().lower()
    ht = normalise_type(t.get("head_type", ""))
    tt = normalise_type(t.get("tail_type", ""))
    return (head, ht, rel, tail, tt)


def is_valid_triple(t):
    rel = str(t.get("relation", "")).strip().lower()
    ht = normalise_type(t.get("head_type", ""))
    tt = normalise_type(t.get("tail_type", ""))
    return rel in REL_SET and ht in ENT_SET and tt in ENT_SET


# ─── Markdown helpers ─────────────────────────────────────────────────────────

def md_table(headers, rows):
    sep = ["|---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|".join(sep) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def pct_bar(value, total, width=25):
    ratio = min(value / max(total, 1), 1.0)
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled) + f" {ratio*100:5.1f}%"


# ═════════════════════════════════════════════════════════════════════════════
# OPTION 1 — Teacher Self-Consistency (Agent 1 vs Agent 2)
# ═════════════════════════════════════════════════════════════════════════════

def analyse_self_consistency(agent1_path, agent2_path, triplets_df):
    """
    Compares Agent 1 raw extractions against Agent 2 critic decisions.
    Agent 2 output is 'PASS', 'FAIL', or a corrected JSON list.
    """
    results = {
        "agent1_available": False,
        "agent2_available": False,
        "total_reviewed": 0,
        "pass_count": 0,
        "fail_count": 0,
        "correction_count": 0,
        "parse_failures_a1": 0,
        "parse_failures_a2": 0,
        "correction_rate": 0.0,
        "pass_rate": 0.0,
        "fail_rate": 0.0,
        "agreement_triple_f1": None,
        "notes": [],
    }

    if not (agent1_path and os.path.exists(agent1_path)):
        results["notes"].append("Agent 1 parquet not found — skipping.")
        return results
    if not (agent2_path and os.path.exists(agent2_path)):
        results["notes"].append("Agent 2 parquet not found — skipping.")
        return results

    try:
        df1 = pd.read_parquet(agent1_path)
        df2 = pd.read_parquet(agent2_path)
        results["agent1_available"] = True
        results["agent2_available"] = True
    except Exception as e:
        results["notes"].append(f"Error loading parquets: {e}")
        return results

    # Align indices — both should have the same length
    n = min(len(df1), len(df2))
    results["total_reviewed"] = n

    a1_col = df1.columns[0]
    a2_col = df2.columns[0]

    tp_total, fp_total, fn_total = 0, 0, 0

    for i in range(n):
        a1_raw = str(df1.iloc[i][a1_col])
        a2_raw = str(df2.iloc[i][a2_col]).strip()

        a1_triples = robust_parse(a1_raw)
        if not a1_triples:
            results["parse_failures_a1"] += 1

        # Classify Agent 2 decision
        a2_upper = a2_raw.upper()
        if a2_upper.startswith("PASS"):
            results["pass_count"] += 1
            # Agent 2 agreed — use agent 1 triples as final
            final_triples = a1_triples
        elif a2_upper.startswith("FAIL"):
            results["fail_count"] += 1
            final_triples = []
        else:
            # Agent 2 returned a corrected list
            corrected = robust_parse(a2_raw)
            if corrected:
                results["correction_count"] += 1
                final_triples = corrected
            else:
                results["parse_failures_a2"] += 1
                final_triples = a1_triples  # fall back

        # Compute triple-level agreement between a1 and final
        if a1_triples and final_triples:
            a1_set = {triple_to_tuple(t) for t in a1_triples if isinstance(t, dict)}
            final_set = {triple_to_tuple(t) for t in final_triples if isinstance(t, dict)}
            tp_total += len(a1_set & final_set)
            fp_total += len(a1_set - final_set)
            fn_total += len(final_set - a1_set)

    results["correction_rate"] = results["correction_count"] / max(n, 1)
    results["pass_rate"] = results["pass_count"] / max(n, 1)
    results["fail_rate"] = results["fail_count"] / max(n, 1)

    if (tp_total + fp_total + fn_total) > 0:
        prec = tp_total / max(tp_total + fp_total, 1)
        rec = tp_total / max(tp_total + fn_total, 1)
        results["agreement_triple_f1"] = 2 * prec * rec / max(prec + rec, 1e-9)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# OPTION 2 — Schema Compliance & Class Imbalance
# ═════════════════════════════════════════════════════════════════════════════

def analyse_schema_compliance(df_teacher):
    rel_counter = Counter()
    head_type_counter = Counter()
    tail_type_counter = Counter()
    unknown_rels = Counter()
    unknown_head_types = Counter()
    unknown_tail_types = Counter()

    total_triples = 0
    valid_triples = 0
    invalid_rel = 0
    invalid_etype = 0

    for _, row in tqdm(df_teacher.iterrows(), total=len(df_teacher), desc="[Option 2] Schema compliance"):
        for t in robust_parse(row.get("triples")):
            if not isinstance(t, dict):
                continue
            total_triples += 1
            rel = str(t.get("relation", "")).strip().lower()
            ht = normalise_type(t.get("head_type", ""))
            tt = normalise_type(t.get("tail_type", ""))

            rel_counter[rel] += 1
            head_type_counter[ht] += 1
            tail_type_counter[tt] += 1

            if rel not in REL_SET:
                unknown_rels[rel] += 1
                invalid_rel += 1
            if ht not in ENT_SET:
                unknown_head_types[ht] += 1
                invalid_etype += 1
            if tt not in ENT_SET:
                unknown_tail_types[tt] += 1
                invalid_etype += 1

            if rel in REL_SET and ht in ENT_SET and tt in ENT_SET:
                valid_triples += 1

    # Compute focal-loss alpha weights (inverse frequency, normalised)
    valid_rel_counts = {r: rel_counter.get(r, 0) for r in KNOWN_RELATIONS}
    total_valid_rels = sum(valid_rel_counts.values())
    # Alpha_r = 1 / freq_r, then normalise so mean alpha = 1
    inv_freq = {r: 1.0 / max(c, 1) for r, c in valid_rel_counts.items()}
    mean_inv = np.mean(list(inv_freq.values()))
    alphas = {r: v / mean_inv for r, v in inv_freq.items()}

    return {
        "total_triples": total_triples,
        "valid_triples": valid_triples,
        "invalid_rel": invalid_rel,
        "invalid_etype": invalid_etype,
        "rel_counter": rel_counter,
        "head_type_counter": head_type_counter,
        "tail_type_counter": tail_type_counter,
        "unknown_rels": unknown_rels,
        "unknown_head_types": unknown_head_types,
        "unknown_tail_types": unknown_tail_types,
        "valid_rel_counts": valid_rel_counts,
        "total_valid_rels": total_valid_rels,
        "focal_alphas": alphas,
    }


# ═════════════════════════════════════════════════════════════════════════════
# OPTION 3 — Teacher-as-Ceiling Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ceiling(agent1_path, agent2_path, seed=42, sample_size=2000):
    """
    Since we only have one set of teacher labels (not two independent runs),
    we estimate the ceiling by computing agreement between the raw Agent 1
    extractions and the final accepted triples (after Agent 2's critique).
    This gives a 'label noise' estimate representing the theoretical ceiling.
    """
    if not (agent1_path and os.path.exists(agent1_path) and agent2_path and os.path.exists(agent2_path)):
        return {"error": "Requires both Agent 1 and Agent 2 parquet files to compute ceiling F1."}

    df1 = pd.read_parquet(agent1_path)
    df2 = pd.read_parquet(agent2_path)
    n = min(len(df1), len(df2))

    tp, fp, fn = 0, 0, 0
    span_tp, span_fp, span_fn = 0, 0, 0
    jaccard_scores = []
    analysed_chunks = 0

    random.seed(seed)
    
    # Sample indices to speed up
    indices = random.sample(range(n), min(n, sample_size))
    
    a1_col = df1.columns[0]
    a2_col = df2.columns[0]

    for i in tqdm(indices, desc="[Option 3] Ceiling evaluation"):
        a1_raw = str(df1.iloc[i][a1_col])
        a2_raw = str(df2.iloc[i][a2_col]).strip()

        a1_triples = robust_parse(a1_raw)
        
        a2_upper = a2_raw.upper()
        if a2_upper.startswith("PASS"):
            final_triples = a1_triples
        elif a2_upper.startswith("FAIL"):
            final_triples = []
        else:
            corrected = robust_parse(a2_raw)
            final_triples = corrected if corrected else a1_triples

        if not a1_triples and not final_triples:
            continue

        gold_set = {triple_to_tuple(t) for t in a1_triples if isinstance(t, dict) and is_valid_triple(t)}
        pred_set = {triple_to_tuple(t) for t in final_triples if isinstance(t, dict) and is_valid_triple(t)}

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        # Span-level (head_text, tail_text only)
        gold_span = {(g[0], g[3]) for g in gold_set}
        pred_span = {(p[0], p[3]) for p in pred_set}
        span_tp += len(gold_span & pred_span)
        span_fp += len(pred_span - gold_span)
        span_fn += len(gold_span - pred_span)

        # Jaccard on head text tokens
        for g in gold_set:
            best_j = max(
                (len(set(g[0].split()) & set(p[0].split())) /
                 max(len(set(g[0].split()) | set(p[0].split())), 1)
                 for p in pred_set),
                default=0.0,
            )
            jaccard_scores.append(best_j)

        analysed_chunks += 1

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    sp = span_tp / max(span_tp + span_fp, 1)
    sr = span_tp / max(span_tp + span_fn, 1)
    span_f1 = 2 * sp * sr / max(sp + sr, 1e-9)

    return {
        "analysed_chunks": analysed_chunks,
        "sample_size": len(indices),
        "strict_f1": f1,
        "strict_precision": prec,
        "strict_recall": rec,
        "span_f1": span_f1,
        "span_precision": sp,
        "span_recall": sr,
        "jaccard_mean": float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
        "jaccard_median": float(np.median(jaccard_scores)) if jaccard_scores else 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# OPTION 4 — Entity Verbosity / Token-Slot Analysis
# ═════════════════════════════════════════════════════════════════════════════

def analyse_token_slots(df_teacher, tokenizer_name, num_token_slots):
    if not HAS_TRANSFORMERS:
        return {"error": "transformers not installed"}

    print(f"  Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    head_lengths, tail_lengths = [], []
    truncated_heads, truncated_tails = 0, 0
    total_heads, total_tails = 0, 0

    for _, row in tqdm(df_teacher.iterrows(), total=len(df_teacher), desc="[Option 4] Tokenising entities"):
        for t in robust_parse(row.get("triples")):
            if not isinstance(t, dict):
                continue
            head = str(t.get("head", "")).strip()
            tail = str(t.get("tail", "")).strip()

            if head:
                toks = tokenizer.encode(head, add_special_tokens=False)
                l = len(toks)
                head_lengths.append(l)
                total_heads += 1
                if l > num_token_slots:
                    truncated_heads += 1

            if tail:
                toks = tokenizer.encode(tail, add_special_tokens=False)
                l = len(toks)
                tail_lengths.append(l)
                total_tails += 1
                if l > num_token_slots:
                    truncated_tails += 1

    all_lengths = np.array(head_lengths + tail_lengths)
    head_arr = np.array(head_lengths)
    tail_arr = np.array(tail_lengths)

    def stats(arr):
        if len(arr) == 0:
            return {}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "max": int(arr.max()),
            "percentiles": {p: float(np.percentile(arr, p)) for p in [50, 75, 90, 95, 98, 99]},
        }

    # Find recommended K: cover at least 95th percentile
    recommended_k = int(np.percentile(all_lengths, 95)) if len(all_lengths) else num_token_slots

    return {
        "head_stats": stats(head_arr),
        "tail_stats": stats(tail_arr),
        "combined_stats": stats(all_lengths),
        "total_heads": total_heads,
        "total_tails": total_tails,
        "truncated_heads": truncated_heads,
        "truncated_tails": truncated_tails,
        "truncation_rate_heads": truncated_heads / max(total_heads, 1),
        "truncation_rate_tails": truncated_tails / max(total_tails, 1),
        "current_token_slots": num_token_slots,
        "recommended_k": recommended_k,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═════════════════════════════════════════════════════════════════════════════

def build_report(args, consistency, schema, ceiling, slots):
    lines = []

    def h(level, title):
        lines.append(f"\n{'#' * level} {title}\n")

    def p(*args_):
        lines.append(" ".join(str(a) for a in args_))

    def hr():
        lines.append("\n---\n")

    lines.append("# Teacher Data Analysis Report")
    lines.append(f"\n> Generated: **{datetime.now().strftime('%Y-%m-%d %H:%M')}**  ")
    lines.append(f"> Dataset: `{args.triplets}`  ")
    lines.append(f"> Tokenizer: `{args.tokenizer}`  ")
    lines.append(f"> Token slots (K): `{args.token_slots}`\n")

    lines.append("> **This report covers four analyses:**")
    lines.append("> 1. Teacher Self-Consistency (Agent 1 vs Agent 2)")
    lines.append("> 2. Schema Compliance & Focal-Loss Alpha Weights")
    lines.append("> 3. Teacher-as-Ceiling Evaluation (Upper-Bound F1)")
    lines.append("> 4. Entity Verbosity & Token-Slot Sizing")

    hr()

    # ─────────────────────────────────────────────────────────────────────────
    h(2, "1. Teacher Self-Consistency (Agent 1 vs Agent 2)")
    # ─────────────────────────────────────────────────────────────────────────

    if consistency.get("notes"):
        for note in consistency["notes"]:
            p(f"> ⚠️  {note}")
    
    if consistency.get("agent1_available") and consistency.get("agent2_available"):
        n = consistency["total_reviewed"]
        p(f"**Total chunks reviewed by both agents: `{n:,}`**\n")
        lines.append(md_table(
            ["Decision", "Count", "Share", ""],
            [
                ["✅ PASS (Agent 2 agreed with Agent 1)", f"{consistency['pass_count']:,}", f"{consistency['pass_rate']*100:.1f}%", pct_bar(consistency["pass_count"], n)],
                ["✏️  CORRECTED (Agent 2 rewrote triples)", f"{consistency['correction_count']:,}", f"{consistency['correction_rate']*100:.1f}%", pct_bar(consistency["correction_count"], n)],
                ["❌ FAIL (Agent 2 rejected entirely)", f"{consistency['fail_count']:,}", f"{consistency['fail_rate']*100:.1f}%", pct_bar(consistency["fail_count"], n)],
                ["🔴 Parse failures (A1)", f"{consistency['parse_failures_a1']:,}", f"{consistency['parse_failures_a1']/max(n,1)*100:.1f}%", ""],
                ["🔴 Parse failures (A2)", f"{consistency['parse_failures_a2']:,}", f"{consistency['parse_failures_a2']/max(n,1)*100:.1f}%", ""],
            ]
        ))

        if consistency.get("agreement_triple_f1") is not None:
            f1 = consistency["agreement_triple_f1"]
            p(f"\n**Agent 1 → Agent 2 Triple-Level Agreement F1: `{f1:.4f}`**")
            if f1 > 0.8:
                p("> ✅ High agreement — the teacher labeler is internally consistent.")
            elif f1 > 0.5:
                p("> ⚠️  Moderate agreement — some label noise expected.")
            else:
                p("> ❌ Low agreement — high label noise. Consider re-running Agent 2 with stricter prompts.")
    else:
        p("> Agent parquet files not available. Run the teacher pipeline first.")
        p("> You can still interpret this from the correction rate visible in `teacher_distill.log`.")

    h(3, "1.1 Interpretation")
    p("""
The **PASS rate** tells you how often Agent 1's raw extraction was already correct
enough that Agent 2 didn't need to change anything. A high PASS rate (>70%) means
the teacher is reliable and label noise is low.

The **correction rate** is your effective label noise proxy. If Agent 2 rewrites
>30% of extractions, the student is learning from inconsistent targets — you should
apply label smoothing during training (already done in `bipartite_loss.py` with
`label_smoothing=0.1`).
""")

    hr()
    # ─────────────────────────────────────────────────────────────────────────
    h(2, "2. Schema Compliance & Class Imbalance")
    # ─────────────────────────────────────────────────────────────────────────

    total = schema["total_triples"]
    valid = schema["valid_triples"]
    p(f"**Total teacher triplets: `{total:,}`**  ")
    p(f"**Valid (in-schema) triplets: `{valid:,}` ({valid/max(total,1)*100:.1f}%)**  ")
    p(f"**Will be dropped (out-of-schema): `{total - valid:,}` ({(total-valid)/max(total,1)*100:.1f}%)**\n")

    h(3, "2.1 Relation Type Distribution")
    lines.append(md_table(
        ["Relation", "Count", "% of total", "Class bar"],
        [
            [f"`{r}`", f"{schema['rel_counter'].get(r, 0):,}",
             f"{schema['rel_counter'].get(r, 0)/max(total,1)*100:.1f}%",
             pct_bar(schema["rel_counter"].get(r, 0), total)]
            for r in KNOWN_RELATIONS
        ]
    ))

    h(3, "2.2 Entity Type Distribution")
    lines.append(md_table(
        ["Entity Type", "As Head", "As Tail"],
        [
            [f"`{e}`",
             f"{schema['head_type_counter'].get(e, 0):,}",
             f"{schema['tail_type_counter'].get(e, 0):,}"]
            for e in KNOWN_ENTITY_TYPES
        ]
    ))

    if schema["unknown_rels"]:
        h(3, "2.3 Out-of-Schema Relation Labels (will be dropped)")
        rows = [[f"`{r}`", f"{c:,}", f"{c/max(total,1)*100:.2f}%"]
                for r, c in schema["unknown_rels"].most_common(20)]
        lines.append(md_table(["Unknown Relation", "Count", "% total"], rows))

    h(3, "2.4 Auto-Computed Focal Loss Alpha Weights")
    p("""
These inverse-frequency weights can be passed directly to `multiclass_focal_loss`
in `bipartite_loss.py` to counteract class imbalance. Higher alpha = upweight rare class.
Copy these values into `parameters.yml` under `benchmark.focal_alpha_weights`.
""")
    lines.append(md_table(
        ["Relation", "Count", "Alpha Weight"],
        sorted(
            [[f"`{r}`", f"{schema['valid_rel_counts'].get(r, 0):,}", f"{schema['focal_alphas'].get(r, 1.0):.4f}"]
             for r in KNOWN_RELATIONS],
            key=lambda x: float(x[2]),
            reverse=True,
        )
    ))
    p("\n**Paste this into `parameters.yml`:**")
    lines.append("```yaml")
    lines.append("benchmark:")
    lines.append("  focal_alpha_weights:")
    for r in sorted(schema["focal_alphas"], key=lambda x: schema["focal_alphas"][x], reverse=True):
        lines.append(f"    {r}: {schema['focal_alphas'][r]:.4f}")
    lines.append("```")

    hr()
    # ─────────────────────────────────────────────────────────────────────────
    h(2, "3. Teacher-as-Ceiling Evaluation")
    # ─────────────────────────────────────────────────────────────────────────

    if "error" in ceiling:
        p(f"> ⚠️  {ceiling['error']}")
    else:
        p(f"*Approximated from a random sample of `{ceiling['sample_size']:,}` chunks.*")
        p(f"*Valid chunks analysed: `{ceiling['analysed_chunks']:,}`*\n")
        p("""
    **How to read this:** The score here represents what F1 a *perfect student*
    could expect if it exactly reproduced the teacher's extractions.
    It is measured by comparing Agent 1's initial raw extractions to the final
    accepted label.
    """)
        lines.append(md_table(
            ["Metric", "Value", "Interpretation"],
            [
                ["Strict F1 (exact 5-tuple match)", f"`{ceiling['strict_f1']:.4f}`",
                 "Upper bound for student Strict F1"],
                ["Strict Precision", f"`{ceiling['strict_precision']:.4f}`", ""],
                ["Strict Recall", f"`{ceiling['strict_recall']:.4f}`", ""],
                ["Span F1 (head+tail text only)", f"`{ceiling['span_f1']:.4f}`",
                 "Upper bound for student Span F1"],
                ["Span Precision", f"`{ceiling['span_precision']:.4f}`", ""],
                ["Span Recall", f"`{ceiling['span_recall']:.4f}`", ""],
                ["Jaccard Similarity (mean)", f"`{ceiling['jaccard_mean']:.4f}`",
                 "Token-overlap agreement between teacher labels"],
            ]
        ))
    
        h(3, "3.1 Interpretation")
        sf1 = ceiling["strict_f1"]
        if sf1 > 0.7:
            label_quality = "**high quality**. The teacher is self-consistent and the student has a clear target to aim for."
        elif sf1 > 0.4:
            label_quality = "**moderate quality**. There is meaningful label noise — apply aggressive label smoothing during training."
        else:
            label_quality = "**low quality**. The teacher labels are too noisy — consider re-running with a lower temperature or a stricter critic."
        p(f"Teacher label quality is {label_quality}")
        p(f"\n> 🎯 Your trained student model should aim to reach **~{sf1*0.6:.1%}–{sf1*0.8:.1%}** of this ceiling on held-out test data.")

    hr()
    # ─────────────────────────────────────────────────────────────────────────
    h(2, "4. Entity Verbosity & Token-Slot Sizing")
    # ─────────────────────────────────────────────────────────────────────────

    if "error" in slots:
        p(f"> ⚠️  Token analysis skipped: {slots['error']}")
    else:
        K = slots["current_token_slots"]
        rec = slots["recommended_k"]
        p(f"**Current `num_token_slots` (K): `{K}`**")
        p(f"**Recommended K (covers 95th percentile): `{rec}`**\n")

        h(3, "4.1 Token Length Statistics")
        lines.append(md_table(
            ["Statistic", "Head (Subject)", "Tail (Object)", "Combined"],
            [
                ["Mean",
                 f"{slots['head_stats'].get('mean', 0):.2f}",
                 f"{slots['tail_stats'].get('mean', 0):.2f}",
                 f"{slots['combined_stats'].get('mean', 0):.2f}"],
                ["Median",
                 f"{slots['head_stats'].get('median', 0):.2f}",
                 f"{slots['tail_stats'].get('median', 0):.2f}",
                 f"{slots['combined_stats'].get('median', 0):.2f}"],
                ["Max",
                 f"{slots['head_stats'].get('max', 0)}",
                 f"{slots['tail_stats'].get('max', 0)}",
                 f"{slots['combined_stats'].get('max', 0)}"],
            ]
        ))

        h(3, "4.2 Percentile Breakdown (Combined)")
        pcts = slots["combined_stats"].get("percentiles", {})
        lines.append(md_table(
            ["Percentile", "Token Length", "Covered by K=" + str(K)],
            [
                [f"{p_}th", f"{v:.0f}", "✅" if v <= K else "❌"]
                for p_, v in pcts.items()
            ]
        ))

        h(3, "4.3 Truncation Analysis")
        trunc_h = slots["truncation_rate_heads"]
        trunc_t = slots["truncation_rate_tails"]
        trunc_both = (slots["truncated_heads"] + slots["truncated_tails"]) / max(slots["total_heads"] + slots["total_tails"], 1)
        lines.append(md_table(
            ["Entity Role", "Total", "Truncated at K=" + str(K), "Truncation Rate", ""],
            [
                ["Head (Subject)", f"{slots['total_heads']:,}", f"{slots['truncated_heads']:,}", f"{trunc_h*100:.1f}%", pct_bar(slots["truncated_heads"], slots["total_heads"])],
                ["Tail (Object)", f"{slots['total_tails']:,}", f"{slots['truncated_tails']:,}", f"{trunc_t*100:.1f}%", pct_bar(slots["truncated_tails"], slots["total_tails"])],
                ["**Combined**", f"**{slots['total_heads']+slots['total_tails']:,}**", "", f"**{trunc_both*100:.1f}%**", ""],
            ]
        ))

        h(3, "4.4 Recommendation")
        if rec <= K:
            p(f"> ✅ **K={K} is sufficient.** The 95th percentile entity length is `{rec}` tokens, which fits within your current `num_token_slots`.")
        elif rec <= K + 2:
            p(f"> ⚠️  **Consider increasing K to `{rec}`.** The current K={K} truncates `{trunc_both*100:.1f}%` of entities at the 95th-percentile boundary.")
        else:
            p(f"> ❌ **Increase K to at least `{rec}`.** The current K={K} truncates `{trunc_both*100:.1f}%` of entities — this will severely degrade localisation quality.")
        
        p(f"\nTo update, change in `conf/base/parameters.yml`:")
        lines.append("```yaml")
        lines.append("training:")
        lines.append(f"  num_token_slots: {rec}  # covers 95th percentile entity length")
        lines.append("benchmark:")
        lines.append(f"  num_token_slots: {rec}")
        lines.append("```")

    hr()
    lines.append("*Auto-generated by `analyse_teacher.py` — BERT-KG Extraction Pipeline*")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full teacher dataset analysis — 4 reports in one.")
    parser.add_argument("--triplets", default="data/02_intermediate/teacher_extracted_triplets.csv")
    parser.add_argument("--agent1", default="data/02_intermediate/teacher_agent1_extractions.parquet",
                        help="Agent 1 raw extraction parquet (optional)")
    parser.add_argument("--agent2", default="data/02_intermediate/teacher_agent2_critiques.parquet",
                        help="Agent 2 critic parquet (optional)")
    parser.add_argument("--tokenizer", default="nlpaueb/sec-bert-base")
    parser.add_argument("--token_slots", type=int, default=8,
                        help="Current num_token_slots in parameters.yml")
    parser.add_argument("--ceiling_samples", type=int, default=2000,
                        help="Number of chunks to sample for ceiling evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="docs/teacher_analysis.md")
    parser.add_argument("--skip_tokenizer", action="store_true",
                        help="Skip tokenizer analysis (Option 4) — much faster")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  BERT-KG Teacher Data Analysis")
    print(f"{'='*60}\n")

    # Load main dataset
    print(f"Loading: {args.triplets}")
    df = pd.read_csv(args.triplets)
    print(f"  → {len(df):,} rows\n")

    # Option 1
    print("[1/4] Teacher Self-Consistency...")
    consistency = analyse_self_consistency(args.agent1, args.agent2, df)

    # Option 2
    print("\n[2/4] Schema Compliance & Class Imbalance...")
    schema = analyse_schema_compliance(df)

    # Option 3
    print("\n[3/4] Teacher-as-Ceiling Evaluation...")
    ceiling = evaluate_ceiling(args.agent1, args.agent2, seed=args.seed, sample_size=args.ceiling_samples)

    # Option 4
    if args.skip_tokenizer:
        print("\n[4/4] Skipping tokenizer analysis (--skip_tokenizer).")
        slots = {"error": "Skipped via --skip_tokenizer flag"}
    else:
        print("\n[4/4] Entity Token-Slot Analysis...")
        slots = analyse_token_slots(df, args.tokenizer, args.token_slots)

    # Build report
    print("\nBuilding Markdown report...")
    report = build_report(args, consistency, schema, ceiling, slots)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Report written to: {args.output}")
    print(f"\nQuick Summary:")
    print(f"  Valid triplets:     {schema['valid_triples']:,} / {schema['total_triples']:,} ({schema['valid_triples']/max(schema['total_triples'],1)*100:.1f}%)")
    if "strict_f1" in ceiling:
        print(f"  Ceiling Strict F1:  {ceiling['strict_f1']:.4f}")
        print(f"  Ceiling Span F1:    {ceiling['span_f1']:.4f}")
    if "truncation_rate_heads" in slots:
        print(f"  Entity truncation:  {(slots['truncated_heads']+slots['truncated_tails'])/(max(slots['total_heads']+slots['total_tails'],1))*100:.1f}% at K={args.token_slots}")


if __name__ == "__main__":
    main()
