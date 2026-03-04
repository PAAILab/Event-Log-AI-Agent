#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast error detection for process mining event log.

Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_distorted_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_distorted_noLabel.detected.csv

Constraints:
- Use ONLY Case, Activity, Timestamp, Resource for detection.
- Empty Resource is NOT an error.
- Must be fast (<60s): no df.iterrows(), no O(n^2) loops over all rows.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter

IN_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_distorted_noLabel.csv"
OUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_distorted_noLabel.detected.csv"

# -----------------------------
# Helpers (vectorized-friendly)
# -----------------------------
POLLUTED_RE = re.compile(
    r"""^(?P<base>.+?)_([A-Za-z0-9]{5,12})_(\d{8}\s\d{6}\d{3}\d{3})$"""
)

def normalize_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

def strip_polluted(activity: pd.Series) -> pd.Series:
    # Extract base label if polluted pattern matches; else keep original
    a = activity.fillna("").astype(str)
    m = a.str.extract(POLLUTED_RE)
    base = m["base"]
    out = a.where(base.isna(), base)
    return out

def simple_norm_for_match(s: pd.Series) -> pd.Series:
    # Lowercase, remove non-alnum, collapse spaces
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[_\-]+", " ", regex=True)
    s = s.str.replace(r"[^a-z0-9 ]+", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def bigram_jaccard(a: pd.Series, b: pd.Series) -> np.ndarray:
    """
    Fast-ish approximate similarity for distorted detection.
    Computes Jaccard on character bigrams for aligned pairs (a[i], b[i]).
    Vectorized via python loop over unique pairs only (bounded by topK).
    """
    # Build on numpy arrays
    aa = a.to_numpy(dtype=object)
    bb = b.to_numpy(dtype=object)

    # Unique pair compression to avoid per-row heavy work
    pairs = pd.DataFrame({"a": aa, "b": bb})
    # factorize pairs
    key = pairs["a"].astype(str) + "\u0001" + pairs["b"].astype(str)
    codes, uniques = pd.factorize(key, sort=False)

    sim_u = np.zeros(len(uniques), dtype=float)

    # Compute similarity per unique pair (<= nrows but typically much less;
    # and we only call this on candidates, not all rows)
    for i, u in enumerate(uniques):
        sa, sb = u.split("\u0001", 1)
        if sa == sb:
            sim_u[i] = 1.0
            continue
        if not sa or not sb:
            sim_u[i] = 0.0
            continue
        # bigrams
        ga = {sa[j:j+2] for j in range(len(sa)-1)} if len(sa) > 1 else {sa}
        gb = {sb[j:j+2] for j in range(len(sb)-1)} if len(sb) > 1 else {sb}
        inter = len(ga & gb)
        union = len(ga | gb)
        sim_u[i] = inter / union if union else 0.0

    return sim_u[codes]

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -----------------------------
# Main
# -----------------------------
def main():
    # Read only required columns
    usecols = ["Case", "Activity", "Timestamp", "Resource"]
    df = pd.read_csv(IN_PATH, usecols=usecols, low_memory=False)

    n = len(df)
    row_id = np.arange(n, dtype=int)

    # Normalize columns
    df["Case"] = normalize_text(df["Case"])
    df["Activity_raw"] = normalize_text(df["Activity"])
    df["Resource"] = df["Resource"]  # keep as-is; empty is allowed
    # Timestamp parsing (coerce errors)
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
    df["ts"] = ts

    # Prepare activity variants
    df["Activity_base"] = strip_polluted(df["Activity_raw"])
    df["act_norm"] = simple_norm_for_match(df["Activity_base"])
    df["act_raw_norm"] = simple_norm_for_match(df["Activity_raw"])

    # Canonical list: topK most frequent base labels (after stripping polluted)
    topK = 80  # within requested 50-100
    freq = df["Activity_base"].value_counts(dropna=False)
    canon = freq.head(topK).index.astype(str).tolist()
    canon_norm = simple_norm_for_match(pd.Series(canon)).tolist()
    canon_norm_set = set(canon_norm)

    # Output containers
    err_types = np.array([""] * n, dtype=object)
    err_tags = np.array([""] * n, dtype=object)
    err_evid = np.array([""] * n, dtype=object)
    err_desc = np.array([""] * n, dtype=object)
    conf = np.zeros(n, dtype=float)

    def add_error(mask, etype, tag, evidence, desc, c):
        nonlocal err_types, err_tags, err_evid, err_desc, conf
        idx = np.where(mask)[0]
        if idx.size == 0:
            return
        # append pipe-separated unique-ish (cheap)
        for arr, val in [(err_types, etype), (err_tags, tag), (err_evid, evidence), (err_desc, desc)]:
            cur = arr[idx]
            arr[idx] = np.where(cur == "", val, cur + " | " + val)
        conf[idx] = np.maximum(conf[idx], c)

    # -----------------------------
    # 0) Basic format/range issues (aggressive)
    # -----------------------------
    # Missing/invalid Case
    m_case_bad = df["Case"].isna() | (df["Case"].str.len() == 0)
    add_error(
        m_case_bad, "format", "CASE_EMPTY",
        "Case is empty/null",
        "Case identifier is missing; cannot reliably group events into cases.",
        0.95
    )

    # Missing/empty Activity
    m_act_bad = df["Activity_raw"].isna() | (df["Activity_raw"].str.len() == 0)
    add_error(
        m_act_bad, "format", "ACTIVITY_EMPTY",
        "Activity is empty/null",
        "Activity label is missing.",
        0.95
    )

    # Invalid timestamp
    m_ts_bad = df["ts"].isna()
    add_error(
        m_ts_bad, "format", "TIMESTAMP_PARSE_FAIL",
        "Timestamp could not be parsed to datetime",
        "Timestamp is invalid/unparseable.",
        0.98
    )

    # -----------------------------
    # 1) POLLUTED
    # -----------------------------
    m_polluted = df["Activity_raw"].str.match(POLLUTED_RE, na=False)
    # canonical guess = stripped base
    add_error(
        m_polluted, "polluted", "ACT_POLLUTED_SUFFIX",
        "Activity matches pattern '<base>_<5-12alnum>_<YYYYMMDD HHMMSS...>'",
        "Activity appears polluted by a machine-generated suffix; base label extracted as canonical.",
        0.92
    )

    # -----------------------------
    # 2) SYNONYMOUS (dictionary-based, fast)
    # -----------------------------
    # Keep small but diverse; only uses Activity text.
    syn_map = {
        # review
        "review case": "Review application",
        "assess application": "Review application",
        "evaluate application": "Review application",
        "inspect application": "Review application",
        # reject
        "reject application": "Reject request",
        "deny request": "Reject request",
        "decline application": "Reject request",
        "refuse request": "Reject request",
        # diagnose
        "make diagnosis": "Diagnose patient",
        "establish diagnosis": "Diagnose patient",
        "determine diagnosis": "Diagnose patient",
        "confirm diagnosis": "Diagnose patient",
        # production
        "start manufacturing": "Start production",
        "begin production": "Start production",
        "initiate production run": "Start production",
        "launch production": "Start production",
        # approvals (common)
        "grant approval": "Approve request",
        "approve application": "Approve request",
        "accept request": "Approve request",
        # request info
        "request additional info": "Request info",
        "request information": "Request info",
    }
    syn_keys = list(syn_map.keys())
    syn_vals = [syn_map[k] for k in syn_keys]
    syn_keys_norm = simple_norm_for_match(pd.Series(syn_keys)).tolist()
    syn_lookup = dict(zip(syn_keys_norm, syn_vals))

    actn = df["act_norm"]
    m_syn = actn.isin(syn_lookup.keys())
    if m_syn.any():
        canon_guess = actn.map(syn_lookup).fillna("")
        add_error(
            m_syn, "synonymous", "ACT_SYNONYM_DICT",
            ("Synonym match; canonical=" + canon_guess).astype(str),
            "Activity text matches a known synonym of a canonical activity label.",
            0.85
        )

    # -----------------------------
    # 3) DISTORTED (approximate to topK canonicals)
    #    Strategy:
    #      - candidate rows: not polluted-suffix-only (we still allow), not empty
    #      - if normalized activity not in canonical set, try nearest canonical by bigram Jaccard
    #      - only compare against topK canonicals (<=80)
    # -----------------------------
    # Candidates: have activity, have some letters, and not already exact canonical
    m_dist_cand = (~m_act_bad) & (df["act_norm"].str.len() >= 4) & (~df["act_norm"].isin(canon_norm_set))

    # Build similarity to each canonical for candidates using vectorized loop over canonicals (K<=80)
    cand_idx = np.where(m_dist_cand.to_numpy())[0]
    if cand_idx.size > 0 and len(canon_norm) > 0:
        cand_act = df.loc[cand_idx, "act_norm"].astype(str)

        best_sim = np.zeros(cand_idx.size, dtype=float)
        best_canon = np.array([""] * cand_idx.size, dtype=object)

        # Compare to each canonical (O(K * #candidates), K small)
        for cn, cn_raw in zip(canon_norm, canon):
            sim = bigram_jaccard(cand_act, pd.Series([cn] * cand_idx.size))
            better = sim > best_sim
            if np.any(better):
                best_sim[better] = sim[better]
                best_canon[better] = cn_raw

        # Distorted if similarity high enough but not exact
        # Aggressive thresholds: 0.72+ likely typo; 0.80+ very likely
        m_dist = best_sim >= 0.78
        if np.any(m_dist):
            idx2 = cand_idx[m_dist]
            ev = pd.Series(best_sim[m_dist]).round(3).astype(str).to_numpy()
            add_error(
                np.isin(row_id, idx2),
                "distorted",
                "ACT_DISTORT_BIGRAM_TOPK",
                ("Nearest canonical=" + pd.Series(best_canon[m_dist]).astype(str) +
                 ", sim=" + pd.Series(ev).astype(str)).astype(str),
                "Activity label is very similar to a frequent canonical label (likely typo/character swap).",
                np.clip(best_sim[m_dist], 0.78, 0.95).mean() if m_dist.sum() else 0.0
            )

    # -----------------------------
    # 4) FORMBASED (same Case + same Timestamp repeated for multiple events)
    #    Detect groups within a case where identical timestamp occurs >=3 times
    # -----------------------------
    # Only consider valid timestamps
    df_valid_ts = df[~m_ts_bad].copy()
    if len(df_valid_ts) > 0:
        grp_sizes = df_valid_ts.groupby(["Case", "ts"], sort=False).size()
        bad_keys = grp_sizes[grp_sizes >= 3].index  # aggressive: 3+ at same ts in same case
        if len(bad_keys) > 0:
            # mark all but the first occurrence in each (Case, ts) group as formbased
            df_valid_ts["rank_in_ts"] = df_valid_ts.groupby(["Case", "ts"], sort=False).cumcount()
            m_form = df_valid_ts.set_index(["Case", "ts"]).index.isin(bad_keys) & (df_valid_ts["rank_in_ts"] >= 1)
            form_row_ids = df_valid_ts.loc[m_form].index.to_numpy()

            add_error(
                np.isin(row_id, form_row_ids),
                "formbased",
                "TS_SAME_WITHIN_CASE_GE3",
                "Same timestamp repeats >=3 times within same case; non-first occurrences flagged",
                "Multiple events share the exact same timestamp within a case (likely form overwrite/submission artifact).",
                0.80
            )

    # -----------------------------
    # 5) COLLATERAL (duplicates / near-duplicates)
    #    a) exact duplicates: same Case+Activity_base+ts+Resource (Resource can be NaN)
    #    b) near duplicates: same Case+Activity_base+Resource within <=2 seconds
    # -----------------------------
    # a) exact duplicates (including NaN resource treated as equal by fillna marker)
    res_key = df["Resource"].fillna("__NA__").astype(str)
    key_exact = (
        df["Case"].astype(str) + "\u0001" +
        df["Activity_base"].astype(str) + "\u0001" +
        df["ts"].astype(str) + "\u0001" +
        res_key
    )
    dup_exact = key_exact.duplicated(keep="first") & (~m_ts_bad) & (~m_act_bad) & (~m_case_bad)
    add_error(
        dup_exact,
        "collateral",
        "DUP_EXACT_CASE_ACT_TS_RES",
        "Exact duplicate of (Case, Activity, Timestamp, Resource) found",
        "Event appears to be an exact duplicate log entry (collateral logging artifact).",
        0.93
    )

    # b) near duplicates within case+activity+resource
    # Sort by (Case, Activity_base, Resource, ts) and diff
    df_nd = df[~m_ts_bad & ~m_act_bad & ~m_case_bad].copy()
    if len(df_nd) > 0:
        df_nd["res_key"] = df_nd["Resource"].fillna("__NA__").astype(str)
        df_nd = df_nd.sort_values(["Case", "Activity_base", "res_key", "ts"], kind="mergesort")
        dt = df_nd.groupby(["Case", "Activity_base", "res_key"], sort=False)["ts"].diff().dt.total_seconds()
        m_near = dt.notna() & (dt >= 0) & (dt <= 2.0)
        near_ids = df_nd.loc[m_near].index.to_numpy()
        add_error(
            np.isin(row_id, near_ids),
            "collateral",
            "NEAR_DUP_WITHIN_2S_SAME_CTX",
            "Same Case+Activity+Resource repeated within <=2 seconds",
            "Event is a near-duplicate occurring implausibly soon after the same activity by the same resource in the same case.",
            0.82
        )

    # -----------------------------
    # 6) HOMONYMOUS (aggressive heuristic)
    #    Without ground truth semantics, detect labels that split into 2+ distinct contexts:
    #      - same Activity_base appears with very different typical predecessor/successor activities across cases
    #    We only evaluate top frequent activities to keep fast.
    # -----------------------------
    # Build within-case order by timestamp (stable)
    df_ord = df[~m_ts_bad & ~m_case_bad & ~m_act_bad].copy()
    if len(df_ord) > 0:
        df_ord = df_ord.sort_values(["Case", "ts", "Activity_raw"], kind="mergesort")
        df_ord["prev_act"] = df_ord.groupby("Case", sort=False)["Activity_base"].shift(1)
        df_ord["next_act"] = df_ord.groupby("Case", sort=False)["Activity_base"].shift(-1)

        # Focus on topM activities
        topM = 40
        top_acts = df_ord["Activity_base"].value_counts().head(topM).index.astype(str)

        sub = df_ord[df_ord["Activity_base"].isin(top_acts)].copy()
        # Context signature: (prev, next) normalized
        sub["ctx"] = (
            sub["prev_act"].fillna("__START__").astype(str) + " -> " +
            sub["next_act"].fillna("__END__").astype(str)
        )

        # For each activity, compute concentration of top context
        ctx_counts = sub.groupby(["Activity_base", "ctx"], sort=False).size().rename("cnt").reset_index()
        total = ctx_counts.groupby("Activity_base", sort=False)["cnt"].sum().rename("total").reset_index()
        merged = ctx_counts.merge(total, on="Activity_base", how="left")
        merged["share"] = merged["cnt"] / merged["total"]

        # Homonymous candidate: activity has at least 2 contexts each >= 0.25 share and total >= 30
        # (means it behaves like two different actions)
        totals = total.set_index("Activity_base")["total"]
        strong = merged[(merged["share"] >= 0.25)]
        strong_n = strong.groupby("Activity_base", sort=False).size()
        homo_acts = strong_n[(strong_n >= 2) & (totals.loc[strong_n.index] >= 30)].index

        if len(homo_acts) > 0:
            m_homo = df["Activity_base"].isin(homo_acts)
            add_error(
                m_homo,
                "homonymous",
                "CTX_SPLIT_PREV_NEXT",
                "Same label shows 2+ dominant (prev,next) contexts across cases",
                "Activity label appears to represent multiple different behaviors (context splits strongly), suggesting homonymous usage.",
                0.60
            )

    # -----------------------------
    # Finalize
    # -----------------------------
    error_flag = err_types != ""
    # normalize error_types to pipe-separated lowercase tokens
    # (keep only known types + format)
    def norm_types(s):
        if not s:
            return ""
        parts = [p.strip().lower() for p in s.split("|")]
        # map to canonical names
        mapped = []
        for p in parts:
            if p in ("polluted", "distorted", "synonymous", "formbased", "collateral", "homonymous", "format"):
                mapped.append(p)
            else:
                mapped.append(p)
        # de-dup preserve order
        out = []
        seen = set()
        for p in mapped:
            if p not in seen:
                out.append(p); seen.add(p)
        return "|".join(out)

    error_types_norm = pd.Series(err_types).fillna("").map(norm_types).to_numpy()

    # Confidence: if multiple errors, boost slightly but cap
    num_types = pd.Series(error_types_norm).replace("", np.nan).dropna().map(lambda x: len(x.split("|")))
    boost = np.zeros(n, dtype=float)
    boost[num_types.index.to_numpy()] = np.clip((num_types.to_numpy() - 1) * 0.05, 0, 0.15)
    error_conf = np.clip(conf + boost, 0.0, 1.0)

    out = pd.DataFrame({
        "row_id": row_id,
        "error_flag": error_flag,
        "error_types": error_types_norm,
        "error_confidence": error_conf.round(4),
        "error_tags": err_tags,
        "error_evidence": err_evid,
        "error_description": err_desc,
    })

    ensure_dir(OUT_PATH)
    out.to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()