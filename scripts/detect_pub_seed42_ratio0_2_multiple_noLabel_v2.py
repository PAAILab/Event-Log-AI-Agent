#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast error detection for process mining event log.

Input:
  /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv

Output:
  /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv

Constraints:
- Use ONLY: Case, Activity, Timestamp, Resource
- Resource empty is NOT an error
- Must be fast (<60s): no iterrows, no O(n^2) loops, limit canonical list
"""

import os
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

IN_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv"
OUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv"

# -----------------------------
# Tunables (performance/quality)
# -----------------------------
TOP_CANONICAL = 80          # required: limit canonical list to top 50-100
COLLATERAL_WINDOW_S = 3.0   # near-duplicate window
FORMBASED_MIN_GROUP = 3     # repeated same timestamp in same case
DISTORT_MIN_RATIO = 0.86    # similarity threshold for distorted vs canonical
DISTORT_MAX_RATIO = 0.97    # avoid flagging exact canonical as distorted
SYNON_MIN_RATIO = 0.78      # looser threshold for synonym-ish phrases
SYNON_MAX_RATIO = 0.90      # keep separate from distorted
HOMONYMOUS_MIN_CONTEXTS = 2 # if same label appears in >=2 distinct contexts strongly

# Polluted pattern: base + "_" + 5-12 alnum + "_" + yyyymmdd + space + digits(>=6)
POLLUTED_RE = re.compile(r"^(?P<base>.+?)_([A-Za-z0-9]{5,12})_(\d{8})\s+(\d{6,})$")

# quick tokenization
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# synonym cue verbs (domain-agnostic-ish)
SYN_VERBS = {
    "review": {"review", "assess", "evaluate", "inspect", "check", "examine"},
    "reject": {"reject", "deny", "decline", "refuse"},
    "approve": {"approve", "grant", "accept", "authorize", "confirm"},
    "diagnose": {"diagnose", "diagnosis", "determine", "establish", "confirm"},
    "start": {"start", "begin", "initiate", "launch"},
    "close": {"close", "finish", "complete", "terminate", "end"},
    "request": {"request", "ask", "require", "solicit"},
}

def norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_key(s: str) -> str:
    # lower + collapse spaces + remove punctuation except spaces
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    return TOKEN_RE.findall(norm_key(s))

def seq_ratio(a: str, b: str) -> float:
    # difflib is in C for core loops; still expensive if used too much.
    # We only use it against TOP_CANONICAL and only for suspicious rows.
    return SequenceMatcher(None, a, b).ratio()

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    # Read only required columns
    usecols = ["Case", "Activity", "Timestamp", "Resource"]
    df = pd.read_csv(IN_PATH, usecols=usecols)

    n = len(df)
    row_id = np.arange(n, dtype=np.int64)

    # Basic normalization
    act_raw = df["Activity"].astype("string")
    act = act_raw.fillna("").astype(str)
    act_norm = act.map(norm_text)
    act_key = act_norm.map(norm_key)

    # Timestamp parsing (vectorized)
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)

    case = df["Case"].astype("string").fillna("").astype(str)
    res = df["Resource"].astype("string")  # keep NaN; not an error

    # Prepare output containers (vectorized-friendly: store lists then join)
    err_flag = np.zeros(n, dtype=bool)
    err_conf = np.zeros(n, dtype=float)
    err_types = np.array([""] * n, dtype=object)
    err_tags = np.array([""] * n, dtype=object)
    err_evid = np.array([""] * n, dtype=object)
    err_desc = np.array([""] * n, dtype=object)

    def add_error(mask, etype, conf, tag, evidence, desc):
        nonlocal err_flag, err_conf, err_types, err_tags, err_evid, err_desc
        idx = np.where(mask)[0]
        if idx.size == 0:
            return
        err_flag[idx] = True
        # combine confidence: noisy-or
        err_conf[idx] = 1 - (1 - err_conf[idx]) * (1 - conf)

        def merge_pipe(old, new):
            if old == "":
                return new
            parts = set(old.split("|"))
            parts.add(new)
            return "|".join(sorted(parts))

        def merge_semicolon(old, new):
            if old == "":
                return new
            return old + " ; " + new

        # vectorized-ish updates via python loop over idx (small per rule typically)
        # but could be large; keep operations minimal
        for i in idx:
            err_types[i] = merge_pipe(err_types[i], etype)
            err_tags[i] = merge_pipe(err_tags[i], tag)
            err_evid[i] = merge_semicolon(err_evid[i], evidence[i] if isinstance(evidence, (np.ndarray, list, pd.Series)) else evidence)
            err_desc[i] = merge_semicolon(err_desc[i], desc[i] if isinstance(desc, (np.ndarray, list, pd.Series)) else desc)

    # -----------------------------
    # 0) Structural / format issues
    # -----------------------------
    # Missing Case / Activity / Timestamp are strong errors (Resource missing is allowed)
    miss_case = case.str.len().eq(0)
    miss_act = act_norm.str.len().eq(0)
    bad_ts = ts.isna()

    add_error(
        miss_case,
        "format",
        0.95,
        "missing_case",
        evidence=np.where(miss_case, "Case is empty/null", ""),
        desc=np.where(miss_case, "Missing Case identifier.", "")
    )
    add_error(
        miss_act,
        "format",
        0.95,
        "missing_activity",
        evidence=np.where(miss_act, "Activity is empty/null", ""),
        desc=np.where(miss_act, "Missing Activity label.", "")
    )
    add_error(
        bad_ts,
        "format",
        0.98,
        "invalid_timestamp",
        evidence=np.where(bad_ts, "Timestamp failed to parse", ""),
        desc=np.where(bad_ts, "Invalid/unparseable Timestamp.", "")
    )

    # -----------------------------
    # 1) POLLUTED (vectorized regex)
    # -----------------------------
    m = act_norm.str.match(POLLUTED_RE)
    polluted_base = act_norm.str.extract(POLLUTED_RE)["base"].fillna("")
    add_error(
        m.to_numpy(),
        "polluted",
        0.97,
        "polluted_suffix_pattern",
        evidence=np.where(m, "Activity matches pattern: <base>_<5-12alnum>_<yyyymmdd time>", ""),
        desc=np.where(m, "Activity label contains machine-generated suffix; base extracted.", "")
    )

    # For later canonical matching, use base for polluted rows
    act_for_match = act_key.copy()
    base_key = polluted_base.map(norm_key)
    act_for_match = np.where(m, base_key.to_numpy(), act_for_match.to_numpy())
    act_for_match = pd.Series(act_for_match, index=df.index)

    # -----------------------------
    # Canonical activity list (TOP N most frequent, excluding polluted suffixes)
    # -----------------------------
    # Use cleaned key (polluted -> base) to avoid suffix inflation
    freq = act_for_match.value_counts(dropna=False)
    canonical_keys = freq.head(TOP_CANONICAL).index.tolist()
    canonical_keys = [k for k in canonical_keys if isinstance(k, str) and k != ""]
    canonical_set = set(canonical_keys)

    # Map canonical key -> representative original label (most frequent raw among that key)
    # (fast groupby)
    rep = (
        pd.DataFrame({"key": act_for_match, "raw": act_norm})
        .groupby("key")["raw"]
        .agg(lambda s: s.value_counts().index[0])
    )
    canonical_rep = {k: rep.get(k, k) for k in canonical_keys}

    # -----------------------------
    # 2) DISTORTED / SYNONYMOUS (limited comparisons)
    # -----------------------------
    # Candidates: not already canonical, not empty, not missing, not polluted-only base already canonical
    candidate_mask = (~miss_act) & (~act_for_match.isin(canonical_set)) & (act_for_match.str.len() > 0)
    cand_idx = np.where(candidate_mask.to_numpy())[0]

    # Precompute token sets for canonical keys (cheap)
    canon_tokens = {k: set(tokens(k)) for k in canonical_keys}

    # For speed: only compare to canonicals sharing at least one token OR same first token
    cand_key = act_for_match.iloc[cand_idx].astype(str)
    cand_tokens = cand_key.map(lambda s: set(tokens(s)))
    cand_first = cand_key.map(lambda s: (tokens(s)[0] if tokens(s) else ""))

    canon_first = {k: (tokens(k)[0] if tokens(k) else "") for k in canonical_keys}

    best_match_key = np.array([""] * len(cand_idx), dtype=object)
    best_ratio = np.zeros(len(cand_idx), dtype=float)

    # Loop over candidates only (not full df), and compare to filtered canonicals (small)
    # This is the only Python loop; bounded by suspicious rows and TOP_CANONICAL.
    for j, (ckey, ctoks, cfirst) in enumerate(zip(cand_key.tolist(), cand_tokens.tolist(), cand_first.tolist())):
        # filter canonicals
        possible = []
        for k in canonical_keys:
            if canon_first[k] == cfirst and cfirst != "":
                possible.append(k)
            elif len(ctoks & canon_tokens[k]) > 0:
                possible.append(k)
        if not possible:
            # fallback: compare to top 15 canonicals only
            possible = canonical_keys[:15]

        br = 0.0
        bk = ""
        for k in possible:
            r = seq_ratio(ckey, k)
            if r > br:
                br = r
                bk = k
        best_ratio[j] = br
        best_match_key[j] = bk

    best_match_rep = np.array([canonical_rep.get(k, k) for k in best_match_key], dtype=object)

    # Distorted: high similarity to canonical but not equal
    distorted_mask_local = (best_ratio >= DISTORT_MIN_RATIO) & (best_ratio <= DISTORT_MAX_RATIO) & (best_match_key != "")
    distorted_idx = cand_idx[distorted_mask_local]
    distorted_evidence = np.array([""] * n, dtype=object)
    distorted_desc = np.array([""] * n, dtype=object)
    distorted_evidence[distorted_idx] = [
        f"Activity='{act_norm.iat[i]}' ~ canonical='{best_match_rep[np.where(distorted_idx==i)[0][0]]}' (ratio={best_ratio[distorted_mask_local][np.where(distorted_idx==i)[0][0]]:.3f})"
        for i in distorted_idx
    ]
    distorted_desc[distorted_idx] = [
        f"Likely typo/spelling distortion of '{best_match_rep[np.where(distorted_idx==i)[0][0]]}'."
        for i in distorted_idx
    ]
    add_error(
        np.isin(row_id, distorted_idx),
        "distorted",
        0.78,
        "string_similarity_to_canonical",
        evidence=distorted_evidence,
        desc=distorted_desc
    )

    # Synonymous: moderate similarity + verb synonym cue overlap
    # Build a quick verb-cue detector
    def verb_bucket(tokset):
        for bucket, words in SYN_VERBS.items():
            if len(tokset & words) > 0:
                return bucket
        return ""

    cand_bucket = np.array([verb_bucket(t) for t in cand_tokens.tolist()], dtype=object)
    match_bucket = np.array([verb_bucket(canon_tokens.get(k, set())) for k in best_match_key], dtype=object)

    synonymous_mask_local = (
        (best_ratio >= SYNON_MIN_RATIO) & (best_ratio < SYNON_MAX_RATIO) &
        (best_match_key != "") &
        (cand_bucket != "") & (cand_bucket == match_bucket)
    )
    syn_idx = cand_idx[synonymous_mask_local]
    syn_evidence = np.array([""] * n, dtype=object)
    syn_desc = np.array([""] * n, dtype=object)
    syn_evidence[syn_idx] = [
        f"Verb-synonym bucket='{cand_bucket[np.where(syn_idx==i)[0][0]]}', Activity='{act_norm.iat[i]}' -> canonical='{best_match_rep[np.where(syn_idx==i)[0][0]]}' (ratio={best_ratio[synonymous_mask_local][np.where(syn_idx==i)[0][0]]:.3f})"
        for i in syn_idx
    ]
    syn_desc[syn_idx] = [
        f"Activity wording appears synonymous with canonical '{best_match_rep[np.where(syn_idx==i)[0][0]]}'."
        for i in syn_idx
    ]
    add_error(
        np.isin(row_id, syn_idx),
        "synonymous",
        0.62,
        "verb_synonym_bucket+similarity",
        evidence=syn_evidence,
        desc=syn_desc
    )

    # -----------------------------
    # 3) COLLATERAL duplicates (vectorized within-case)
    # -----------------------------
    # Sort by case+timestamp for diff
    order = np.lexsort((ts.fillna(pd.Timestamp.min).to_numpy(), case.to_numpy()))
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    case_s = case.to_numpy()[order]
    act_s = act_for_match.to_numpy()[order]
    ts_s = ts.to_numpy()[order]
    res_s = res.astype("string").fillna("").astype(str).to_numpy()[order]

    same_case = case_s[1:] == case_s[:-1]
    same_act = act_s[1:] == act_s[:-1]
    same_res = res_s[1:] == res_s[:-1]
    # exact duplicate timestamp
    same_ts = ts_s[1:] == ts_s[:-1]

    # near duplicate: within window seconds (requires valid timestamps)
    dt = (ts_s[1:] - ts_s[:-1]) / np.timedelta64(1, "s")
    near = (dt >= 0) & (dt <= COLLATERAL_WINDOW_S)

    collateral_pair = same_case & same_act & same_res & (same_ts | near)
    collateral_idx_sorted = np.where(collateral_pair)[0] + 1  # mark the later one
    collateral_idx = order[collateral_idx_sorted]

    coll_evidence = np.array([""] * n, dtype=object)
    coll_desc = np.array([""] * n, dtype=object)
    coll_evidence[collateral_idx] = [
        f"Near/dup event: Case={df['Case'].iat[i]}, Activity='{df['Activity'].iat[i]}', Timestamp={df['Timestamp'].iat[i]}, Resource='{df['Resource'].iat[i]}'"
        for i in collateral_idx
    ]
    coll_desc[collateral_idx] = [
        f"Duplicate/near-duplicate event within {COLLATERAL_WINDOW_S}s for same case/activity/resource."
        for i in collateral_idx
    ]
    add_error(
        np.isin(row_id, collateral_idx),
        "collateral",
        0.90,
        "near_duplicate_within_case",
        evidence=coll_evidence,
        desc=coll_desc
    )

    # -----------------------------
    # 4) FORMBASED (same timestamp repeated in same case)
    # -----------------------------
    # Count events per (case, timestamp) where timestamp valid
    valid_ts_mask = ~bad_ts
    grp = (
        pd.DataFrame({"Case": case[valid_ts_mask], "Timestamp": ts[valid_ts_mask]})
        .groupby(["Case", "Timestamp"], sort=False)
        .size()
    )
    repeated = grp[grp >= FORMBASED_MIN_GROUP].reset_index().rename(columns={0: "cnt"})
    if len(repeated) > 0:
        # merge back to mark rows
        key_df = pd.DataFrame({"Case": case, "Timestamp": ts})
        rep_key = repeated[["Case", "Timestamp"]]
        form_mask = key_df.merge(rep_key.assign(_m=1), on=["Case", "Timestamp"], how="left")["_m"].fillna(0).to_numpy().astype(bool)
        # avoid flagging if it's just collateral duplicates already? still can be both; keep both.
        form_evidence = np.where(form_mask, "Same (Case, Timestamp) occurs >=3 times (form overwrite pattern)", "")
        form_desc = np.where(form_mask, "Multiple events share identical timestamp within same case (form-based timestamp overwrite).", "")
        add_error(
            form_mask,
            "formbased",
            0.72,
            "repeated_timestamp_within_case",
            evidence=form_evidence,
            desc=form_desc
        )

    # -----------------------------
    # 5) HOMONYMOUS (same label used in distinct contexts)
    # -----------------------------
    # Context signature: (prev_activity_key, next_activity_key) within same case order
    # Build per-case ordered sequences (using sorted order already)
    prev_act = np.empty(n, dtype=object)
    next_act = np.empty(n, dtype=object)
    prev_act[:] = ""
    next_act[:] = ""

    prev_act_sorted = np.empty(n, dtype=object)
    next_act_sorted = np.empty(n, dtype=object)
    prev_act_sorted[:] = ""
    next_act_sorted[:] = ""

    # compute prev/next in sorted arrays (vectorized with shifts + case boundaries)
    prev_act_sorted[1:] = act_s[:-1]
    next_act_sorted[:-1] = act_s[1:]
    # clear across case boundaries
    boundary = np.where(case_s[1:] != case_s[:-1])[0]
    prev_act_sorted[boundary + 1] = ""
    next_act_sorted[boundary] = ""

    # map back to original row order
    prev_act[order] = prev_act_sorted
    next_act[order] = next_act_sorted

    # For each activity key, count distinct context signatures among top canonicals only (to limit)
    act_key_series = act_for_match.astype(str)
    ctx = pd.Series(list(zip(prev_act, next_act)), index=df.index)

    # Only consider labels with enough occurrences to be meaningful
    occ = act_key_series.value_counts()
    candidate_labels = occ[(occ >= 20)].index  # avoid noise
    candidate_labels = [k for k in candidate_labels if k in canonical_set]  # focus on canonical labels

    homo_mask = np.zeros(n, dtype=bool)
    homo_evidence = np.array([""] * n, dtype=object)
    homo_desc = np.array([""] * n, dtype=object)

    if candidate_labels:
        sub = pd.DataFrame({"a": act_key_series, "ctx": ctx})
        sub = sub[sub["a"].isin(candidate_labels)]
        # distinct contexts per label
        ctx_counts = sub.groupby("a")["ctx"].nunique()
        homo_labels = ctx_counts[ctx_counts >= HOMONYMOUS_MIN_CONTEXTS].index.tolist()

        if homo_labels:
            # stronger: contexts should be "separated" (not just minor variation). Approx by top2 contexts share low overlap.
            # We'll mark rows of labels where top context frequency < 0.85 (i.e., multiple substantial contexts).
            freq_ctx = sub.groupby(["a", "ctx"]).size().reset_index(name="cnt")
            top_share = freq_ctx.sort_values(["a", "cnt"], ascending=[True, False]).groupby("a")["cnt"].apply(
                lambda s: (s.iloc[0] / s.sum()) if len(s) else 1.0
            )
            strong_homo = top_share[top_share < 0.85].index.tolist()

            if strong_homo:
                homo_mask = act_key_series.isin(strong_homo).to_numpy()
                # evidence: show (prev,next) for that row
                homo_evidence = np.where(
                    homo_mask,
                    "Same activity label appears in multiple substantial (prev,next) contexts within cases",
                    ""
                )
                homo_desc = np.where(
                    homo_mask,
                    "Potential homonymous label: used for different meanings across distinct process contexts.",
                    ""
                )
                add_error(
                    homo_mask,
                    "homonymous",
                    0.55,
                    "multi_context_signature",
                    evidence=homo_evidence,
                    desc=homo_desc
                )

    # -----------------------------
    # Post-processing: if polluted and base matches canonical, boost confidence slightly
    # -----------------------------
    polluted_idx = np.where(m.to_numpy())[0]
    if polluted_idx.size > 0:
        base_is_canon = base_key.isin(canonical_set).to_numpy()
        boost_mask = m.to_numpy() & base_is_canon
        err_conf[boost_mask] = np.minimum(1.0, err_conf[boost_mask] + 0.02)

    # Finalize output
    out = pd.DataFrame({
        "row_id": row_id,
        "error_flag": err_flag,
        "error_types": err_types,
        "error_confidence": np.round(err_conf, 4),
        "error_tags": err_tags,
        "error_evidence": err_evid,
        "error_description": err_desc,
    })

    ensure_dir(OUT_PATH)
    out.to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()