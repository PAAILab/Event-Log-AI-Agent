#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_multiple_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_multiple_noLabel.detected.csv

Detects errors ONLY using: Case, Activity, Timestamp, Resource
(Resource being empty is NOT an error.)
"""

import os
import re
import json
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_multiple_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_multiple_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]


# -----------------------------
# Utility: string normalization
# -----------------------------
def norm_space(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_key(s: str) -> str:
    """Aggressive normalization for comparing activity labels."""
    s = norm_space(s).lower()
    # keep alnum and spaces
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_set(s: str):
    s = norm_key(s)
    return set([t for t in s.split(" ") if t])

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def levenshtein(a: str, b: str) -> int:
    """DP Levenshtein distance (no external deps)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    a, b = str(a), str(b)
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        cur = [j] + [0] * n
        bj = b[j - 1]
        for i in range(1, n + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(
                prev[i] + 1,      # deletion
                cur[i - 1] + 1,   # insertion
                prev[i - 1] + cost  # substitution
            )
        prev = cur
    return prev[n]

def lev_ratio(a: str, b: str) -> float:
    a = norm_key(a)
    b = norm_key(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = levenshtein(a, b)
    return 1.0 - d / max(len(a), len(b), 1)

def safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# -----------------------------
# Polluted pattern
# -----------------------------
# Examples show: "<base>_<5-12 alnum>_<YYYYMMDD HHMMSSmmm000>"
POLLUTED_RE = re.compile(
    r"^(?P<base>.+?)_(?P<suffix>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6}\d{3}\d{3})$"
)

def parse_polluted(activity: str):
    s = norm_space(activity)
    m = POLLUTED_RE.match(s)
    if not m:
        return None
    base = norm_space(m.group("base"))
    suffix = m.group("suffix")
    dt = m.group("dt")
    return {"base": base, "suffix": suffix, "embedded_dt": dt}


# -----------------------------
# Confidence helpers
# -----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def combine_confidences(confs):
    """Noisy-OR combine."""
    p_not = 1.0
    for c in confs:
        p_not *= (1.0 - clamp01(c))
    return clamp01(1.0 - p_not)


# -----------------------------
# Main detection
# -----------------------------
def main():
    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # row_id = original row index (0-based)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    # Normalize core fields
    df["Case_n"] = df["Case"].astype(str)
    df["Activity_raw"] = df["Activity"].astype(str)
    df["Activity_n"] = df["Activity_raw"].map(norm_space)
    df["Activity_key"] = df["Activity_n"].map(norm_key)

    # Timestamp parsing
    df["Timestamp_raw"] = df["Timestamp"]
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp_raw"], errors="coerce", utc=False)

    # Resource: empty allowed; still normalize for comparisons
    df["Resource_n"] = df["Resource"].astype(str)
    # treat NaN as empty string for grouping comparisons
    df.loc[df["Resource"].isna(), "Resource_n"] = ""
    df["Resource_n"] = df["Resource_n"].map(norm_space)

    # Prepare output containers
    out = pd.DataFrame({"row_id": df["row_id"]})
    out["error_flag"] = False
    out["error_types"] = ""
    out["error_confidence"] = 0.0
    out["error_tags"] = ""
    out["error_evidence"] = ""
    out["error_description"] = ""

    # Per-row accumulators
    err_types = defaultdict(set)
    err_tags = defaultdict(set)
    err_evidence = defaultdict(list)
    err_conf = defaultdict(list)
    err_desc = defaultdict(list)

    # -----------------------------
    # Basic format/range checks (aggressive)
    # -----------------------------
    # Timestamp unparseable is a real error in event logs
    bad_ts = df["Timestamp_dt"].isna()
    for rid in df.loc[bad_ts, "row_id"].tolist():
        err_types[rid].add("timestamp_format")
        err_tags[rid].add("TS_PARSE_FAIL")
        raw = df.loc[rid, "Timestamp_raw"]
        err_evidence[rid].append({"timestamp_raw": None if pd.isna(raw) else str(raw)})
        err_conf[rid].append(0.95)
        err_desc[rid].append("Timestamp is missing or cannot be parsed into a datetime.")

    # Empty/Null Case is an error (cannot assign to trace)
    bad_case = df["Case"].isna() | (df["Case_n"].map(norm_space) == "")
    for rid in df.loc[bad_case, "row_id"].tolist():
        err_types[rid].add("case_missing")
        err_tags[rid].add("CASE_MISSING")
        err_evidence[rid].append({"case": None if pd.isna(df.loc[rid, "Case"]) else str(df.loc[rid, "Case"])})
        err_conf[rid].append(0.98)
        err_desc[rid].append("Case identifier is missing/empty, so the event cannot be assigned to a trace.")

    # Empty/Null Activity is an error
    bad_act = df["Activity"].isna() | (df["Activity_n"] == "")
    for rid in df.loc[bad_act, "row_id"].tolist():
        err_types[rid].add("activity_missing")
        err_tags[rid].add("ACT_MISSING")
        err_evidence[rid].append({"activity": None if pd.isna(df.loc[rid, "Activity"]) else str(df.loc[rid, "Activity"])})
        err_conf[rid].append(0.98)
        err_desc[rid].append("Activity label is missing/empty.")

    # -----------------------------
    # Polluted detection
    # -----------------------------
    polluted_info = df["Activity_n"].map(parse_polluted)
    is_polluted = polluted_info.notna()
    for idx in df.index[is_polluted]:
        rid = int(df.loc[idx, "row_id"])
        info = polluted_info.loc[idx]
        err_types[rid].add("polluted")
        err_tags[rid].add("ACT_POLLUTED_PATTERN")
        err_evidence[rid].append({"activity": df.loc[idx, "Activity_n"], "base_candidate": info["base"], "suffix": info["suffix"], "embedded_dt": info["embedded_dt"]})
        # strong evidence: exact pattern match
        err_conf[rid].append(0.97)
        err_desc[rid].append(f"Activity matches polluted pattern with machine suffix; base candidate='{info['base']}'.")

    # -----------------------------
    # Build canonical activity set (from "cleaner" labels)
    # Use activities that are NOT polluted and not empty.
    # -----------------------------
    clean_mask = (~is_polluted) & (~bad_act)
    clean_acts = df.loc[clean_mask, "Activity_n"].tolist()

    # If dataset is heavily corrupted, still build from most frequent normalized keys
    clean_keys = [norm_key(a) for a in clean_acts if norm_key(a)]
    key_counts = Counter(clean_keys)
    # canonical keys: frequent ones (top 200 or those with count>=2)
    canonical_keys = set([k for k, c in key_counts.items() if c >= 2])
    if len(canonical_keys) < 10:
        canonical_keys = set([k for k, _ in key_counts.most_common(200)])

    # Map canonical key -> representative label (most common original)
    rep_label = {}
    for k in canonical_keys:
        # pick most common original label among those with this key
        labels = df.loc[df["Activity_key"] == k, "Activity_n"].tolist()
        if labels:
            rep_label[k] = Counter(labels).most_common(1)[0][0]
        else:
            rep_label[k] = k

    canonical_list = list(canonical_keys)

    # -----------------------------
    # Distorted & Synonymous (data-driven, aggressive)
    # We do NOT have a domain synonym dictionary, so we infer:
    # - Distorted: very high character similarity to a canonical label but not equal
    # - Synonymous: high token overlap but lower character similarity (different wording)
    # -----------------------------
    def best_match(activity_key: str, activity_label: str):
        if not activity_key:
            return None
        # quick candidate pruning by token overlap with canonical reps
        a_tokens = token_set(activity_label)
        best = None
        for ck in canonical_list:
            rep = rep_label.get(ck, ck)
            jac = jaccard(a_tokens, token_set(rep))
            if jac < 0.34:
                continue
            lr = lev_ratio(activity_label, rep)
            score = 0.55 * jac + 0.45 * lr
            if (best is None) or (score > best["score"]):
                best = {"ck": ck, "rep": rep, "jac": jac, "lr": lr, "score": score}
        return best

    for idx in df.index:
        rid = int(df.loc[idx, "row_id"])
        if bad_act.loc[idx]:
            continue

        act = df.loc[idx, "Activity_n"]
        key = df.loc[idx, "Activity_key"]

        # If polluted, use base for matching too (can be polluted+distorted/synonymous)
        base_for_match = act
        if is_polluted.loc[idx]:
            base_for_match = polluted_info.loc[idx]["base"]

        bm = best_match(norm_key(base_for_match), base_for_match)
        if not bm:
            continue

        rep = bm["rep"]
        rep_key = norm_key(rep)
        base_key = norm_key(base_for_match)

        if base_key == rep_key:
            # exact match to canonical after normalization -> no distorted/synonymous
            continue

        # Distorted: very high lev ratio, small edit distance, same tokens mostly
        # Synonymous: decent token overlap but lower lev ratio (different words)
        lr = bm["lr"]
        jac = bm["jac"]

        # compute edit distance on normalized keys for evidence
        ed = levenshtein(base_key, rep_key)
        maxlen = max(len(base_key), len(rep_key), 1)

        if lr >= 0.90 and ed <= max(2, int(0.08 * maxlen)):
            err_types[rid].add("distorted")
            err_tags[rid].add("ACT_TYPO_SIMILAR")
            err_evidence[rid].append({
                "activity": act,
                "match_to": rep,
                "lev_ratio": round(lr, 3),
                "edit_distance": int(ed),
                "jaccard_tokens": round(jac, 3),
                "used_text": base_for_match
            })
            # confidence depends on similarity strength
            c = 0.75 + 0.25 * min(1.0, (lr - 0.90) / 0.10)
            err_conf[rid].append(clamp01(c))
            err_desc[rid].append(f"Activity looks like a typo/character distortion of '{rep}' (high string similarity).")
        elif jac >= 0.50 and lr <= 0.82:
            err_types[rid].add("synonymous")
            err_tags[rid].add("ACT_SYNONYM_INFERRED")
            err_evidence[rid].append({
                "activity": act,
                "match_to": rep,
                "lev_ratio": round(lr, 3),
                "jaccard_tokens": round(jac, 3),
                "used_text": base_for_match
            })
            # weaker than polluted/distorted because inferred without dictionary
            c = 0.55 + 0.25 * min(1.0, (jac - 0.50) / 0.50)
            err_conf[rid].append(clamp01(c))
            err_desc[rid].append(f"Activity wording differs but overlaps strongly with '{rep}' (likely synonym).")

    # -----------------------------
    # Collateral detection (duplicates / near-duplicates)
    # - Exact duplicates: same Case, Activity, Timestamp (resource ignored? include resource as context)
    # - Near duplicates: same Case+Activity+Resource within <= 3 seconds
    # -----------------------------
    # Exact duplicates
    grp_cols = ["Case_n", "Activity_n", "Timestamp_dt", "Resource_n"]
    exact_dup = df["Timestamp_dt"].notna()
    dup_groups = df.loc[exact_dup].groupby(grp_cols).size()
    dup_keys = dup_groups[dup_groups >= 2].index.tolist()

    if dup_keys:
        dup_df = df.loc[exact_dup].set_index(grp_cols)
        for k in dup_keys:
            rows = dup_df.loc[k, "row_id"]
            if isinstance(rows, pd.Series):
                rids = rows.astype(int).tolist()
            else:
                rids = [int(rows)]
            for rid in rids:
                err_types[rid].add("collateral")
                err_tags[rid].add("DUP_EXACT_CASE_ACT_TS_RES")
                err_evidence[rid].append({"group": {"Case": k[0], "Activity": k[1], "Timestamp": str(k[2]), "Resource": k[3]}, "duplicate_count": int(dup_groups.loc[k])})
                err_conf[rid].append(0.96)
                err_desc[rid].append("Exact duplicate event detected (same case, activity, timestamp, resource).")

    # Near-duplicates within a short interval
    # Sort within (Case, Activity, Resource)
    near_window_s = 3.0
    df_ts = df[df["Timestamp_dt"].notna()].copy()
    df_ts = df_ts.sort_values(["Case_n", "Activity_n", "Resource_n", "Timestamp_dt", "row_id"])

    for (case, act, res), g in df_ts.groupby(["Case_n", "Activity_n", "Resource_n"], sort=False):
        if len(g) < 2:
            continue
        t = g["Timestamp_dt"].values.astype("datetime64[ns]")
        rids = g["row_id"].astype(int).tolist()
        # compute deltas between consecutive events
        deltas = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
        for i, dt_s in enumerate(deltas):
            if dt_s <= near_window_s:
                # flag the later one (and also earlier if extremely tight)
                rid_late = rids[i + 1]
                rid_early = rids[i]
                for rid in [rid_early, rid_late]:
                    err_types[rid].add("collateral")
                    err_tags[rid].add("DUP_NEAR_CASE_ACT_RES")
                    err_evidence[rid].append({
                        "Case": case, "Activity": act, "Resource": res,
                        "delta_seconds": float(dt_s),
                        "pair_row_ids": [rid_early, rid_late]
                    })
                    # confidence increases as delta approaches 0
                    c = 0.70 + 0.25 * (1.0 - min(1.0, dt_s / near_window_s))
                    err_conf[rid].append(clamp01(c))
                    err_desc[rid].append(f"Near-duplicate events for same case/activity/resource within {near_window_s}s.")

    # -----------------------------
    # Form-based detection
    # Multiple different activities in same case share identical timestamp
    # (not just duplicates). Flag all but the first occurrence at that timestamp.
    # -----------------------------
    df_ts2 = df[df["Timestamp_dt"].notna()].copy()
    df_ts2 = df_ts2.sort_values(["Case_n", "Timestamp_dt", "row_id"])

    for (case, ts), g in df_ts2.groupby(["Case_n", "Timestamp_dt"], sort=False):
        if len(g) < 2:
            continue
        # if multiple distinct activities at same timestamp -> form-based overwrite likely
        distinct_acts = g["Activity_key"].nunique(dropna=True)
        if distinct_acts <= 1:
            continue

        # Flag all rows at that timestamp as formbased (aggressive),
        # but slightly lower confidence for the earliest row.
        rids = g["row_id"].astype(int).tolist()
        # evidence: show activities/resources at same timestamp
        snapshot = g[["row_id", "Activity_n", "Resource_n"]].to_dict("records")

        for j, rid in enumerate(rids):
            err_types[rid].add("formbased")
            err_tags[rid].add("SAME_TS_MULTI_ACT_IN_CASE")
            err_evidence[rid].append({
                "Case": case,
                "Timestamp": str(ts),
                "events_at_timestamp": snapshot,
                "distinct_activities": int(distinct_acts)
            })
            base_c = 0.78
            if j == 0:
                base_c -= 0.10  # earliest could be the true one
            err_conf[rid].append(clamp01(base_c))
            err_desc[rid].append("Multiple different activities share the exact same timestamp within a case (form-based overwrite pattern).")

    # -----------------------------
    # Homonymous detection (data-driven, conservative but aggressive where evidence strong)
    # Same activity label used in very different contexts:
    # - If an activity appears with two or more dominant predecessor activities in the same log,
    #   and those predecessor sets are dissimilar, mark those occurrences as homonymous candidates.
    # -----------------------------
    # Build predecessor per case sequence
    df_seq = df[df["Timestamp_dt"].notna() & (~bad_case) & (~bad_act)].copy()
    df_seq = df_seq.sort_values(["Case_n", "Timestamp_dt", "row_id"])
    df_seq["prev_act_key"] = df_seq.groupby("Case_n")["Activity_key"].shift(1)
    df_seq["prev_act_label"] = df_seq.groupby("Case_n")["Activity_n"].shift(1)

    # For each activity, distribution of predecessors
    pred_counts = df_seq.groupby(["Activity_key", "prev_act_key"]).size().reset_index(name="cnt")
    # compute top predecessors per activity
    homonymous_candidates = set()
    pred_map = defaultdict(list)
    for _, row in pred_counts.iterrows():
        ak = row["Activity_key"]
        pk = row["prev_act_key"]
        if pd.isna(ak) or pd.isna(pk):
            continue
        pred_map[ak].append((pk, int(row["cnt"])))

    for ak, lst in pred_map.items():
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
        if len(lst_sorted) < 2:
            continue
        top1, top2 = lst_sorted[0], lst_sorted[1]
        total = sum(c for _, c in lst_sorted)
        # require both predecessors to be substantial
        if total < 30:
            continue
        if top1[1] / total < 0.25 or top2[1] / total < 0.20:
            continue
        # if predecessor labels are very different (low token overlap), suspect homonym
        rep_act = rep_label.get(ak, ak)
        p1 = rep_label.get(top1[0], top1[0])
        p2 = rep_label.get(top2[0], top2[0])
        jacp = jaccard(token_set(p1), token_set(p2))
        if jacp <= 0.15:
            homonymous_candidates.add(ak)

    if homonymous_candidates:
        # Flag occurrences of those activities where predecessor is one of the competing modes
        for idx in df_seq.index:
            rid = int(df_seq.loc[idx, "row_id"])
            ak = df_seq.loc[idx, "Activity_key"]
            if ak not in homonymous_candidates:
                continue
            prevk = df_seq.loc[idx, "prev_act_key"]
            # evidence: show predecessor and activity
            err_types[rid].add("homonymous")
            err_tags[rid].add("ACT_MULTI_CONTEXT_PREDECESSOR")
            err_evidence[rid].append({
                "activity": df_seq.loc[idx, "Activity_n"],
                "prev_activity": None if pd.isna(prevk) else rep_label.get(prevk, str(prevk)),
                "case": df_seq.loc[idx, "Case_n"]
            })
            # moderate confidence: inferred semantics from context only
            err_conf[rid].append(0.58)
            err_desc[rid].append("Same activity label appears in strongly different predecessor contexts (possible homonym).")

    # -----------------------------
    # Finalize output
    # -----------------------------
    rows = []
    for rid in out["row_id"].tolist():
        types = sorted(list(err_types.get(rid, set())))
        tags = sorted(list(err_tags.get(rid, set())))
        confs = err_conf.get(rid, [])
        conf = combine_confidences(confs) if confs else 0.0
        flag = bool(types)

        evidence = err_evidence.get(rid, [])
        descs = err_desc.get(rid, [])

        rows.append({
            "row_id": int(rid),
            "error_flag": flag,
            "error_types": "|".join(types),
            "error_confidence": round(float(conf), 4),
            "error_tags": "|".join(tags),
            "error_evidence": safe_json(evidence),
            "error_description": " ".join(descs).strip()
        })

    out_df = pd.DataFrame(rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(out_df["error_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()