#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection (PUB dataset)

Goal: detect erroneous rows using ONLY: Case, Activity, Timestamp, Resource.

Fixes vs prior low-quality attempt:
- Remove/disable "homonymous" detection (was massively over-flagging).
- Make rules conservative to avoid flagging huge portions of the log.
- Focus on high-evidence patterns: polluted suffixes, timestamp collisions (form-based),
  near-duplicate bursts (collateral), and typo-like distortions (only when strongly supported).
- Resource being empty is NOT an error (explicitly ignored).

Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv
"""

import os
import re
import json
import math
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict, Counter

INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv"

# -----------------------------
# Helpers
# -----------------------------
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def norm_activity(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    # keep case for evidence, but normalize for comparisons
    return norm_space(s)

def norm_activity_cmp(s: str) -> str:
    s = norm_activity(s).lower()
    # normalize punctuation spacing lightly
    s = re.sub(r"\s*\(\s*", " (", s)
    s = re.sub(r"\s*\)\s*", ") ", s)
    s = norm_space(s)
    return s

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def safe_bool(x):
    return bool(x) and str(x).lower() not in ("nan", "none")

def add_err(errs, etype, conf, tag, evidence, desc):
    errs["types"].append(etype)
    errs["conf"].append(float(conf))
    errs["tags"].append(tag)
    errs["evidence"].append(evidence)
    errs["desc"].append(desc)

def finalize_row(errs):
    if not errs["types"]:
        return False, "", 0.0, "", "", ""
    # merge types unique but keep stable order
    seen = set()
    types = []
    for t in errs["types"]:
        if t not in seen:
            seen.add(t)
            types.append(t)
    # confidence: combine conservatively (noisy-or)
    # p = 1 - Π(1-ci)
    p = 1.0
    for c in errs["conf"]:
        p *= (1.0 - max(0.0, min(1.0, c)))
    conf = 1.0 - p
    tags = "|".join(sorted(set(errs["tags"])))
    evidence = " | ".join(errs["evidence"][:6])  # cap length
    desc = " ; ".join(errs["desc"][:4])
    return True, "|".join(types), float(round(conf, 4)), tags, evidence, desc

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(INPUT_PATH)

required = ["Case", "Activity", "Timestamp", "Resource"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

# row_id = original row index (0-based)
df = df.reset_index(drop=True)
df["row_id"] = df.index.astype(int)

# Parse timestamp
ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
df["_ts"] = ts
df["_case"] = df["Case"].astype(str)
df["_act_raw"] = df["Activity"].astype(str)
df["_act"] = df["_act_raw"].map(norm_activity)
df["_act_cmp"] = df["_act_raw"].map(norm_activity_cmp)
# Resource empty is allowed; still use it for duplicate-context checks when present
df["_res"] = df["Resource"].astype(str)

# -----------------------------
# Rule 0: basic format issues (very conservative)
# (Not requested explicitly, but helps catch truly broken rows without over-flagging)
# - Missing/invalid timestamp is almost certainly erroneous for event logs.
# - Missing case or activity is also likely erroneous.
# -----------------------------
invalid_ts = df["_ts"].isna()
missing_case = df["_case"].isna() | (df["_case"].str.strip() == "") | (df["_case"].str.lower() == "nan")
missing_act = df["_act"].isna() | (df["_act"].str.strip() == "") | (df["_act"].str.lower() == "nan")

# -----------------------------
# Rule 1: POLLUTED
# Pattern: "<base>_<5-12 alnum>_<YYYYMMDD HHMMSS...>" (as in examples)
# Keep strict to avoid false positives.
# -----------------------------
polluted_re = re.compile(
    r"^(?P<base>.+?)_(?P<code>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s+\d{6,}\d*)$"
)
polluted_match = df["_act"].str.match(polluted_re)

# -----------------------------
# Rule 2: FORM-BASED
# Detect within-case timestamp collisions where multiple different activities share exact same timestamp.
# Conservative: require >=3 events at same timestamp in same case AND >=2 distinct activities.
# (This matches the "form overwrite" symptom without flagging normal concurrency too much.)
# -----------------------------
# Only consider rows with valid timestamps
df_valid_ts = df[~invalid_ts].copy()

grp = df_valid_ts.groupby(["_case", "_ts"])
size = grp.size()
n_acts = grp["_act_cmp"].nunique()

form_groups = set(
    (case, ts_val)
    for (case, ts_val), cnt in size.items()
    if cnt >= 3 and n_acts.loc[(case, ts_val)] >= 2
)

is_formbased = df_valid_ts.apply(lambda r: (r["_case"], r["_ts"]) in form_groups, axis=1)
formbased_row_ids = set(df_valid_ts.loc[is_formbased, "row_id"].tolist())

# -----------------------------
# Rule 3: COLLATERAL
# Detect duplicates / near-duplicates:
# A) exact duplicates: same case, activity, timestamp, resource (resource may be empty; still counts)
# B) burst duplicates: same case+activity+resource repeated within <=2 seconds (and not same timestamp)
# Conservative thresholds to avoid over-flagging.
# -----------------------------
# A) exact duplicates
dup_cols = ["_case", "_act_cmp", "_ts", "_res"]
exact_dup_mask = (~invalid_ts) & df.duplicated(subset=dup_cols, keep=False)

# B) burst duplicates within case+activity+resource
df_sorted = df[~invalid_ts].sort_values(["_case", "_act_cmp", "_res", "_ts", "row_id"]).copy()
df_sorted["_prev_ts"] = df_sorted.groupby(["_case", "_act_cmp", "_res"])["_ts"].shift(1)
df_sorted["_dt_s"] = (df_sorted["_ts"] - df_sorted["_prev_ts"]).dt.total_seconds()
burst_mask_sorted = df_sorted["_dt_s"].notna() & (df_sorted["_dt_s"] > 0) & (df_sorted["_dt_s"] <= 2.0)

burst_row_ids = set(df_sorted.loc[burst_mask_sorted, "row_id"].tolist())
# also mark the previous row in the burst (the "source" duplicate)
prev_row_ids = set(df_sorted.loc[burst_mask_sorted].groupby(["_case", "_act_cmp", "_res"]).apply(
    lambda g: g["row_id"].shift(1)
).dropna().astype(int).tolist())
burst_row_ids |= prev_row_ids

is_collateral = exact_dup_mask | df["row_id"].isin(burst_row_ids)

# -----------------------------
# Rule 4: DISTORTED (typos)
# Hard part without a dictionary. Use a conservative, data-driven approach:
# - Build canonical candidates as the most frequent "clean" base labels.
# - Only flag an activity as distorted if:
#   * it is NOT polluted
#   * it is rare (frequency <= 2)
#   * it is very similar to a frequent label (ratio >= 0.92)
#   * and differs by small edit footprint (length diff <= 3)
# This avoids flagging legitimate variants.
# -----------------------------
act_freq = df["_act_cmp"].value_counts(dropna=False)

# Frequent labels as potential canonicals
canonicals = act_freq[act_freq >= 30].index.tolist()  # frequency threshold
canon_set = set(canonicals)

def best_canonical(a):
    # return (best_label, best_ratio)
    best = None
    best_r = 0.0
    for c in canonicals:
        # quick filters
        if abs(len(a) - len(c)) > 3:
            continue
        r = seq_ratio(a, c)
        if r > best_r:
            best_r = r
            best = c
    return best, best_r

distorted_row_ids = set()
distorted_map = {}  # row_id -> canonical
if canonicals:
    # candidates: rare, non-empty, not canonical, not polluted
    candidates = df.loc[
        (~missing_act)
        & (~polluted_match)
        & (~df["_act_cmp"].isin(canon_set))
        & (df["_act_cmp"].map(lambda x: act_freq.get(x, 0) <= 2))
    , ["row_id", "_act_cmp"]].copy()

    for rid, a in candidates.itertuples(index=False):
        if not safe_bool(a):
            continue
        c, r = best_canonical(a)
        if c is not None and r >= 0.92:
            distorted_row_ids.add(int(rid))
            distorted_map[int(rid)] = {"canonical": c, "similarity": float(r)}

is_distorted = df["row_id"].isin(distorted_row_ids)

# -----------------------------
# Rule 5: SYNONYMOUS
# Without domain ontology, do NOT guess synonyms aggressively.
# Implement only a tiny, high-precision heuristic:
# - If two labels differ only by common verb synonyms AND appear in same case at same timestamp
#   with same resource (suggesting same event recorded with alternate wording).
# This is rare but high-evidence.
# -----------------------------
# Minimal synonym normalization map (very small to keep precision high)
syn_map = {
    "assess": "review",
    "evaluate": "review",
    "inspect": "review",
    "deny": "reject",
    "decline": "reject",
    "refuse": "reject",
    "begin": "start",
    "initiate": "start",
    "launch": "start",
    "establish": "determine",
    "confirm": "determine",
}
def synonym_key(label_cmp: str) -> str:
    toks = label_cmp.split()
    if not toks:
        return label_cmp
    t0 = toks[0]
    if t0 in syn_map:
        toks[0] = syn_map[t0]
    return " ".join(toks)

df["_syn_key"] = df["_act_cmp"].map(synonym_key)

# same case+timestamp+resource, different raw label but same synonym key => likely synonymous labeling
syn_groups = df.loc[~invalid_ts].groupby(["_case", "_ts", "_res", "_syn_key"])
syn_flag_ids = set()
for _, g in syn_groups:
    if len(g) >= 2 and g["_act_cmp"].nunique() >= 2:
        # require both labels reasonably frequent to avoid distorted being mis-tagged as synonym
        labels = g["_act_cmp"].tolist()
        if all(act_freq.get(x, 0) >= 5 for x in set(labels)):
            syn_flag_ids |= set(g["row_id"].astype(int).tolist())

is_synonymous = df["row_id"].isin(syn_flag_ids)

# -----------------------------
# HOMONYMOUS: DISABLED (too error-prone without semantics/labels)
# -----------------------------

# -----------------------------
# Build output
# -----------------------------
out_rows = []
for i, r in df.iterrows():
    errs = {"types": [], "conf": [], "tags": [], "evidence": [], "desc": []}

    rid = int(r["row_id"])
    case = r["_case"]
    act = r["_act"]
    ts_val = r["_ts"]
    res = r["_res"]

    # Basic format
    if missing_case.iloc[i]:
        add_err(
            errs, "format", 0.95, "R0_MISSING_CASE",
            f"Case='{r['Case']}'", "Missing/blank Case identifier"
        )
    if missing_act.iloc[i]:
        add_err(
            errs, "format", 0.95, "R0_MISSING_ACTIVITY",
            f"Activity='{r['Activity']}'", "Missing/blank Activity label"
        )
    if invalid_ts.iloc[i]:
        add_err(
            errs, "format", 0.98, "R0_INVALID_TIMESTAMP",
            f"Timestamp='{r['Timestamp']}'", "Invalid/unparseable Timestamp"
        )

    # Polluted
    if bool(polluted_match.iloc[i]):
        m = polluted_re.match(act)
        base = m.group("base") if m else ""
        add_err(
            errs, "polluted", 0.97, "R1_POLLUTED_SUFFIX",
            f"Activity='{act}' base='{base}'",
            "Activity has machine-generated suffix pattern"
        )

    # Form-based
    if rid in formbased_row_ids:
        # evidence: group size
        if pd.notna(ts_val):
            key = (case, ts_val)
            cnt = int(size.loc[key])
            uniq = int(n_acts.loc[key])
            add_err(
                errs, "form-based", 0.85, "R2_SAME_TS_MULTI_EVENTS",
                f"case={case} ts={ts_val} events_at_ts={cnt} distinct_activities={uniq}",
                "Multiple events in same case share identical timestamp (form-based overwrite suspected)"
            )

    # Collateral
    if bool(is_collateral.iloc[i]):
        if bool(exact_dup_mask.iloc[i]):
            add_err(
                errs, "collateral", 0.92, "R3_EXACT_DUP",
                f"(case,act,ts,res)=({case},{r['_act_cmp']},{ts_val},{res}) duplicated",
                "Exact duplicate event detected"
            )
        elif rid in burst_row_ids:
            add_err(
                errs, "collateral", 0.78, "R3_BURST_DUP_2S",
                f"Repeated (case,act,res)=({case},{r['_act_cmp']},{res}) within <=2s",
                "Near-duplicate burst detected within implausibly short interval"
            )

    # Distorted
    if bool(is_distorted.iloc[i]):
        info = distorted_map.get(rid, {})
        add_err(
            errs, "distorted", 0.72 + 0.25 * max(0.0, min(1.0, info.get("similarity", 0.92) - 0.92) / 0.08),
            "R4_RARE_SIMILAR_TO_FREQUENT",
            f"Activity='{r['_act_cmp']}' ~ '{info.get('canonical')}' sim={info.get('similarity'):.3f}",
            "Rare activity label is highly similar to a frequent label (likely typo/distortion)"
        )

    # Synonymous (very conservative)
    if bool(is_synonymous.iloc[i]):
        add_err(
            errs, "synonymous", 0.70, "R5_SAME_EVENT_DIFF_WORDING",
            f"case={case} ts={ts_val} res={res} syn_key='{r['_syn_key']}' labels_in_group>1",
            "Same case/timestamp/resource has multiple labels that normalize to same synonym key"
        )

    error_flag, error_types, error_conf, error_tags, error_evidence, error_desc = finalize_row(errs)

    out_rows.append({
        "row_id": rid,
        "error_flag": error_flag,
        "error_types": error_types,
        "error_confidence": error_conf,
        "error_tags": error_tags,
        "error_evidence": error_evidence,
        "error_description": error_desc
    })

out = pd.DataFrame(out_rows)

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Wrote: {OUTPUT_PATH}")
print("Summary:")
print(out["error_flag"].value_counts(dropna=False).to_string())
print("Error type counts:")
type_counts = Counter()
for s in out.loc[out["error_flag"], "error_types"].astype(str):
    for t in s.split("|"):
        if t:
            type_counts[t] += 1
print(pd.Series(type_counts).sort_values(ascending=False).to_string())