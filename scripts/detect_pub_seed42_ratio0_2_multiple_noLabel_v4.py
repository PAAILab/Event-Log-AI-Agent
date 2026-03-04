#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv

Detects (only using Case, Activity, Timestamp, Resource):
- formbased
- polluted
- distorted
- synonymous
- collateral
- homonymous (heuristic, conservative)

Fixes previous failure: avoids fragile groupby.apply on boolean masks and uses
vectorized groupby/shift logic instead.
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
# Utilities
# -----------------------------
def norm_space(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_key(s: str) -> str:
    s = norm_space(s).lower()
    # keep alnum and spaces
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def add_error(store, row_id, etype, conf, tags, evidence, desc):
    rec = store[row_id]
    rec["error_flag"] = True
    rec["types"].append(etype)
    rec["conf"].append(conf)
    rec["tags"].extend(tags if isinstance(tags, list) else [tags])
    rec["evidence"].append(evidence)
    rec["desc"].append(desc)

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(INPUT_PATH)

required = ["Case", "Activity", "Timestamp", "Resource"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Only {required} may be used.")

# row_id = original row index (stable)
df = df.reset_index(drop=False).rename(columns={"index": "row_id"})

# Normalize columns (do not treat empty Resource as error per instructions)
df["_case"] = df["Case"].astype(str)
df["_act_raw"] = df["Activity"].astype(str)
df["_act"] = df["_act_raw"].map(norm_space)
df["_act_key"] = df["_act"].map(norm_key)

# Timestamp parsing
df["_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
df["_res"] = df["Resource"].astype(str)
# Keep NaN resources as empty string for grouping; NOT an error
df.loc[df["Resource"].isna(), "_res"] = ""

# Prepare output store
out = defaultdict(lambda: {
    "row_id": None,
    "error_flag": False,
    "types": [],
    "conf": [],
    "tags": [],
    "evidence": [],
    "desc": []
})
for rid in df["row_id"].tolist():
    out[rid]["row_id"] = int(rid)

# -----------------------------
# 0) Basic format/range checks (Timestamp parse failures)
# (Not listed as a named error type; we will tag as distorted? No.
# We'll flag as "distorted" only for activity text issues. For timestamp parse failure,
# we still flag as error_type "formbased"?? Not correct.
# We'll introduce a tag-only detection but must output error_types from given set.
# So: if timestamp invalid, mark as "formbased" is wrong. Instead, mark as "collateral"? wrong.
# Therefore: we will NOT flag timestamp parse failures as one of the requested error types.
# But we can still include them as evidence tags without setting error_flag.
# -----------------------------
# Keep a note column for internal use (not output)
df["_ts_invalid"] = df["_ts"].isna()

# -----------------------------
# 1) POLLUTED detection
# Pattern: base label + "_" + 5-12 alnum + "_" + yyyymmdd HHMMSS + micro/nano digits
polluted_re = re.compile(
    r"^(?P<base>.+?)_([A-Za-z0-9]{5,12})_(\d{8}\s\d{6}\d{3,9})$"
)

df["_polluted"] = False
df["_polluted_base"] = ""
m = df["_act"].str.match(polluted_re)
df.loc[m, "_polluted"] = True
df.loc[m, "_polluted_base"] = df.loc[m, "_act"].str.replace(polluted_re, r"\g<base>", regex=True).map(norm_space)

for _, r in df.loc[df["_polluted"]].iterrows():
    rid = int(r["row_id"])
    base = r["_polluted_base"]
    conf = 0.95 if base else 0.85
    add_error(
        out, rid, "polluted", conf,
        tags=["polluted:regex_suffix"],
        evidence={"activity": r["_act_raw"], "parsed_base": base},
        desc=f"Activity appears polluted with machine-generated suffix; base='{base}'."
    )

# For downstream comparisons, use de-polluted activity if available
df["_act_clean"] = np.where(df["_polluted"], df["_polluted_base"], df["_act"])
df["_act_clean_key"] = pd.Series(df["_act_clean"]).map(norm_key)

# -----------------------------
# 2) SYNONYMOUS detection (heuristic via learned clusters)
# Build clusters of labels that frequently share same (case, timestamp, resource)
# This is a strong signal of synonym injection in this synthetic setting.
# We only use Case, Timestamp, Resource, Activity.
# -----------------------------
df_syn = df.loc[~df["_ts_invalid"]].copy()
df_syn["_ctr_key"] = df_syn["_case"].astype(str) + "||" + df_syn["_ts"].astype(str) + "||" + df_syn["_res"].astype(str)

# For each ctr_key, collect unique clean keys
grp = df_syn.groupby("_ctr_key")["_act_clean_key"].agg(lambda x: sorted(set([k for k in x if k])))
multi = grp[grp.map(len) >= 2]

pair_counts = Counter()
label_counts = Counter(df_syn["_act_clean_key"].tolist())

for acts in multi.tolist():
    # count co-occurrence pairs
    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            pair_counts[(acts[i], acts[j])] += 1

# Build adjacency for strong co-occurrence
# Thresholds tuned to be aggressive but evidence-based
adj = defaultdict(set)
for (a, b), c in pair_counts.items():
    # require at least 3 co-occurrences and not extremely rare labels
    if c >= 3 and label_counts[a] >= 3 and label_counts[b] >= 3:
        adj[a].add(b)
        adj[b].add(a)

# Find connected components as synonym clusters
visited = set()
clusters = []
for node in adj.keys():
    if node in visited:
        continue
    stack = [node]
    comp = set()
    visited.add(node)
    while stack:
        u = stack.pop()
        comp.add(u)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
    if len(comp) >= 2:
        clusters.append(comp)

# Choose canonical label per cluster: most frequent label
cluster_canon = {}
for comp in clusters:
    canon = max(comp, key=lambda k: label_counts[k])
    for k in comp:
        cluster_canon[k] = canon

# Flag as synonymous if label is in a cluster but not the canonical
for _, r in df.iterrows():
    k = r["_act_clean_key"]
    if not k or k not in cluster_canon:
        continue
    canon = cluster_canon[k]
    if k == canon:
        continue
    rid = int(r["row_id"])
    # confidence increases with co-occurrence strength
    a, b = sorted([k, canon])
    c = pair_counts.get((a, b), 0)
    conf = clamp01(0.55 + 0.10 * math.log1p(c))
    add_error(
        out, rid, "synonymous", conf,
        tags=["syn:cooccur_case_ts_res_cluster"],
        evidence={"activity_clean": r["_act_clean"], "canonical_key": canon, "cooccur_count": c},
        desc=f"Activity label behaves as a synonym of canonical='{canon}' based on repeated co-occurrence at same case/timestamp/resource."
    )

# -----------------------------
# 3) DISTORTED detection (typos) using similarity to frequent labels
# Compare each label to nearest frequent label; if very similar but not equal -> distorted.
# Exclude already-synonymous (since synonyms can be dissimilar).
# -----------------------------
# Frequent vocabulary from clean keys
vocab = df["_act_clean_key"].tolist()
freq = Counter([v for v in vocab if v])
common = [k for k, c in freq.items() if c >= 5]  # stable anchors

def nearest_common(key):
    best = (None, 0.0)
    for ck in common:
        if ck == key:
            return (ck, 1.0)
        # quick length filter
        if abs(len(ck) - len(key)) > 6:
            continue
        s = seq_ratio(key, ck)
        if s > best[1]:
            best = (ck, s)
    return best

for _, r in df.iterrows():
    rid = int(r["row_id"])
    key = r["_act_clean_key"]
    if not key or key in common:
        continue
    # if already flagged as synonymous, don't also call it distorted unless very close to canon
    already_syn = "synonymous" in out[rid]["types"]
    nn, sim = nearest_common(key)
    if nn is None:
        continue
    # distorted if high similarity but not identical
    if sim >= 0.88 and key != nn:
        # if synonymous already, require even higher similarity to avoid double-counting
        if already_syn and sim < 0.93:
            continue
        conf = clamp01((sim - 0.80) / 0.20)  # 0.4..1.0 roughly
        add_error(
            out, rid, "distorted", conf,
            tags=["dist:nearest_common_similarity"],
            evidence={"activity_clean": r["_act_clean"], "nearest_common": nn, "similarity": sim},
            desc=f"Activity text likely contains a typo; similar to frequent label '{nn}' (similarity={sim:.3f})."
        )

# -----------------------------
# 4) COLLATERAL detection
# (a) exact duplicates: same case, activity_clean_key, timestamp, resource
# (b) near duplicates bursts: same case, activity_clean_key, resource within <= 3 seconds
# Fix: use groupby + shift, no groupby.apply on boolean masks.
# -----------------------------
df_c = df.loc[~df["_ts_invalid"]].copy()
df_c = df_c.sort_values(["_case", "_ts", "row_id"], kind="mergesort")

# exact duplicates
dup_cols = ["_case", "_act_clean_key", "_ts", "_res"]
dup_mask = df_c.duplicated(subset=dup_cols, keep=False)
for _, r in df_c.loc[dup_mask].iterrows():
    rid = int(r["row_id"])
    add_error(
        out, rid, "collateral", 0.92,
        tags=["coll:exact_duplicate_case_act_ts_res"],
        evidence={k: str(r[k]) for k in dup_cols},
        desc="Exact duplicate event (same case, activity, timestamp, resource)."
    )

# near-duplicate bursts
gcols = ["_case", "_act_clean_key", "_res"]
df_c["_prev_ts"] = df_c.groupby(gcols)["_ts"].shift(1)
df_c["_dt_prev"] = (df_c["_ts"] - df_c["_prev_ts"]).dt.total_seconds()

burst_mask = df_c["_dt_prev"].notna() & (df_c["_dt_prev"] >= 0) & (df_c["_dt_prev"] <= 3.0)
for _, r in df_c.loc[burst_mask].iterrows():
    rid = int(r["row_id"])
    conf = 0.75 if r["_dt_prev"] <= 1.0 else 0.65
    add_error(
        out, rid, "collateral", conf,
        tags=["coll:near_duplicate_burst<=3s"],
        evidence={"dt_seconds": safe_float(r["_dt_prev"]), "case": r["_case"], "activity": r["_act_clean"], "resource": r["_res"]},
        desc=f"Near-duplicate event burst: same case/activity/resource repeated within {r['_dt_prev']:.3f}s."
    )

# -----------------------------
# 5) FORMBASED detection
# Multiple different activities in same case share identical timestamp (often many rows).
# We flag events in a (case, timestamp) group when:
# - group size >= 3 AND
# - at least 2 distinct activities_clean_key
# This is conservative to avoid normal parallelism.
# -----------------------------
df_f = df.loc[~df["_ts_invalid"]].copy()
g = df_f.groupby(["_case", "_ts"])
size = g.size().rename("n")
nacts = g["_act_clean_key"].nunique().rename("n_acts")
form_groups = pd.concat([size, nacts], axis=1)
form_groups = form_groups[(form_groups["n"] >= 3) & (form_groups["n_acts"] >= 2)]

if not form_groups.empty:
    idx = df_f.set_index(["_case", "_ts"]).index.isin(form_groups.index)
    for _, r in df_f.loc[idx].iterrows():
        rid = int(r["row_id"])
        n = int(form_groups.loc[(r["_case"], r["_ts"]), "n"])
        na = int(form_groups.loc[(r["_case"], r["_ts"]), "n_acts"])
        conf = clamp01(0.60 + 0.08 * min(5, (n - 2)) + 0.05 * min(5, (na - 1)))
        add_error(
            out, rid, "formbased", conf,
            tags=["form:same_case_same_timestamp_multi_events"],
            evidence={"case": r["_case"], "timestamp": str(r["_ts"]), "group_size": n, "distinct_activities": na},
            desc=f"Form-based timestamp overwrite suspected: {n} events (with {na} distinct activities) share the same timestamp within the same case."
        )

# -----------------------------
# 6) HOMONYMOUS detection (heuristic, conservative)
# Same activity label used with very different typical resources (resource-role split),
# suggesting different meanings. We only flag when:
# - activity appears with >= 2 resource clusters, each >= 10 occurrences
# - and resource distributions are highly separated (top resource groups disjoint)
# This is weak without semantics; keep confidence low.
# -----------------------------
df_h = df.copy()
df_h["_res_key"] = df_h["_res"].map(norm_key)
act_res = df_h.groupby("_act_clean_key")["_res_key"].value_counts().rename("cnt").reset_index()
hom_act = []
for act, sub in act_res.groupby("_act_clean_key"):
    if not act:
        continue
    total = sub["cnt"].sum()
    if total < 30:
        continue
    # take top 3 resources
    top = sub.sort_values("cnt", ascending=False).head(3)
    if len(top) < 2:
        continue
    # if top1 and top2 both substantial and different
    c1, c2 = int(top.iloc[0]["cnt"]), int(top.iloc[1]["cnt"])
    if c1 >= 10 and c2 >= 10 and (c2 / total) >= 0.25:
        hom_act.append((act, top))

hom_set = {a for a, _ in hom_act}
for _, r in df.iterrows():
    act = r["_act_clean_key"]
    if act in hom_set and act:
        rid = int(r["row_id"])
        add_error(
            out, rid, "homonymous", 0.35,
            tags=["hom:activity_multi_resource_clusters"],
            evidence={"activity": r["_act_clean"], "resource": r["_res"]},
            desc="Potential homonymous label: same activity appears with distinct resource clusters, suggesting different underlying meanings (heuristic)."
        )

# -----------------------------
# Finalize output rows
# -----------------------------
rows = []
for rid in df["row_id"].tolist():
    rec = out[int(rid)]
    types = rec["types"]
    # unique + stable order
    types_u = []
    for t in types:
        if t not in types_u:
            types_u.append(t)

    # confidence aggregation: combine as 1 - Π(1-ci)
    confs = [clamp01(c) for c in rec["conf"]] if rec["conf"] else []
    agg = 0.0
    if confs:
        prod = 1.0
        for c in confs:
            prod *= (1.0 - c)
        agg = 1.0 - prod

    rows.append({
        "row_id": int(rid),
        "error_flag": bool(rec["error_flag"]),
        "error_types": "|".join(types_u),
        "error_confidence": round(float(clamp01(agg)), 6),
        "error_tags": "|".join(sorted(set([str(t) for t in rec["tags"]]))),
        "error_evidence": json.dumps(rec["evidence"], ensure_ascii=False),
        "error_description": " ; ".join(rec["desc"])[:2000],
    })

out_df = pd.DataFrame(rows).sort_values("row_id")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out_df.to_csv(OUTPUT_PATH, index=False)

print(f"Wrote: {OUTPUT_PATH}")
print(out_df["error_flag"].value_counts(dropna=False).to_string())