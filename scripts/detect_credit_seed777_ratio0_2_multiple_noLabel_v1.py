#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/credit/credit_seed777_ratio0.2_multiple_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/credit_seed777_ratio0.2_multiple_noLabel.detected.csv

Detects errors using ONLY: Case, Activity, Timestamp, Resource
(Resource empty is NOT an error.)
"""

import os
import re
import math
import json
import difflib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed777_ratio0.2_multiple_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed777_ratio0.2_multiple_noLabel.detected.csv"

REQ_COLS = ["Case", "Activity", "Timestamp", "Resource"]


# -----------------------------
# Helpers
# -----------------------------
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def norm_space(s: str) -> str:
    s = safe_str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_case(s: str) -> str:
    return norm_space(s).lower()

def parse_ts(series: pd.Series) -> pd.Series:
    # aggressive parsing; keep NaT for invalid
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def add_finding(findings, idx, etype, conf, tag, evidence, desc):
    findings[idx]["types"].add(etype)
    findings[idx]["tags"].add(tag)
    findings[idx]["evidence"].append(evidence)
    findings[idx]["descs"].append(desc)
    # combine confidence conservatively: 1 - Π(1-ci)
    prev = findings[idx]["conf"]
    findings[idx]["conf"] = 1.0 - (1.0 - prev) * (1.0 - conf)

def base_activity_from_polluted(act: str):
    """
    Detect polluted suffix pattern:
      <label>_<5-12 alnum>_<YYYYMMDD HHMMSSmmm...>
    Return (is_polluted, base_label, suffix)
    """
    a = norm_space(act)
    # allow mixed case, spaces in label; suffix token 5-12 alnum
    m = re.match(r"^(.*)_([A-Za-z0-9]{5,12})_(\d{8}\s\d{6}\d{3,6})$", a)
    if not m:
        return False, None, None
    base = m.group(1).strip()
    return True, base, f"{m.group(2)}_{m.group(3)}"

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_case(a), norm_case(b)).ratio()

def tokenize(s: str):
    s = norm_case(s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks

def jaccard(a: str, b: str) -> float:
    A, B = set(tokenize(a)), set(tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def is_blank(s):
    return norm_space(s) == ""

def robust_mode(values):
    c = Counter(values)
    if not c:
        return None, 0
    v, n = c.most_common(1)[0]
    return v, n


# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(INPUT_PATH)

missing = [c for c in REQ_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

# Keep only required columns for detection (but we still output row_id only)
work = df[REQ_COLS].copy()

# Normalize
work["Case_n"] = work["Case"].astype(str)
work["Activity_raw"] = work["Activity"].apply(safe_str)
work["Activity_n"] = work["Activity_raw"].apply(norm_space)
work["Activity_lc"] = work["Activity_n"].apply(norm_case)
work["Resource_raw"] = work["Resource"].apply(safe_str)
work["Resource_n"] = work["Resource_raw"].apply(norm_space)  # empty allowed
work["Timestamp_raw"] = work["Timestamp"].apply(safe_str)
work["Timestamp_dt"] = parse_ts(work["Timestamp_raw"])

n_rows = len(work)

# Findings structure
findings = [
    {"types": set(), "tags": set(), "evidence": [], "descs": [], "conf": 0.0}
    for _ in range(n_rows)
]

# -----------------------------
# Basic format/range checks (Timestamp only; Resource empty is allowed)
# (Not listed as an error type in prompt; we will tag as "timestamp_invalid"
# but DO NOT set error_flag unless it supports one of the requested types.
# So we only use invalid timestamps as evidence for other types? Here: none.
# We'll still record as tag/evidence but not as an error type.
# -----------------------------
invalid_ts_idx = work.index[work["Timestamp_dt"].isna()].tolist()
for idx in invalid_ts_idx:
    # Not a requested error type; store as tag only (no error type)
    findings[idx]["tags"].add("timestamp_parse_failed")
    findings[idx]["evidence"].append(
        f"Timestamp '{work.at[idx,'Timestamp_raw']}' could not be parsed"
    )
    findings[idx]["descs"].append(
        "Timestamp format appears invalid/unparseable; cannot reliably order events in this case."
    )
    # no confidence update, no type

# -----------------------------
# 2) POLLUTED detection
# -----------------------------
polluted_base = {}
for idx, act in enumerate(work["Activity_n"].tolist()):
    is_pol, base, suffix = base_activity_from_polluted(act)
    if is_pol and base:
        polluted_base[idx] = base
        conf = 0.98  # strong regex evidence
        add_finding(
            findings, idx, "polluted", conf,
            "polluted_suffix_regex",
            f"Activity='{act}' matches pattern '<label>_<5-12alnum>_<YYYYMMDD HHMMSS...>'; base='{base}', suffix='{suffix}'",
            f"Activity label appears polluted by machine-generated suffix; canonical label likely '{base}'."
        )

# -----------------------------
# Build canonical activity set (from clean + de-polluted)
# -----------------------------
# Candidate canonical labels: all non-empty activities, but prefer de-polluted base
canon_candidates = []
for idx, act in enumerate(work["Activity_n"].tolist()):
    if is_blank(act):
        continue
    if idx in polluted_base:
        canon_candidates.append(polluted_base[idx])
    else:
        canon_candidates.append(act)

# Normalize canonical candidates by collapsing spaces
canon_candidates = [norm_space(x) for x in canon_candidates if not is_blank(x)]

# Frequency-based canonical list (top unique)
canon_counts = Counter([norm_case(x) for x in canon_candidates])
# Map lower->most common original casing (pick first seen)
lc_to_repr = {}
for x in canon_candidates:
    lc = norm_case(x)
    if lc not in lc_to_repr:
        lc_to_repr[lc] = x

# Keep all labels that appear at least twice OR are among top 200 (avoid tiny noise)
most_common = canon_counts.most_common()
canon_lc = set()
for lc, cnt in most_common:
    if cnt >= 2:
        canon_lc.add(lc)
# ensure some coverage
for lc, cnt in most_common[:200]:
    canon_lc.add(lc)

canon_list = [lc_to_repr[lc] for lc in canon_lc]

# -----------------------------
# 3) DISTORTED detection (typos) using similarity to canonical labels
#    - Exclude already polluted suffix part by using base if polluted
#    - Distorted if close to a canonical label but not equal (case-insensitive)
# -----------------------------
def best_match(label: str, candidates):
    best = None
    best_sim = -1.0
    best_j = -1.0
    for c in candidates:
        sim = similarity(label, c)
        if sim > best_sim:
            best_sim = sim
            best = c
            best_j = jaccard(label, c)
    return best, best_sim, best_j

for idx in range(n_rows):
    raw = work.at[idx, "Activity_n"]
    if is_blank(raw):
        continue

    label = polluted_base.get(idx, raw)  # de-pollute for matching
    # If exact canonical (case-insensitive), not distorted
    if norm_case(label) in canon_lc:
        continue

    bm, sim, jac = best_match(label, canon_list)
    if bm is None:
        continue

    # Aggressive but controlled thresholds:
    # - high character similarity but not identical
    # - and token overlap not too low (avoid unrelated)
    if sim >= 0.88 and jac >= 0.5:
        # confidence increases with similarity
        conf = clamp01(0.55 + (sim - 0.88) * 3.0)  # sim 0.88->0.55, 0.98->0.85
        conf = max(conf, 0.65 if sim >= 0.92 else conf)
        add_finding(
            findings, idx, "distorted", conf,
            "distorted_fuzzy_match",
            f"Activity='{label}' best_match='{bm}' similarity={sim:.3f} jaccard={jac:.3f}",
            f"Activity label likely contains a typo/spelling distortion; canonical label likely '{bm}'."
        )

# -----------------------------
# 4) SYNONYMOUS detection (heuristic)
#    We do NOT have a domain synonym dictionary; infer via:
#      - same case: two different labels used in similar positions/resources
#      - global: labels with high token overlap but low character similarity (different wording)
#    This is inherently weaker -> lower confidence.
# -----------------------------
# Build per-case sequences
work_valid_ts = work.copy()
# For ordering, put NaT at end
work_valid_ts["Timestamp_sort"] = work_valid_ts["Timestamp_dt"].fillna(pd.Timestamp.max)
work_valid_ts["row_id"] = np.arange(n_rows)

case_groups = work_valid_ts.sort_values(["Case_n", "Timestamp_sort", "row_id"]).groupby("Case_n")

# Collect adjacency contexts: (prev_label, next_label, resource) -> label
contexts = defaultdict(list)
labels_all = set()

for case, g in case_groups:
    acts = g["Activity_n"].tolist()
    ress = g["Resource_n"].tolist()
    rids = g["row_id"].tolist()
    for i, (a, r, rid) in enumerate(zip(acts, ress, rids)):
        if is_blank(a):
            continue
        labels_all.add(a)
        prev_a = acts[i-1] if i-1 >= 0 else ""
        next_a = acts[i+1] if i+1 < len(acts) else ""
        key = (norm_case(prev_a), norm_case(next_a), norm_case(r))
        contexts[key].append((a, rid))

# For each context, if multiple different labels appear, they may be synonyms
for key, items in contexts.items():
    labs = [norm_case(a) for a, _ in items if not is_blank(a)]
    if len(set(labs)) <= 1:
        continue
    # pick dominant label as canonical for this context
    dom_lc, dom_n = robust_mode(labs)
    if dom_lc is None:
        continue
    dom_label = None
    for a, _ in items:
        if norm_case(a) == dom_lc:
            dom_label = a
            break
    if dom_label is None:
        continue

    for a, rid in items:
        if norm_case(a) == dom_lc:
            continue
        # avoid calling distorted as synonymous if it's just a typo of dom_label
        sim = similarity(a, dom_label)
        jac = jaccard(a, dom_label)
        if sim >= 0.88:
            continue  # likely distortion, handled elsewhere
        if jac < 0.34:
            continue  # too different, weak
        # confidence depends on how strong the context dominance is
        total = len(items)
        dominance = dom_n / total
        conf = clamp01(0.35 + 0.4 * (dominance - 0.5) + 0.3 * jac)  # typically 0.35-0.75
        add_finding(
            findings, rid, "synonymous", conf,
            "syn_context_same_prev_next_resource",
            f"context(prev='{key[0]}', next='{key[1]}', res='{key[2]}') label='{a}' dominant='{dom_label}' dominance={dominance:.2f} jaccard={jac:.2f} sim={sim:.2f}",
            f"Activity label '{a}' appears to be a synonym/alternative wording for '{dom_label}' in the same process context."
        )

# -----------------------------
# 5) COLLATERAL detection (duplicates / near-duplicates)
#    - exact duplicates: same Case, Activity, Timestamp (and Resource if present)
#    - near duplicates: same Case, Activity, Resource within short interval (<= 3 seconds)
# -----------------------------
# Exact duplicates
key_cols = ["Case_n", "Activity_lc", "Timestamp_raw", "Resource_n"]
dup_groups = work.groupby(key_cols, dropna=False).indices
for k, idxs in dup_groups.items():
    if len(idxs) >= 2:
        for idx in idxs:
            conf = 0.97 if len(idxs) > 2 else 0.93
            add_finding(
                findings, idx, "collateral", conf,
                "collateral_exact_duplicate",
                f"Duplicate group size={len(idxs)} for (Case='{k[0]}', Activity='{k[1]}', Timestamp='{k[2]}', Resource='{k[3]}') rows={list(map(int, idxs))}",
                "Exact duplicate event detected (same case, activity, timestamp, resource)."
            )

# Near duplicates within case
# Sort by case/activity/resource/time
tmp = work.copy()
tmp["row_id"] = np.arange(n_rows)
tmp = tmp.sort_values(["Case_n", "Activity_lc", "Resource_n", "Timestamp_dt", "row_id"])
for (case, act, res), g in tmp.groupby(["Case_n", "Activity_lc", "Resource_n"], dropna=False):
    if len(g) < 2:
        continue
    times = g["Timestamp_dt"].tolist()
    rids = g["row_id"].tolist()
    for i in range(1, len(g)):
        t0, t1 = times[i-1], times[i]
        if pd.isna(t0) or pd.isna(t1):
            continue
        dt = (t1 - t0).total_seconds()
        if 0.0 < dt <= 3.0:
            # mark the later one (and optionally earlier) as collateral
            conf = clamp01(0.75 + (3.0 - dt) * 0.07)  # dt=0.1 -> ~0.95, dt=3 -> 0.75
            add_finding(
                findings, rids[i], "collateral", conf,
                "collateral_near_duplicate_3s",
                f"Near-duplicate within {dt:.3f}s: Case='{case}', Activity='{act}', Resource='{res}', prev_row={int(rids[i-1])}, row={int(rids[i])}",
                f"Near-duplicate event detected: same case/activity/resource repeated within {dt:.3f}s."
            )

# -----------------------------
# 1) FORMBASED detection (same timestamp repeated within a case)
#    Heuristic:
#      - within same case, if >=3 events share identical timestamp (string or parsed)
#      - and activities are different (not just duplicates) -> likely form-based overwrite
#    We flag all but the first occurrence at that timestamp as formbased.
# -----------------------------
case_ts_groups = work_valid_ts.sort_values(["Case_n", "Timestamp_sort", "row_id"]).groupby(["Case_n", "Timestamp_raw"], dropna=False)
for (case, ts_raw), g in case_ts_groups:
    if is_blank(ts_raw):
        continue
    if len(g) < 3:
        continue
    # require at least 2 distinct activities to avoid pure duplicates
    acts = [norm_case(a) for a in g["Activity_n"].tolist() if not is_blank(a)]
    if len(set(acts)) < 2:
        continue
    # If many share same timestamp, likely formbased for later ones
    rids = g["row_id"].tolist()
    # keep first as "possibly correct", flag rest
    for rid in rids[1:]:
        conf = clamp01(0.70 + 0.05 * min(6, len(g)))  # size 3->0.85, 7+->1.0 capped
        add_finding(
            findings, rid, "formbased", conf,
            "formbased_same_timestamp_cluster",
            f"Case='{case}' has {len(g)} events with identical Timestamp='{ts_raw}' (distinct_activities={len(set(acts))}); flagged row={int(rid)}",
            "Multiple different events share the exact same timestamp within a case, consistent with form-based timestamp overwrite."
        )

# -----------------------------
# 6) HOMONYMOUS detection (same label used for different meanings)
#    Without semantic ground truth, use a strong proxy:
#      - same activity label appears in two distinct, stable contexts (prev/next) that are
#        mutually exclusive and both frequent, suggesting different meanings.
#    This is weak; keep confidence low and only flag when evidence is strong.
# -----------------------------
# Build context signatures for each activity label
act_context_counts = defaultdict(Counter)  # act_lc -> Counter(context_signature)
act_rows_by_context = defaultdict(lambda: defaultdict(list))  # act_lc -> context -> [row_ids]

for case, g in case_groups:
    acts = g["Activity_n"].tolist()
    rids = g["row_id"].tolist()
    for i, (a, rid) in enumerate(zip(acts, rids)):
        if is_blank(a):
            continue
        prev_a = acts[i-1] if i-1 >= 0 else ""
        next_a = acts[i+1] if i+1 < len(acts) else ""
        ctx = (norm_case(prev_a), norm_case(next_a))
        alc = norm_case(a)
        act_context_counts[alc][ctx] += 1
        act_rows_by_context[alc][ctx].append(rid)

for alc, ctx_counter in act_context_counts.items():
    total = sum(ctx_counter.values())
    if total < 30:
        continue  # need enough evidence
    top = ctx_counter.most_common(3)
    if len(top) < 2:
        continue
    (ctx1, n1), (ctx2, n2) = top[0], top[1]
    # both contexts must be substantial and different
    if n1 < 10 or n2 < 10:
        continue
    if ctx1 == ctx2:
        continue
    # contexts should be "far": different prev and different next
    if ctx1[0] == ctx2[0] and ctx1[1] == ctx2[1]:
        continue
    # If contexts are very different, suspect homonymy
    # Flag rows belonging to minority context (ctx2) as homonymous relative to dominant meaning (ctx1)
    dominance = n1 / total
    if dominance < 0.55:
        continue  # not a clear dominant meaning
    # confidence modest; this is proxy-based
    conf = clamp01(0.35 + 0.5 * (dominance - 0.55) + 0.15 * (min(n2, n1) / max(n2, n1)))
    # Only if conf reaches a minimum
    if conf < 0.45:
        continue

    # Representative label for reporting
    label_repr = lc_to_repr.get(alc, alc)

    for rid in act_rows_by_context[alc][ctx2]:
        add_finding(
            findings, rid, "homonymous", conf,
            "homonymous_multi_context_prev_next",
            f"Activity='{label_repr}' appears in two frequent contexts: dominant(prev='{ctx1[0]}', next='{ctx1[1]}', n={n1}) vs alt(prev='{ctx2[0]}', next='{ctx2[1]}', n={n2}); total={total}",
            f"Same activity label '{label_repr}' is used in a substantially different process context, suggesting a possible homonymous meaning."
        )

# -----------------------------
# Compose output
# -----------------------------
out = pd.DataFrame({"row_id": np.arange(n_rows, dtype=int)})

def types_pipe(s):
    if not s:
        return ""
    return "|".join(sorted(s))

def tags_pipe(s):
    if not s:
        return ""
    return "|".join(sorted(s))

error_types_list = []
error_flags = []
conf_list = []
tags_list = []
evidence_list = []
desc_list = []

for idx in range(n_rows):
    f = findings[idx]
    types = f["types"]
    # Only requested error types count for error_flag
    error_flag = len(types) > 0
    error_flags.append(bool(error_flag))
    error_types_list.append(types_pipe(types))
    conf_list.append(round(clamp01(f["conf"]), 4) if error_flag else 0.0)
    tags_list.append(tags_pipe(f["tags"]))
    # evidence/desc: keep compact but specific
    evidence_list.append(" || ".join(f["evidence"])[:20000])
    desc_list.append(" ".join(f["descs"])[:20000])

out["error_flag"] = error_flags
out["error_types"] = error_types_list
out["error_confidence"] = conf_list
out["error_tags"] = tags_list
out["error_evidence"] = evidence_list
out["error_description"] = desc_list

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Saved detected errors to: {OUTPUT_PATH}")
print("Error type counts:")
print(out.loc[out["error_flag"], "error_types"].value_counts().head(30).to_string())