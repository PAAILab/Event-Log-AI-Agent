#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection (Credit dataset)

Goal: detect erroneous rows using ONLY: Case, Activity, Timestamp, Resource.
Output: /home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_synonymous_noLabel.detected.csv

Fixes vs prior over-flagging:
- Do NOT attempt broad "synonymous" detection (requires domain dictionary; caused massive false positives).
- Do NOT attempt "homonymous" detection (semantic; cannot be inferred reliably from these 4 columns).
- Distorted detection is conservative: only flags when activity is very close to a high-frequency canonical label.
- Polluted detection uses strong regex evidence.
- Formbased detection only when same-case same-timestamp repeats across >=3 events (strong evidence).
- Collateral detection only for exact duplicates OR very tight repeats (<=2s) with same case+activity+resource.
"""

import os
import re
import json
import math
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict, Counter

INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_synonymous_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_synonymous_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# -----------------------------
# Helpers
# -----------------------------
def norm_space(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def base_activity_for_similarity(s: str) -> str:
    """Normalize activity for similarity comparisons (lowercase, collapse spaces, remove obvious polluted suffix)."""
    s = norm_space(s).lower()

    # Remove common polluted suffix patterns if present (but do not *assume* it's polluted unless regex matches later)
    # e.g., "Request Info_47xiDPl_20230929 130852312000"
    s = re.sub(r"_[a-z0-9]{5,12}_[0-9]{8}\s[0-9]{9,15}$", "", s, flags=re.IGNORECASE)
    # also handle "..._47xiDPl_20230929130852312000" (no space)
    s = re.sub(r"_[a-z0-9]{5,12}_[0-9]{17,23}$", "", s, flags=re.IGNORECASE)

    # Remove stray punctuation differences
    s = re.sub(r"[^\w\s\-\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(INPUT_PATH)

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

# Keep only required columns for detection (but we still output row_id aligned to original rows)
work = df[REQUIRED_COLS].copy()

# row_id = original row index (0-based)
work["row_id"] = np.arange(len(work), dtype=int)

# Parse timestamps
work["Timestamp_raw"] = work["Timestamp"]
work["Timestamp"] = pd.to_datetime(work["Timestamp"], errors="coerce", utc=False)

# Normalize strings
work["Case_n"] = work["Case"].astype(str)
work["Activity_n"] = work["Activity"].astype(str).map(norm_space)
work["Activity_sim"] = work["Activity"].astype(str).map(base_activity_for_similarity)
# Resource can be empty and is NOT an error; still normalize for collateral checks
work["Resource_n"] = work["Resource"].astype(str).map(lambda x: "" if str(x).lower() in ["nan", "none"] else norm_space(x))

n_rows = len(work)

# -----------------------------
# Initialize outputs
# -----------------------------
out = pd.DataFrame({
    "row_id": work["row_id"].values,
    "error_flag": np.zeros(n_rows, dtype=bool),
    "error_types": [""] * n_rows,
    "error_confidence": np.zeros(n_rows, dtype=float),
    "error_tags": [""] * n_rows,
    "error_evidence": [""] * n_rows,
    "error_description": [""] * n_rows,
})

# We'll accumulate per-row detections then finalize
det_types = defaultdict(list)
det_tags = defaultdict(list)
det_evidence = defaultdict(list)
det_desc = defaultdict(list)
det_conf = defaultdict(list)

def add_det(row_id: int, etype: str, conf: float, tag: str, evidence: dict, desc: str):
    det_types[row_id].append(etype)
    det_tags[row_id].append(tag)
    det_evidence[row_id].append(evidence)
    det_desc[row_id].append(desc)
    det_conf[row_id].append(float(conf))

# -----------------------------
# 1) POLLUTED detection (strong regex)
# -----------------------------
# Pattern: baseLabel_<5-12 alnum>_<YYYYMMDD HHMMSS...> or similar
polluted_re = re.compile(r"^(?P<base>.+?)_(?P<rand>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}(?:\s?\d{6,15}))$")

for idx, (rid, act) in enumerate(zip(work["row_id"].values, work["Activity_n"].values)):
    m = polluted_re.match(act)
    if m:
        base = norm_space(m.group("base"))
        # Confidence: very high because pattern is specific
        conf = 0.97
        add_det(
            rid, "polluted", conf,
            "POLLUTED_REGEX_SUFFIX",
            {"activity": act, "parsed_base": base, "suffix": m.group("rand"), "suffix_dt": m.group("dt")},
            f"Activity has machine-generated suffix pattern; base label likely '{base}'."
        )

# -----------------------------
# 2) COLLATERAL detection
#   a) exact duplicates: same Case+Activity+Timestamp+Resource
#   b) near duplicates: same Case+Activity+Resource within <=2 seconds (and not just same timestamp group)
# -----------------------------
# a) exact duplicates
dup_key = ["Case_n", "Activity_n", "Timestamp", "Resource_n"]
dup_counts = work.groupby(dup_key, dropna=False).size().reset_index(name="cnt")
dup_groups = dup_counts[dup_counts["cnt"] >= 2]

if len(dup_groups) > 0:
    # Map group keys to row_ids
    merged = work.merge(dup_groups[dup_key], on=dup_key, how="inner")
    for rid, row in merged[["row_id", "Case_n", "Activity_n", "Timestamp_raw", "Resource_n"]].itertuples(index=False):
        add_det(
            rid, "collateral", 0.95,
            "COLLATERAL_EXACT_DUP",
            {"case": row[0] if isinstance(row, tuple) else None},
            "Exact duplicate event (same case, activity, timestamp, resource) appears multiple times."
        )

# b) near duplicates within 2 seconds
# Sort within case
work_sorted = work.sort_values(["Case_n", "Timestamp", "row_id"], kind="mergesort").copy()
# Compute previous event within same case
work_sorted["prev_case"] = work_sorted["Case_n"].shift(1)
work_sorted["prev_act"] = work_sorted["Activity_n"].shift(1)
work_sorted["prev_res"] = work_sorted["Resource_n"].shift(1)
work_sorted["prev_ts"] = work_sorted["Timestamp"].shift(1)
work_sorted["dt_prev_s"] = (work_sorted["Timestamp"] - work_sorted["prev_ts"]).dt.total_seconds()

near_mask = (
    (work_sorted["Case_n"] == work_sorted["prev_case"]) &
    (work_sorted["Activity_n"] == work_sorted["prev_act"]) &
    (work_sorted["Resource_n"] == work_sorted["prev_res"]) &
    (work_sorted["Timestamp"].notna()) &
    (work_sorted["prev_ts"].notna()) &
    (work_sorted["dt_prev_s"] > 0) &
    (work_sorted["dt_prev_s"] <= 2.0)
)

near = work_sorted.loc[near_mask, ["row_id", "Case_n", "Activity_n", "Resource_n", "Timestamp_raw", "dt_prev_s"]]
for rid, case, act, res, tsraw, dts in near.itertuples(index=False):
    # Avoid double-flagging exact duplicates already caught; still okay if both apply
    add_det(
        rid, "collateral", 0.80,
        "COLLATERAL_NEAR_DUP_2S",
        {"case": case, "activity": act, "resource": res, "timestamp": tsraw, "delta_seconds": float(dts)},
        f"Near-duplicate event: same case/activity/resource repeated within {dts:.3f}s."
    )

# -----------------------------
# 3) FORMBASED detection (conservative)
# Rule: within same case, same timestamp repeated across >=3 events (strong sign of overwrite/form submit).
# Flag all but the first occurrence at that timestamp (per case).
# -----------------------------
ts_grp = work.groupby(["Case_n", "Timestamp"], dropna=False).size().reset_index(name="cnt")
form_groups = ts_grp[(ts_grp["Timestamp"].notna()) & (ts_grp["cnt"] >= 3)]

if len(form_groups) > 0:
    wfg = work.merge(form_groups[["Case_n", "Timestamp", "cnt"]], on=["Case_n", "Timestamp"], how="inner")
    # For each (case,timestamp) group, sort by row_id and keep first as "possibly real", others as formbased
    wfg = wfg.sort_values(["Case_n", "Timestamp", "row_id"], kind="mergesort")
    wfg["rank_in_group"] = wfg.groupby(["Case_n", "Timestamp"]).cumcount()

    fb = wfg[wfg["rank_in_group"] >= 1]
    for rid, case, ts, cnt in fb[["row_id", "Case_n", "Timestamp_raw", "cnt"]].itertuples(index=False):
        add_det(
            rid, "formbased", 0.88,
            "FORMBASED_SAME_TS_GE3",
            {"case": case, "timestamp": ts, "same_ts_count_in_case": int(cnt)},
            f"Same timestamp reused {cnt} times within the same case; likely form-based overwrite."
        )

# -----------------------------
# 4) DISTORTED detection (very conservative)
# Build canonical labels as high-frequency normalized activities (after removing polluted suffix).
# Then flag an activity as distorted only if:
#   - it is NOT exactly a canonical label
#   - it is very similar to a canonical label (ratio >= 0.92)
#   - and edit evidence suggests typo-like change (length difference small)
# Also avoid flagging if activity is very rare canonical itself (we only trust frequent canonicals).
# -----------------------------
# Canonical candidates: base-normalized activity_sim
freq = Counter(work["Activity_sim"].values)
# Trust only frequent labels to avoid learning noise
MIN_CANON_FREQ = max(20, int(0.002 * n_rows))  # at least 20 or 0.2% of log
canon = [a for a, c in freq.items() if c >= MIN_CANON_FREQ and a != ""]
canon_set = set(canon)

# Pre-index canonicals by first character to reduce comparisons
canon_by_initial = defaultdict(list)
for a in canon:
    key = a[:1]
    canon_by_initial[key].append(a)

def best_canon_match(s: str):
    if not s:
        return None, 0.0
    cands = canon_by_initial.get(s[:1], canon)  # fallback
    best = None
    best_r = 0.0
    for c in cands:
        r = seq_ratio(s, c)
        if r > best_r:
            best_r = r
            best = c
    return best, best_r

for rid, act_raw, act_sim in zip(work["row_id"].values, work["Activity_n"].values, work["Activity_sim"].values):
    # Skip if already polluted; polluted can also be distorted but we only add distorted if base looks typoed
    # We'll still allow multi-error, but keep conservative.
    if act_sim in canon_set:
        continue

    # If activity is extremely short or empty, skip (can't infer distortion reliably)
    if len(act_sim) < 4:
        continue

    best, r = best_canon_match(act_sim)
    if best is None:
        continue

    # Conservative thresholds
    if r >= 0.92:
        # length difference small => typo-like
        if abs(len(act_sim) - len(best)) <= 3:
            # Additional guard: require visible "typo signal" (space inserted, swapped char, or one-off)
            # We'll approximate by requiring ratio not perfect and raw differs from best in normalized form.
            if act_sim != best:
                conf = min(0.90, 0.60 + (r - 0.92) * 3.5)  # 0.60..0.90
                add_det(
                    rid, "distorted", conf,
                    "DISTORTED_SIM_TO_FREQ_CANON",
                    {"activity": act_raw, "activity_norm": act_sim, "closest_canonical": best, "similarity": round(r, 4),
                     "canonical_freq": int(freq.get(best, 0))},
                    f"Activity text looks like a typo of frequent label '{best}' (similarity={r:.3f})."
                )

# -----------------------------
# Finalize output rows
# -----------------------------
for i in range(n_rows):
    rid = int(out.at[i, "row_id"])
    if rid not in det_types:
        continue

    # Merge types unique, stable order
    types = []
    for t in det_types[rid]:
        if t not in types:
            types.append(t)

    # Confidence: combine multiple signals (noisy-or)
    confs = det_conf[rid]
    p_no = 1.0
    for c in confs:
        p_no *= (1.0 - max(0.0, min(1.0, c)))
    combined = 1.0 - p_no
    combined = float(max(0.0, min(1.0, combined)))

    out.at[i, "error_flag"] = True
    out.at[i, "error_types"] = "|".join(types)
    out.at[i, "error_confidence"] = combined
    out.at[i, "error_tags"] = "|".join(dict.fromkeys(det_tags[rid]))
    out.at[i, "error_evidence"] = safe_json(det_evidence[rid])
    out.at[i, "error_description"] = " ; ".join(dict.fromkeys(det_desc[rid]))

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

print(f"Wrote: {OUTPUT_PATH}")
print("Summary:")
print(out["error_flag"].value_counts(dropna=False).to_string())
print("Error type counts:")
type_counts = Counter()
for s in out.loc[out["error_flag"], "error_types"].values:
    for t in str(s).split("|"):
        if t:
            type_counts[t] += 1
print(dict(type_counts))