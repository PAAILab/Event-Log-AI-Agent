#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Mining Event Log Error Detection (credit)
Input : /home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_polluted_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_polluted_noLabel.detected.csv

Detects errors ONLY using: Case, Activity, Timestamp, Resource
(Resource empty is NOT an error by domain instruction.)
"""

import os
import re
import json
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_polluted_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_polluted_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]


# -----------------------------
# Utility helpers
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

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def jaccard(a, b):
    a = set(a); b = set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def tokenize_words(s: str):
    s = norm_case(s)
    # keep alphanumerics, split on others
    toks = re.findall(r"[a-z0-9]+", s)
    return toks

def char_ngrams(s: str, n=3):
    s = norm_case(s)
    s = re.sub(r"\s+", " ", s)
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def similarity(a: str, b: str) -> float:
    # robust similarity for typos: combine word-jaccard and char-trigram jaccard
    wa, wb = tokenize_words(a), tokenize_words(b)
    ca, cb = char_ngrams(a, 3), char_ngrams(b, 3)
    return 0.55 * jaccard(wa, wb) + 0.45 * jaccard(ca, cb)

def is_missing_required(s: str) -> bool:
    return norm_space(s) == ""

def parse_ts(series: pd.Series) -> pd.Series:
    # aggressive parsing; coerce invalid to NaT
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True, utc=False)

def to_jsonable(obj):
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except Exception:
        return safe_str(obj)


# -----------------------------
# Polluted detection
# -----------------------------
# Typical pattern: "<base>_<5-12 alnum>_<YYYYMMDD HHMMSSffffff...>"
POLLUTED_RE = re.compile(
    r"^(?P<base>.+?)_(?P<rand>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6}\d{3,6})$"
)

# Also accept: "<base>_<5-12 alnum>" (weaker)
POLLUTED_WEAK_RE = re.compile(r"^(?P<base>.+?)_(?P<rand>[A-Za-z0-9]{5,12})$")


def detect_polluted(activity: str):
    a = norm_space(activity)
    m = POLLUTED_RE.match(a)
    if m:
        base = norm_space(m.group("base"))
        return True, base, "polluted:strong_pattern"
    m2 = POLLUTED_WEAK_RE.match(a)
    if m2:
        base = norm_space(m2.group("base"))
        # weak: could be legitimate underscore usage; keep but lower confidence
        return True, base, "polluted:weak_suffix"
    return False, None, None


# -----------------------------
# Build canonical activity set
# -----------------------------
def build_canonical_activities(df):
    """
    Canonical activities are inferred from the dataset itself:
    - If polluted, use extracted base as canonical candidate.
    - Otherwise use the raw activity.
    Then choose the most frequent normalized form as canonical representative.
    """
    raw = df["Activity"].astype(str).map(norm_space)

    base_candidates = []
    for a in raw:
        is_pol, base, _ = detect_polluted(a)
        base_candidates.append(base if is_pol and base else a)

    # normalize for grouping
    normed = [norm_case(x) for x in base_candidates]
    counts = Counter(normed)

    # map normalized -> best representative (most common original casing/spaces)
    rep = {}
    for n in counts:
        # choose representative among candidates with same normalized form
        candidates = [base_candidates[i] for i in range(len(base_candidates)) if norm_case(base_candidates[i]) == n]
        rep[n] = Counter(candidates).most_common(1)[0][0]

    return counts, rep


# -----------------------------
# Synonym detection (data-driven)
# -----------------------------
def build_synonym_map(counts, rep, min_freq=5, sim_thresh=0.86):
    """
    Aggressive, evidence-based synonym inference:
    - Consider frequent labels as potential canonicals.
    - If two labels have high similarity but different tokens (not just typos),
      treat the less frequent as synonym of the more frequent.
    This is heuristic; confidence is moderate unless very strong evidence.
    """
    # frequent normalized labels
    frequent = [n for n, c in counts.items() if c >= min_freq]
    frequent_sorted = sorted(frequent, key=lambda n: counts[n], reverse=True)

    synonym_to = {}  # norm_label -> norm_canonical
    for i, n in enumerate(frequent_sorted):
        if n in synonym_to:
            continue
        for m in frequent_sorted[i+1:]:
            if m in synonym_to:
                continue
            a = rep[n]; b = rep[m]
            sim = similarity(a, b)
            if sim >= sim_thresh:
                # decide if it's more like typo (distorted) vs synonym:
                # if word sets differ materially, call synonym; else distorted.
                wa, wb = set(tokenize_words(a)), set(tokenize_words(b))
                word_j = jaccard(wa, wb)
                # if word overlap is low but char similarity high -> synonym-ish
                if word_j < 0.75:
                    # map less frequent to more frequent
                    if counts[n] >= counts[m]:
                        synonym_to[m] = n
                    else:
                        synonym_to[n] = m
    return synonym_to


# -----------------------------
# Distorted detection (typos)
# -----------------------------
def detect_distorted(activity_base: str, counts, rep, min_canon_freq=5):
    """
    Detect typos by matching to a frequent canonical label with high similarity.
    """
    a = norm_space(activity_base)
    na = norm_case(a)
    # if it's already a frequent canonical, not distorted
    if counts.get(na, 0) >= min_canon_freq:
        return False, None, None, 0.0

    # compare to frequent canonicals
    frequent = [n for n, c in counts.items() if c >= min_canon_freq]
    best = (None, 0.0)
    for n in frequent:
        s = similarity(a, rep[n])
        if s > best[1]:
            best = (n, s)

    best_n, best_s = best
    if best_n is None:
        return False, None, None, 0.0

    # distorted if very similar but not identical
    if best_s >= 0.88 and na != best_n:
        return True, rep[best_n], "distorted:near_canonical", best_s
    return False, None, None, best_s


# -----------------------------
# Collateral detection (duplicates / near-duplicates)
# -----------------------------
def detect_collateral(df, ts_col="Timestamp_parsed", near_seconds=3):
    """
    Flags:
    - exact duplicates: same Case, Activity, Timestamp, Resource
    - near duplicates: same Case, Activity, Resource within near_seconds
    """
    flags = np.zeros(len(df), dtype=bool)
    tags = [[] for _ in range(len(df))]
    evidence = [dict() for _ in range(len(df))]

    # exact duplicates
    key_cols = ["Case", "Activity", "Timestamp", "Resource"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    for idx in df.index[dup_mask]:
        flags[idx] = True
        tags[idx].append("collateral:exact_duplicate")
        evidence[idx]["exact_duplicate_key"] = {c: to_jsonable(df.at[idx, c]) for c in key_cols}

    # near duplicates (within case/activity/resource)
    # sort by case, activity, resource, timestamp
    tmp = df.reset_index().rename(columns={"index": "row_id"})
    tmp = tmp.sort_values(["Case", "Activity", "Resource", ts_col], kind="mergesort")
    # compute time diff within group
    tmp["prev_ts"] = tmp.groupby(["Case", "Activity", "Resource"])[ts_col].shift(1)
    tmp["dt_prev"] = (tmp[ts_col] - tmp["prev_ts"]).dt.total_seconds()

    near = tmp["dt_prev"].notna() & (tmp["dt_prev"] >= 0) & (tmp["dt_prev"] <= near_seconds)
    for _, r in tmp.loc[near].iterrows():
        rid = int(r["row_id"])
        flags[rid] = True
        tags[rid].append(f"collateral:near_duplicate<= {near_seconds}s")
        evidence[rid]["near_duplicate"] = {
            "dt_prev_seconds": float(r["dt_prev"]),
            "group": {
                "Case": to_jsonable(r["Case"]),
                "Activity": to_jsonable(r["Activity"]),
                "Resource": to_jsonable(r["Resource"]),
            },
        }

    return flags, tags, evidence


# -----------------------------
# Form-based detection (same timestamp bursts)
# -----------------------------
def detect_formbased(df, ts_col="Timestamp_parsed", min_events_same_ts=3):
    """
    Within a case, if many events share the exact same timestamp, likely form-based overwrite.
    We flag all events in that (case, timestamp) bucket when bucket size >= min_events_same_ts.
    """
    flags = np.zeros(len(df), dtype=bool)
    tags = [[] for _ in range(len(df))]
    evidence = [dict() for _ in range(len(df))]

    grp = df.groupby(["Case", ts_col], dropna=False).size().rename("cnt").reset_index()
    suspicious = grp[(grp[ts_col].notna()) & (grp["cnt"] >= min_events_same_ts)]
    if suspicious.empty:
        return flags, tags, evidence

    sus_set = set((row["Case"], row[ts_col]) for _, row in suspicious.iterrows())
    for idx, row in df.iterrows():
        key = (row["Case"], row[ts_col])
        if key in sus_set:
            flags[idx] = True
            tags[idx].append(f"formbased:same_ts_bucket>= {min_events_same_ts}")
            evidence[idx]["same_ts_bucket"] = {
                "Case": to_jsonable(row["Case"]),
                "Timestamp": safe_str(row["Timestamp"]),
                "bucket_size": int(suspicious.loc[(suspicious["Case"] == row["Case"]) & (suspicious[ts_col] == row[ts_col]), "cnt"].iloc[0]),
            }
    return flags, tags, evidence


# -----------------------------
# Homonymous detection (same label, divergent contexts)
# -----------------------------
def detect_homonymous(df, ts_col="Timestamp_parsed", min_label_freq=30):
    """
    Data-driven homonym heuristic:
    If an activity label appears frequently and is executed by clearly different resource groups
    AND appears in very different typical positions within cases, it may be overloaded.

    We flag only when evidence is strong to avoid false positives.
    """
    flags = np.zeros(len(df), dtype=bool)
    tags = [[] for _ in range(len(df))]
    evidence = [dict() for _ in range(len(df))]

    # compute event position percentile within each case
    tmp = df.copy()
    tmp["_order"] = tmp.groupby("Case")[ts_col].rank(method="first")
    tmp["_len"] = tmp.groupby("Case")[ts_col].transform("count")
    tmp["_pos"] = (tmp["_order"] - 1) / (tmp["_len"].clip(lower=1) - 1).replace(0, np.nan)

    # per activity stats
    act_counts = tmp["Activity"].map(norm_space).value_counts()
    frequent_acts = act_counts[act_counts >= min_label_freq].index.tolist()

    for act in frequent_acts:
        sub = tmp[tmp["Activity"].map(norm_space) == act]
        # resource diversity (ignore empty resource; not an error)
        res = sub["Resource"].map(norm_space)
        res_nonempty = res[res != ""]
        top_res = res_nonempty.value_counts().head(5)
        res_entropy = 0.0
        if len(res_nonempty) > 0:
            p = (res_nonempty.value_counts(normalize=True)).values
            res_entropy = float(-(p * np.log2(p + 1e-12)).sum())

        # position spread
        pos = sub["_pos"].dropna()
        pos_std = float(pos.std()) if len(pos) > 5 else 0.0

        # strong evidence thresholds
        # - high entropy indicates multiple distinct resource groups
        # - high position std indicates used in different phases
        if (res_entropy >= 2.0 and pos_std >= 0.35 and len(sub) >= min_label_freq):
            # flag all rows of this activity as potentially homonymous
            for idx in sub.index:
                flags[idx] = True
                tags[idx].append("homonymous:resource_entropy+position_spread")
                evidence[idx]["homonymous_stats"] = {
                    "activity": act,
                    "activity_freq": int(len(sub)),
                    "resource_entropy": res_entropy,
                    "position_std": pos_std,
                    "top_resources": top_res.to_dict(),
                }

    return flags, tags, evidence


# -----------------------------
# Main detection pipeline
# -----------------------------
def main():
    df = pd.read_csv(INPUT_PATH)

    # Validate required columns only
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")

    # Keep only required columns for detection (but we still output row_id aligned to original rows)
    df = df[REQUIRED_COLS].copy()
    df = df.reset_index(drop=True)
    df.index.name = "row_id"

    # Normalize
    df["Case_n"] = df["Case"].map(norm_space)
    df["Activity_n"] = df["Activity"].map(norm_space)
    df["Resource_n"] = df["Resource"].map(norm_space)  # empty allowed
    df["Timestamp_parsed"] = parse_ts(df["Timestamp"])

    # Prepare outputs
    out = pd.DataFrame({"row_id": df.index.astype(int)})
    out["error_flag"] = False
    out["error_types"] = ""
    out["error_confidence"] = 0.0
    out["error_tags"] = ""
    out["error_evidence"] = ""
    out["error_description"] = ""

    # Collect per-row detections
    row_types = defaultdict(set)
    row_tags = defaultdict(list)
    row_evidence = defaultdict(list)
    row_desc = defaultdict(list)
    row_conf_parts = defaultdict(list)

    # 0) Basic format/range checks (aggressive)
    # Timestamp invalid
    invalid_ts = df["Timestamp_parsed"].isna()
    for rid in df.index[invalid_ts]:
        row_types[rid].add("timestamp_invalid")
        row_tags[rid].append("format:timestamp_parse_failed")
        row_evidence[rid].append({"Timestamp": safe_str(df.at[rid, "Timestamp"])})
        row_desc[rid].append("Timestamp cannot be parsed (NaT).")
        row_conf_parts[rid].append(0.95)

    # Case missing
    missing_case = df["Case_n"].map(is_missing_required)
    for rid in df.index[missing_case]:
        row_types[rid].add("case_missing")
        row_tags[rid].append("null:case_empty")
        row_evidence[rid].append({"Case": safe_str(df.at[rid, "Case"])})
        row_desc[rid].append("Case identifier is empty/null.")
        row_conf_parts[rid].append(0.98)

    # Activity missing
    missing_act = df["Activity_n"].map(is_missing_required)
    for rid in df.index[missing_act]:
        row_types[rid].add("activity_missing")
        row_tags[rid].append("null:activity_empty")
        row_evidence[rid].append({"Activity": safe_str(df.at[rid, "Activity"])})
        row_desc[rid].append("Activity label is empty/null.")
        row_conf_parts[rid].append(0.98)

    # 1) Polluted detection (+ extract base)
    polluted_base = {}
    for rid, a in df["Activity_n"].items():
        is_pol, base, tag = detect_polluted(a)
        if is_pol and base:
            row_types[rid].add("polluted")
            row_tags[rid].append(tag)
            row_evidence[rid].append({"Activity": a, "extracted_base": base})
            row_desc[rid].append(f"Activity appears polluted with machine suffix; base='{base}'.")
            # strong pattern higher confidence
            row_conf_parts[rid].append(0.92 if tag == "polluted:strong_pattern" else 0.70)
            polluted_base[rid] = base

    # 2) Build canonicals from de-polluted candidates
    counts, rep = build_canonical_activities(df)

    # 3) Distorted detection (use base if polluted)
    for rid, a in df["Activity_n"].items():
        base = polluted_base.get(rid, a)
        is_dis, canon, tag, sim = detect_distorted(base, counts, rep, min_canon_freq=5)
        if is_dis and canon:
            row_types[rid].add("distorted")
            row_tags[rid].append(tag)
            row_evidence[rid].append({"Activity_base": base, "closest_canonical": canon, "similarity": round(sim, 4)})
            row_desc[rid].append(f"Activity text likely typo/distortion of '{canon}' (sim={sim:.3f}).")
            # confidence scales with similarity
            row_conf_parts[rid].append(clamp01(0.55 + 0.45 * sim))

    # 4) Synonymous detection (data-driven)
    synonym_to = build_synonym_map(counts, rep, min_freq=5, sim_thresh=0.86)
    for rid, a in df["Activity_n"].items():
        base = polluted_base.get(rid, a)
        nb = norm_case(base)
        if nb in synonym_to:
            canon_n = synonym_to[nb]
            canon = rep.get(canon_n, rep.get(nb, base))
            row_types[rid].add("synonymous")
            row_tags[rid].append("synonymous:inferred_from_distributional_similarity")
            row_evidence[rid].append({"Activity_base": base, "mapped_to": canon, "freq_base": int(counts.get(nb, 0)), "freq_canon": int(counts.get(canon_n, 0))})
            row_desc[rid].append(f"Activity label appears synonymous with '{canon}' (inferred).")
            # moderate confidence: stronger if canonical much more frequent
            fb = counts.get(nb, 0); fc = counts.get(canon_n, 0)
            ratio = (fc + 1) / (fb + 1)
            row_conf_parts[rid].append(clamp01(0.55 + 0.15 * math.tanh(math.log(ratio))))

    # 5) Collateral detection
    col_flags, col_tags, col_evid = detect_collateral(df, ts_col="Timestamp_parsed", near_seconds=3)
    for rid in df.index[col_flags]:
        row_types[rid].add("collateral")
        row_tags[rid].extend(col_tags[rid])
        row_evidence[rid].append(col_evid[rid])
        row_desc[rid].append("Duplicate/near-duplicate event detected (collateral logging artifact).")
        # exact duplicates very high; near duplicates high
        conf = 0.97 if any("exact_duplicate" in t for t in col_tags[rid]) else 0.85
        row_conf_parts[rid].append(conf)

    # 6) Form-based detection
    fb_flags, fb_tags, fb_evid = detect_formbased(df, ts_col="Timestamp_parsed", min_events_same_ts=3)
    for rid in df.index[fb_flags]:
        row_types[rid].add("formbased")
        row_tags[rid].extend(fb_tags[rid])
        row_evidence[rid].append(fb_evid[rid])
        row_desc[rid].append("Multiple events in same case share identical timestamp (possible form-based overwrite).")
        # confidence increases with bucket size
        bsz = fb_evid[rid].get("same_ts_bucket", {}).get("bucket_size", 3)
        row_conf_parts[rid].append(clamp01(0.70 + 0.06 * min(6, max(0, bsz - 3))))

    # 7) Homonymous detection (strong evidence only)
    ho_flags, ho_tags, ho_evid = detect_homonymous(df, ts_col="Timestamp_parsed", min_label_freq=30)
    for rid in df.index[ho_flags]:
        row_types[rid].add("homonymous")
        row_tags[rid].extend(ho_tags[rid])
        row_evidence[rid].append(ho_evid[rid])
        row_desc[rid].append("Activity label may be overloaded (homonymous) based on divergent resource/position patterns.")
        # keep moderate-high only when strong stats triggered
        stats = ho_evid[rid].get("homonymous_stats", {})
        ent = float(stats.get("resource_entropy", 2.0))
        ps = float(stats.get("position_std", 0.35))
        row_conf_parts[rid].append(clamp01(0.60 + 0.10 * min(3.0, ent) / 3.0 + 0.20 * min(0.6, ps) / 0.6))

    # -----------------------------
    # Finalize output rows
    # -----------------------------
    for rid in df.index:
        types = sorted(row_types.get(rid, []))
        if types:
            out.at[rid, "error_flag"] = True
            out.at[rid, "error_types"] = "|".join(types)

            # confidence: combine as 1 - Π(1-ci) (increasing with multiple independent evidence)
            confs = row_conf_parts.get(rid, [])
            if confs:
                prod = 1.0
                for c in confs:
                    prod *= (1.0 - clamp01(c))
                out.at[rid, "error_confidence"] = clamp01(1.0 - prod)
            else:
                out.at[rid, "error_confidence"] = 0.5

            out.at[rid, "error_tags"] = "|".join(row_tags.get(rid, []))[:2000]
            out.at[rid, "error_evidence"] = json.dumps(row_evidence.get(rid, []), ensure_ascii=False)[:8000]
            out.at[rid, "error_description"] = " ".join(row_desc.get(rid, []))[:2000]
        else:
            out.at[rid, "error_flag"] = False
            out.at[rid, "error_types"] = ""
            out.at[rid, "error_confidence"] = 0.0
            out.at[rid, "error_tags"] = ""
            out.at[rid, "error_evidence"] = ""
            out.at[rid, "error_description"] = ""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved detected errors to: {OUTPUT_PATH}")
    print(out["error_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()