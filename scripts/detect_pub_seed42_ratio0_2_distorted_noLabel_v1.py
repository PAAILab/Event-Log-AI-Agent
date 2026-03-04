#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_distorted_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_distorted_noLabel.detected.csv

Detects (based ONLY on Case, Activity, Timestamp, Resource):
- formbased
- polluted
- distorted
- synonymous
- collateral
- homonymous (heuristic, evidence-based)
Notes:
- Empty Resource is NOT an error (do not flag for missing resource).
"""

import os
import re
import math
import json
import difflib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_distorted_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_distorted_noLabel.detected.csv"

CASE_COL = "Case"
ACT_COL = "Activity"
TS_COL = "Timestamp"
RES_COL = "Resource"

# --- Tunable thresholds (aggressive by design) ---
COLLATERAL_WINDOW_SECONDS = 3.0          # near-duplicate window
FORMBASED_MIN_EVENTS_SAME_TS = 3         # within a case, same timestamp repeated >= this
FORMBASED_MIN_DISTINCT_ACTIVITIES = 2    # and at least this many distinct activities at that timestamp
DISTORTED_MIN_SIMILARITY = 0.86          # similarity to a canonical label to call it a typo
DISTORTED_MAX_SIMILARITY = 0.985         # if too close to 1.0, likely exact match (avoid false positives)
SYNONYM_MIN_SIMILARITY = 0.70            # similarity to canonical for synonym candidates (looser)
HOMONYM_CONTEXT_TOPK = 2                 # compare top preceding/following activities
HOMONYM_MIN_SUPPORT = 8                  # label must appear at least this many times overall
HOMONYM_MIN_CONTEXT_DIVERGENCE = 0.65    # higher => more divergent contexts
HOMONYM_MIN_RESOURCE_DIVERGENCE = 0.60   # divergence in resource distribution


# ----------------- Helpers -----------------
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def parse_timestamp(series: pd.Series) -> pd.Series:
    # robust parsing; keep NaT for invalid
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def norm_activity(a: str) -> str:
    a = safe_str(a).strip()
    a = re.sub(r"\s+", " ", a)
    return a

def norm_activity_loose(a: str) -> str:
    """Lowercase + collapse spaces + remove punctuation-ish for fuzzy matching."""
    a = norm_activity(a).lower()
    a = re.sub(r"[_\-]+", " ", a)
    a = re.sub(r"[^\w\s\(\)]", " ", a)  # keep parentheses words
    a = re.sub(r"\s+", " ", a).strip()
    return a

POLLUTED_RE = re.compile(
    r"""^(?P<base>.+?)_([A-Za-z0-9]{5,12})_(\d{8}\s\d{6,9})$"""
)

def detect_polluted(activity: str):
    m = POLLUTED_RE.match(norm_activity(activity))
    if not m:
        return None
    base = m.group("base").strip()
    # base must be non-trivial
    if len(base) < 3:
        return None
    return base

def seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Returns JSD in [0,1] (base-2)."""
    p = p.astype(float)
    q = q.astype(float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(x, y):
        x = np.clip(x, 1e-12, 1.0)
        y = np.clip(y, 1e-12, 1.0)
        return np.sum(x * np.log2(x / y))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))

def confidence_from_evidence(base: float, boosts):
    """Combine base with additive boosts, clamp [0,1]."""
    c = base + sum(boosts)
    return float(max(0.0, min(1.0, c)))

def add_error(errs, etype, conf, tag, evidence, desc):
    errs["types"].add(etype)
    errs["tags"].add(tag)
    errs["evidence"].append(evidence)
    errs["descriptions"].append(desc)
    errs["conf"] = max(errs["conf"], conf)

def finalize_row(errs):
    if not errs["types"]:
        return False, "", 0.0, "", "", ""
    types = "|".join(sorted(errs["types"]))
    tags = "|".join(sorted(errs["tags"]))
    # keep evidence compact but specific
    evidence = json.dumps(errs["evidence"], ensure_ascii=False)
    desc = " ; ".join(errs["descriptions"])
    return True, types, float(errs["conf"]), tags, evidence, desc


# ----------------- Main detection -----------------
def main():
    df = pd.read_csv(INPUT_PATH)

    # Validate required columns (do not use others)
    missing = [c for c in [CASE_COL, ACT_COL, TS_COL, RES_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Keep only required columns for detection logic
    work = df[[CASE_COL, ACT_COL, TS_COL, RES_COL]].copy()
    work["row_id"] = np.arange(len(work), dtype=int)

    # Normalize
    work["_act_raw"] = work[ACT_COL].apply(norm_activity)
    work["_act_loose"] = work["_act_raw"].apply(norm_activity_loose)
    work["_case"] = work[CASE_COL].astype(str)
    work["_res"] = work[RES_COL].apply(lambda x: safe_str(x).strip())  # empty allowed
    work["_ts"] = parse_timestamp(work[TS_COL])

    # Precompute canonical activity set:
    # Use "clean-looking" labels as canonical candidates:
    # - not polluted pattern
    # - not empty
    # - not obviously garbled (very short)
    is_polluted = work["_act_raw"].apply(lambda a: detect_polluted(a) is not None)
    canonical_candidates = work.loc[(~is_polluted) & (work["_act_loose"].str.len() >= 3), "_act_raw"]

    # Canonical labels: most frequent raw labels (clean-looking)
    # Keep all unique; frequency used for tie-breaking.
    canon_freq = canonical_candidates.value_counts()
    canon_labels = list(canon_freq.index)

    # Build synonym map heuristically:
    # If a label is not in canon_labels? Actually everything is in canon_labels by definition.
    # We'll detect synonyms by:
    # - label differs by words but is close-ish to a canonical label
    # - AND not a typo (distorted) (i.e., similarity not too high)
    # - AND token overlap is moderate (avoid random matches)
    canon_loose = {c: norm_activity_loose(c) for c in canon_labels}

    # Index canon by first token to speed up
    canon_by_token = defaultdict(list)
    for c in canon_labels:
        toks = canon_loose[c].split()
        if toks:
            canon_by_token[toks[0]].append(c)
        else:
            canon_by_token[""].append(c)

    # Prepare per-row error accumulator
    errors = {rid: {"types": set(), "tags": set(), "evidence": [], "descriptions": [], "conf": 0.0}
              for rid in work["row_id"].tolist()}

    # --------- Basic format/range checks (aggressive but evidence-based) ----------
    # Timestamp parse failures are real errors in event logs (cannot order events).
    bad_ts = work["_ts"].isna()
    for rid, tsval in work.loc[bad_ts, ["row_id", TS_COL]].itertuples(index=False):
        add_error(
            errors[rid], "timestamp_invalid",
            conf=0.98,
            tag="ts_parse_fail",
            evidence={"timestamp": safe_str(tsval)},
            desc=f"Timestamp is not parseable: '{safe_str(tsval)}'"
        )

    # Empty/Null Activity is an error (cannot interpret event)
    bad_act = work["_act_loose"].str.len() == 0
    for rid, aval in work.loc[bad_act, ["row_id", ACT_COL]].itertuples(index=False):
        add_error(
            errors[rid], "activity_missing",
            conf=0.97,
            tag="act_empty",
            evidence={"activity": safe_str(aval)},
            desc="Activity is empty/null"
        )

    # Empty/Null Case is an error (cannot group trace)
    bad_case = work["_case"].str.strip().eq("") | work["_case"].str.lower().eq("nan")
    for rid, cval in work.loc[bad_case, ["row_id", CASE_COL]].itertuples(index=False):
        add_error(
            errors[rid], "case_missing",
            conf=0.97,
            tag="case_empty",
            evidence={"case": safe_str(cval)},
            desc="Case identifier is empty/null"
        )

    # --------- Polluted detection ----------
    for rid, act in work.loc[~bad_act, ["row_id", "_act_raw"]].itertuples(index=False):
        base = detect_polluted(act)
        if base is None:
            continue
        # Confidence higher if base exists as a canonical label
        base_norm = norm_activity(base)
        base_in_log = base_norm in canon_freq.index
        conf = 0.92 if base_in_log else 0.80
        add_error(
            errors[rid], "polluted",
            conf=conf,
            tag="act_polluted_suffix",
            evidence={"activity": act, "extracted_base": base_norm, "base_seen_in_log": base_in_log},
            desc=f"Activity has machine-generated suffix pattern; extracted base='{base_norm}'"
        )

    # --------- Distorted & Synonymous detection (fuzzy against canon) ----------
    # For each activity, find best matching canonical label (excluding itself).
    # If polluted, compare base part too (can be polluted+distorted/synonymous).
    def best_canonical_match(act_raw: str):
        act_l = norm_activity_loose(act_raw)
        if not act_l:
            return None, 0.0
        # candidate set: same first token or all if none
        first = act_l.split()[0] if act_l.split() else ""
        cands = canon_by_token.get(first, canon_labels[:200])  # fallback limited
        best = None
        best_r = -1.0
        for c in cands:
            c_l = canon_loose[c]
            if c_l == act_l:
                continue
            r = seq_ratio(act_l, c_l)
            if r > best_r:
                best_r = r
                best = c
        # if weak, broaden search a bit
        if best_r < 0.80:
            for c in canon_labels[:300]:
                c_l = canon_loose[c]
                if c_l == act_l:
                    continue
                r = seq_ratio(act_l, c_l)
                if r > best_r:
                    best_r = r
                    best = c
        return best, float(best_r)

    for rid, act_raw in work.loc[~bad_act, ["row_id", "_act_raw"]].itertuples(index=False):
        # If polluted, also evaluate base
        base = detect_polluted(act_raw)
        act_to_check = base if base is not None else act_raw

        best, sim = best_canonical_match(act_to_check)
        if best is None:
            continue

        # Token overlap
        a_tokens = set(norm_activity_loose(act_to_check).split())
        b_tokens = set(norm_activity_loose(best).split())
        overlap = (len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens)))

        # Distorted: very close string similarity, but not exact
        if sim >= DISTORTED_MIN_SIMILARITY and sim <= DISTORTED_MAX_SIMILARITY:
            # stronger if overlap high and length similar
            len_ratio = min(len(act_to_check), len(best)) / max(1, max(len(act_to_check), len(best)))
            conf = confidence_from_evidence(
                0.70,
                boosts=[
                    0.15 if sim >= 0.92 else 0.05,
                    0.10 if overlap >= 0.6 else 0.0,
                    0.05 if len_ratio >= 0.85 else 0.0,
                    0.05 if base is not None else 0.0,  # polluted+distorted common
                ],
            )
            add_error(
                errors[rid], "distorted",
                conf=conf,
                tag="act_fuzzy_typo",
                evidence={"activity_checked": act_to_check, "best_canonical": best, "similarity": sim, "token_overlap": overlap},
                desc=f"Activity text looks like a typo/character distortion of '{best}' (sim={sim:.3f})"
            )
            continue

        # Synonymous: moderately similar but not typo-close; also require low char similarity but some semantic proximity
        # Here we approximate semantics via token overlap and shared head token patterns.
        if sim >= SYNONYM_MIN_SIMILARITY and sim < DISTORTED_MIN_SIMILARITY and overlap >= 0.25:
            # Avoid flagging if it's just different casing/punctuation (would have sim high)
            conf = confidence_from_evidence(
                0.55,
                boosts=[
                    0.15 if overlap >= 0.45 else 0.05,
                    0.10 if sim >= 0.78 else 0.0,
                    0.05 if base is not None else 0.0,
                ],
            )
            add_error(
                errors[rid], "synonymous",
                conf=conf,
                tag="act_synonym_heuristic",
                evidence={"activity_checked": act_to_check, "best_canonical": best, "similarity": sim, "token_overlap": overlap},
                desc=f"Activity label may be a synonym/variant of '{best}' (sim={sim:.3f}, overlap={overlap:.2f})"
            )

    # --------- Collateral detection (duplicates / near-duplicates) ----------
    # Exact duplicates: same case, activity, timestamp, resource
    # Near duplicates: same case, activity, resource within COLLATERAL_WINDOW_SECONDS
    # (resource may be empty; still counts if equal)
    work_sorted = work.sort_values(["_case", "_ts", "row_id"], kind="mergesort")

    # Exact duplicates
    exact_key = work_sorted[["_case", "_act_raw", "_ts", "_res"]].astype(str).agg("||".join, axis=1)
    dup_mask = exact_key.duplicated(keep=False) & (~work_sorted["_ts"].isna()) & (~bad_act.loc[work_sorted.index].values)
    for rid, case, act, ts, res in work_sorted.loc[dup_mask, ["row_id", "_case", "_act_raw", "_ts", "_res"]].itertuples(index=False):
        add_error(
            errors[rid], "collateral",
            conf=0.95,
            tag="dup_exact_case_act_ts_res",
            evidence={"case": case, "activity": act, "timestamp": str(ts), "resource": res},
            desc="Exact duplicate event (same case, activity, timestamp, resource)"
        )

    # Near duplicates within window
    # For each (case, activity, resource) group, check consecutive timestamps
    grp_cols = ["_case", "_act_raw", "_res"]
    for (case, act, res), g in work_sorted.dropna(subset=["_ts"]).groupby(grp_cols, sort=False):
        if len(g) < 2:
            continue
        ts = g["_ts"].values
        rids = g["row_id"].values
        # consecutive diffs
        diffs = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
        near_idx = np.where((diffs >= 0) & (diffs <= COLLATERAL_WINDOW_SECONDS))[0]
        for i in near_idx:
            for rid in (rids[i], rids[i+1]):
                add_error(
                    errors[int(rid)], "collateral",
                    conf=0.82,
                    tag="dup_near_case_act_res",
                    evidence={"case": case, "activity": act, "resource": res, "delta_seconds": float(diffs[i])},
                    desc=f"Near-duplicate events within {COLLATERAL_WINDOW_SECONDS}s for same case/activity/resource (Δ={diffs[i]:.3f}s)"
                )

    # --------- Formbased detection (same timestamp reused within a case) ----------
    # Within each case, if a timestamp repeats many times (>=3) and multiple distinct activities share it,
    # flag all but the first occurrence as formbased (conservative within aggressive rule).
    for case, g in work_sorted.dropna(subset=["_ts"]).groupby("_case", sort=False):
        # count per timestamp
        counts = g["_ts"].value_counts()
        suspect_ts = counts[counts >= FORMBASED_MIN_EVENTS_SAME_TS].index
        if len(suspect_ts) == 0:
            continue
        for ts_val in suspect_ts:
            gg = g[g["_ts"] == ts_val].copy()
            if gg["_act_raw"].nunique() < FORMBASED_MIN_DISTINCT_ACTIVITIES:
                continue
            gg = gg.sort_values("row_id")
            # keep first as "anchor", flag the rest
            anchor = gg.iloc[0]
            for _, row in gg.iloc[1:].iterrows():
                rid = int(row["row_id"])
                conf = confidence_from_evidence(
                    0.78,
                    boosts=[
                        0.10 if len(gg) >= 4 else 0.0,
                        0.05 if gg["_act_raw"].nunique() >= 3 else 0.0,
                    ],
                )
                add_error(
                    errors[rid], "formbased",
                    conf=conf,
                    tag="same_ts_many_events_in_case",
                    evidence={
                        "case": case,
                        "timestamp": str(ts_val),
                        "events_at_timestamp": int(len(gg)),
                        "distinct_activities_at_timestamp": int(gg["_act_raw"].nunique()),
                        "anchor_row_id": int(anchor["row_id"]),
                        "anchor_activity": anchor["_act_raw"],
                    },
                    desc=f"Multiple events in same case share identical timestamp {ts_val}; likely form-based overwrite/logging"
                )

    # --------- Homonymous detection (heuristic via divergent contexts) ----------
    # Build context signatures: (prev_activity, next_activity) within each case ordered by timestamp then row_id.
    # If same label appears in two very different context clusters and resource distributions differ, flag.
    ctx_prev = {}
    ctx_next = {}
    for case, g in work_sorted.dropna(subset=["_ts"]).groupby("_case", sort=False):
        gg = g.sort_values(["_ts", "row_id"], kind="mergesort")
        acts = gg["_act_raw"].tolist()
        rids = gg["row_id"].tolist()
        for i, rid in enumerate(rids):
            prev_a = acts[i-1] if i-1 >= 0 else None
            next_a = acts[i+1] if i+1 < len(acts) else None
            ctx_prev[rid] = prev_a
            ctx_next[rid] = next_a

    work["_prev_act"] = work["row_id"].map(ctx_prev)
    work["_next_act"] = work["row_id"].map(ctx_next)

    # Only consider labels with enough support
    act_counts = work["_act_raw"].value_counts()
    candidate_labels = act_counts[act_counts >= HOMONYM_MIN_SUPPORT].index.tolist()

    for label in candidate_labels:
        sub = work[work["_act_raw"] == label].copy()
        # context distribution
        prev_counts = sub["_prev_act"].fillna("<START>").value_counts()
        next_counts = sub["_next_act"].fillna("<END>").value_counts()
        res_counts = sub["_res"].fillna("").value_counts()

        # Split into two "modes" by most common prev and next (simple 2-cluster proxy)
        top_prev = prev_counts.index[:HOMONYM_CONTEXT_TOPK].tolist()
        top_next = next_counts.index[:HOMONYM_CONTEXT_TOPK].tolist()

        modeA = sub[sub["_prev_act"].fillna("<START>").isin(top_prev[:1]) & sub["_next_act"].fillna("<END>").isin(top_next[:1])]
        modeB = sub.drop(modeA.index)

        if len(modeA) < max(3, HOMONYM_MIN_SUPPORT // 4) or len(modeB) < max(3, HOMONYM_MIN_SUPPORT // 4):
            continue

        # Compute divergence in prev/next distributions between modes
        all_prev = sorted(set(modeA["_prev_act"].fillna("<START>")) | set(modeB["_prev_act"].fillna("<START>")))
        all_next = sorted(set(modeA["_next_act"].fillna("<END>")) | set(modeB["_next_act"].fillna("<END>")))
        all_res = sorted(set(modeA["_res"].fillna("")) | set(modeB["_res"].fillna("")))

        def vec(counts, keys):
            return np.array([counts.get(k, 0) for k in keys], dtype=float)

        prevA = vec(modeA["_prev_act"].fillna("<START>").value_counts().to_dict(), all_prev)
        prevB = vec(modeB["_prev_act"].fillna("<START>").value_counts().to_dict(), all_prev)
        nextA = vec(modeA["_next_act"].fillna("<END>").value_counts().to_dict(), all_next)
        nextB = vec(modeB["_next_act"].fillna("<END>").value_counts().to_dict(), all_next)
        resA = vec(modeA["_res"].fillna("").value_counts().to_dict(), all_res)
        resB = vec(modeB["_res"].fillna("").value_counts().to_dict(), all_res)

        ctx_div = 0.5 * (jensen_shannon_divergence(prevA, prevB) + jensen_shannon_divergence(nextA, nextB))
        res_div = jensen_shannon_divergence(resA, resB)

        if ctx_div < HOMONYM_MIN_CONTEXT_DIVERGENCE or res_div < HOMONYM_MIN_RESOURCE_DIVERGENCE:
            continue

        # Flag rows in the smaller mode as more suspicious (less common meaning)
        suspicious = modeA if len(modeA) < len(modeB) else modeB
        common = modeB if suspicious is modeA else modeA

        common_prev = common["_prev_act"].fillna("<START>").value_counts().index[:2].tolist()
        common_next = common["_next_act"].fillna("<END>").value_counts().index[:2].tolist()

        for rid in suspicious["row_id"].tolist():
            conf = confidence_from_evidence(
                0.60,
                boosts=[
                    0.20 if ctx_div >= 0.75 else 0.10,
                    0.15 if res_div >= 0.75 else 0.05,
                    0.05 if act_counts[label] >= 20 else 0.0,
                ],
            )
            add_error(
                errors[int(rid)], "homonymous",
                conf=conf,
                tag="context_divergence_same_label",
                evidence={
                    "activity": label,
                    "ctx_divergence": ctx_div,
                    "resource_divergence": res_div,
                    "label_support": int(act_counts[label]),
                    "common_context_prev_top": common_prev,
                    "common_context_next_top": common_next,
                    "this_prev": safe_str(work.loc[work["row_id"] == rid, "_prev_act"].iloc[0]),
                    "this_next": safe_str(work.loc[work["row_id"] == rid, "_next_act"].iloc[0]),
                },
                desc=f"Same activity label '{label}' appears in divergent contexts/resources; possible homonymous meaning"
            )

    # --------- Combine multi-errors: if polluted and distorted/synonymous were detected on base, keep both ----------
    # (Already possible due to base check; nothing extra required.)

    # Build output
    out_rows = []
    for rid in range(len(work)):
        flag, types, conf, tags, evidence, desc = finalize_row(errors[rid])
        out_rows.append({
            "row_id": rid,
            "error_flag": bool(flag),
            "error_types": types,
            "error_confidence": float(conf),
            "error_tags": tags,
            "error_evidence": evidence,
            "error_description": desc
        })

    out = pd.DataFrame(out_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote: {OUTPUT_PATH}")
    print(out["error_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()