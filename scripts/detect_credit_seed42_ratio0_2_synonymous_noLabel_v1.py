#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_synonymous_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_synonymous_noLabel.detected.csv

Detects (based ONLY on Case, Activity, Timestamp, Resource):
- formbased
- polluted
- distorted
- synonymous
- collateral
- homonymous (heuristic, conservative)

Resource being empty is NOT an error.
"""

import os
import re
import json
import math
import difflib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_synonymous_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_synonymous_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# --- Tunable thresholds (aggressive but not reckless) ---
COLLATERAL_WINDOW_SECONDS = 3.0          # near-duplicate window
FORMBASED_MIN_GROUP_SIZE = 3             # repeated same timestamp in a case
FORMBASED_MIN_DISTINCT_ACTIVITIES = 2    # avoid flagging simple duplicates only
FORMBASED_MIN_DISTINCT_RESOURCES = 2     # if resources vary, stronger evidence
DISTORTED_MIN_RATIO = 0.86               # similarity to canonical to call typo
DISTORTED_MAX_RATIO = 0.97               # above this likely exact/synonym, not typo
SYNONYM_MIN_RATIO = 0.72                 # similarity to canonical for synonym candidate
SYNONYM_MAX_RATIO = 0.90                 # below distorted band; synonyms often lower similarity
HOMONYM_CONTEXT_JACCARD_MAX = 0.15       # very different context => possible homonym
HOMONYM_MIN_OCCURRENCES = 20             # only consider labels with enough support
HOMONYM_MIN_CASES = 10                   # spread across cases
HOMONYM_MIN_CLUSTER_SEPARATION = 0.35    # separation between context clusters


# ----------------- Helpers -----------------
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def norm_space(s: str) -> str:
    s = safe_str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_activity(s: str) -> str:
    s = norm_space(s).lower()
    # normalize separators
    s = s.replace("-", " ")
    s = re.sub(r"[^\w\s\(\)]", " ", s)  # keep parentheses for potential semantics
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_pollution(activity_raw: str):
    """
    Detect and strip machine suffixes like:
      "Request Info_47xiDPl_20230929 130852312000"
    Returns: (is_polluted, canonical_guess, evidence)
    """
    a = norm_space(activity_raw)

    # Pattern: base + _ + 5-12 alnum + _ + yyyymmdd + space + 9-15 digits
    pat = re.compile(r"^(?P<base>.+?)_(?P<tok>[A-Za-z0-9]{5,12})_(?P<date>\d{8})\s(?P<num>\d{9,15})$")
    m = pat.match(a)
    if m:
        base = norm_space(m.group("base"))
        ev = {
            "pattern": "base_TOKEN_YYYYMMDD NNN",
            "token": m.group("tok"),
            "date": m.group("date"),
            "num_len": len(m.group("num")),
            "base": base,
        }
        return True, base, ev

    # Slightly looser: base + _ + 5-12 alnum + _ + yyyymmdd + optional space + digits
    pat2 = re.compile(r"^(?P<base>.+?)_(?P<tok>[A-Za-z0-9]{5,12})_(?P<date>\d{8})(?:\s(?P<num>\d{6,18}))?$")
    m2 = pat2.match(a)
    if m2:
        base = norm_space(m2.group("base"))
        ev = {
            "pattern": "base_TOKEN_YYYYMMDD[ digits]",
            "token": m2.group("tok"),
            "date": m2.group("date"),
            "has_num": m2.group("num") is not None,
            "base": base,
        }
        return True, base, ev

    return False, None, None

def seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def jaccard(a_set, b_set) -> float:
    if not a_set and not b_set:
        return 1.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0

def add_error(row_errors, row_id, etype, conf, tag, evidence, desc):
    rec = row_errors[row_id]
    rec["types"].append(etype)
    rec["conf"].append(conf)
    rec["tags"].append(tag)
    rec["evidence"].append(evidence)
    rec["desc"].append(desc)

def finalize_row(row_errors, n_rows):
    out = []
    for rid in range(n_rows):
        if rid not in row_errors:
            out.append({
                "row_id": rid,
                "error_flag": False,
                "error_types": "",
                "error_confidence": 0.0,
                "error_tags": "",
                "error_evidence": "",
                "error_description": ""
            })
            continue

        rec = row_errors[rid]
        # unique types/tags but keep evidence/desc concatenated
        types = list(dict.fromkeys(rec["types"]))
        tags = list(dict.fromkeys(rec["tags"]))

        # combine confidence: 1 - Π(1-ci) (noisy-or)
        confs = [max(0.0, min(1.0, float(c))) for c in rec["conf"]]
        p_not = 1.0
        for c in confs:
            p_not *= (1.0 - c)
        combined = 1.0 - p_not

        out.append({
            "row_id": rid,
            "error_flag": True,
            "error_types": "|".join(sorted(types)),
            "error_confidence": round(float(combined), 6),
            "error_tags": "|".join(sorted(tags)),
            "error_evidence": json.dumps(rec["evidence"], ensure_ascii=False),
            "error_description": " ; ".join(rec["desc"])
        })
    return pd.DataFrame(out)


# ----------------- Main detection -----------------
def main():
    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

    # Keep only required columns for detection (per instruction)
    work = df[REQUIRED_COLS].copy()

    # row_id is the original row index in file order
    work.reset_index(drop=True, inplace=True)
    n = len(work)

    # Parse timestamps
    work["Timestamp_parsed"] = pd.to_datetime(work["Timestamp"], errors="coerce", utc=False)

    # Normalize
    work["Activity_raw"] = work["Activity"].apply(safe_str)
    work["Activity_norm"] = work["Activity_raw"].apply(norm_activity)
    work["Case_norm"] = work["Case"].apply(safe_str)
    work["Resource_norm"] = work["Resource"].apply(lambda x: norm_space(x))  # empty allowed

    row_errors = defaultdict(lambda: {"types": [], "conf": [], "tags": [], "evidence": [], "desc": []})

    # --- Basic format/range checks (allowed: only these columns) ---
    # Timestamp parse failures are real errors (cannot order events)
    bad_ts = work["Timestamp_parsed"].isna()
    for rid in work.index[bad_ts]:
        add_error(
            row_errors, rid,
            "timestamp_format", 0.98,
            "TS_PARSE_FAIL",
            {"timestamp": safe_str(work.at[rid, "Timestamp"])},
            f"Timestamp not parseable: '{safe_str(work.at[rid, 'Timestamp'])}'"
        )

    # Empty Case or Activity are errors (Resource empty is NOT)
    empty_case = work["Case_norm"].str.strip().eq("") | work["Case_norm"].isna()
    for rid in work.index[empty_case]:
        add_error(
            row_errors, rid,
            "case_missing", 0.95,
            "CASE_EMPTY",
            {"case": safe_str(work.at[rid, "Case"])},
            "Case identifier is missing/empty"
        )

    empty_act = work["Activity_norm"].str.strip().eq("")
    for rid in work.index[empty_act]:
        add_error(
            row_errors, rid,
            "activity_missing", 0.95,
            "ACT_EMPTY",
            {"activity": safe_str(work.at[rid, "Activity"])},
            "Activity label is missing/empty"
        )

    # --- Polluted detection ---
    work["is_polluted"] = False
    work["polluted_base"] = None
    work["polluted_ev"] = None
    for rid, a in enumerate(work["Activity_raw"].tolist()):
        is_p, base, ev = strip_pollution(a)
        if is_p:
            work.at[rid, "is_polluted"] = True
            work.at[rid, "polluted_base"] = base
            work.at[rid, "polluted_ev"] = ev
            add_error(
                row_errors, rid,
                "polluted", 0.97,
                "ACT_POLLUTION_SUFFIX",
                {"activity": a, "canonical_guess": base, "match": ev},
                f"Activity appears polluted with machine suffix; base='{base}'"
            )

    # Canonical activity candidate for similarity checks:
    # if polluted -> use stripped base, else raw
    work["Activity_cand"] = work.apply(
        lambda r: norm_activity(r["polluted_base"]) if r["is_polluted"] and r["polluted_base"] else r["Activity_norm"],
        axis=1
    )

    # --- Collateral detection (duplicates / near-duplicates) ---
    # Exact duplicates: same Case, Activity_cand, Timestamp, Resource_norm
    # (Resource empty allowed; still part of duplicate signature)
    sig_cols = ["Case_norm", "Activity_cand", "Timestamp_parsed", "Resource_norm"]
    dup_mask = work.duplicated(subset=sig_cols, keep=False) & (~work["Timestamp_parsed"].isna())
    for rid in work.index[dup_mask]:
        grp = work.loc[dup_mask & (work[sig_cols].eq(work.loc[rid, sig_cols]).all(axis=1)), sig_cols].head(5)
        add_error(
            row_errors, rid,
            "collateral", 0.96,
            "DUP_EXACT",
            {"signature": {c: safe_str(work.at[rid, c]) for c in sig_cols},
             "sample_group": grp.astype(str).to_dict(orient="records")},
            "Exact duplicate event (same case, activity, timestamp, resource) suggests collateral logging"
        )

    # Near duplicates: within window, same Case + Activity_cand + Resource_norm
    # Sort by case then time
    work_sorted = work.sort_values(["Case_norm", "Timestamp_parsed", "Activity_cand", "Resource_norm"], kind="mergesort")
    idxs = work_sorted.index.to_list()

    # Build per-case sequences for near-dup check
    by_case = defaultdict(list)
    for rid in idxs:
        by_case[work.at[rid, "Case_norm"]].append(rid)

    for case, rids in by_case.items():
        # consider only rows with valid timestamps
        rids_valid = [rid for rid in rids if not pd.isna(work.at[rid, "Timestamp_parsed"])]
        # sort by time
        rids_valid.sort(key=lambda rid: work.at[rid, "Timestamp_parsed"])
        for i in range(1, len(rids_valid)):
            r_prev = rids_valid[i - 1]
            r_cur = rids_valid[i]
            if work.at[r_prev, "Activity_cand"] != work.at[r_cur, "Activity_cand"]:
                continue
            if work.at[r_prev, "Resource_norm"] != work.at[r_cur, "Resource_norm"]:
                continue
            dt = (work.at[r_cur, "Timestamp_parsed"] - work.at[r_prev, "Timestamp_parsed"]).total_seconds()
            if 0.0 <= dt <= COLLATERAL_WINDOW_SECONDS:
                # flag both (aggressive)
                base = work.at[r_cur, "Activity_cand"]
                conf = 0.90 if dt > 0 else 0.95
                ev = {
                    "case": case,
                    "activity": base,
                    "resource": work.at[r_cur, "Resource_norm"],
                    "prev_row_id": int(r_prev),
                    "cur_row_id": int(r_cur),
                    "dt_seconds": float(dt),
                    "prev_ts": str(work.at[r_prev, "Timestamp_parsed"]),
                    "cur_ts": str(work.at[r_cur, "Timestamp_parsed"]),
                }
                add_error(
                    row_errors, r_cur,
                    "collateral", conf,
                    "DUP_NEAR_TIME",
                    ev,
                    f"Near-duplicate event within {dt:.3f}s for same case/activity/resource suggests collateral logging"
                )
                add_error(
                    row_errors, r_prev,
                    "collateral", conf * 0.85,
                    "DUP_NEAR_TIME",
                    ev,
                    f"Near-duplicate event within {dt:.3f}s for same case/activity/resource suggests collateral logging"
                )

    # --- Formbased detection (same timestamp reused within a case) ---
    # For each case, if a timestamp appears many times across different activities/resources, flag those rows.
    valid_ts = ~work["Timestamp_parsed"].isna()
    grp = work[valid_ts].groupby(["Case_norm", "Timestamp_parsed"])
    for (case, ts), g in grp:
        if len(g) < FORMBASED_MIN_GROUP_SIZE:
            continue
        distinct_acts = g["Activity_cand"].nunique(dropna=True)
        distinct_res = g["Resource_norm"].nunique(dropna=False)
        if distinct_acts < FORMBASED_MIN_DISTINCT_ACTIVITIES:
            continue

        # If it's just exact duplicates, collateral already covers; still can be formbased if many different acts.
        # Confidence increases with size and diversity.
        size_factor = min(1.0, (len(g) - 2) / 6.0)  # 3->0.166, 8->1.0
        div_factor = min(1.0, (distinct_acts - 1) / 4.0)
        res_factor = 0.2 if distinct_res >= FORMBASED_MIN_DISTINCT_RESOURCES else 0.0
        conf = 0.70 + 0.20 * size_factor + 0.10 * div_factor + res_factor
        conf = min(0.97, conf)

        ev = {
            "case": case,
            "timestamp": str(ts),
            "group_size": int(len(g)),
            "distinct_activities": int(distinct_acts),
            "distinct_resources": int(distinct_res),
            "activities_sample": g["Activity_raw"].head(8).tolist(),
            "row_ids": g.index.astype(int).tolist()
        }
        for rid in g.index:
            add_error(
                row_errors, int(rid),
                "formbased", conf,
                "CASE_SAME_TIMESTAMP_MANY",
                ev,
                f"Multiple events in same case share identical timestamp {ts} across {distinct_acts} activities (form-based overwrite likely)"
            )

    # --- Build canonical activity set (from "cleaner" labels) ---
    # Use activity candidates that are not polluted and not empty.
    base_acts = work.loc[~work["Activity_cand"].eq("") & (~work["is_polluted"]), "Activity_cand"].tolist()
    base_counts = Counter(base_acts)

    # Keep frequent ones as canonical anchors
    # (aggressive: include even low frequency, but prefer higher frequency for matching)
    canonical = [a for a, c in base_counts.most_common() if len(a) >= 3]
    canonical_set = set(canonical)

    # --- Distorted & Synonymous detection via similarity to canonical anchors ---
    # For each activity candidate, find best canonical match (excluding itself).
    # Distorted: very high similarity but not exact.
    # Synonymous: moderate similarity and different tokens (heuristic).
    def token_set(s):
        return set([t for t in s.split() if t])

    for rid in range(n):
        a = work.at[rid, "Activity_cand"]
        if not a:
            continue

        # If polluted, we still can detect distorted/synonymous on base
        # but avoid matching to itself if it is already canonical.
        best = None
        best_r = -1.0
        for c in canonical:
            if c == a:
                continue
            r = seq_ratio(a, c)
            if r > best_r:
                best_r = r
                best = c

        if best is None:
            continue

        # Distorted: close spelling variants
        if DISTORTED_MIN_RATIO <= best_r <= DISTORTED_MAX_RATIO:
            # extra evidence: token overlap high
            jac = jaccard(token_set(a), token_set(best))
            conf = 0.70 + 0.25 * (best_r - DISTORTED_MIN_RATIO) / (DISTORTED_MAX_RATIO - DISTORTED_MIN_RATIO + 1e-9)
            conf += 0.10 * min(1.0, jac / 0.8)
            conf = min(0.95, conf)

            add_error(
                row_errors, rid,
                "distorted", conf,
                "ACT_SIMILAR_TYPO",
                {"activity": work.at[rid, "Activity_raw"], "normalized": a, "canonical_guess": best,
                 "similarity": round(best_r, 4), "token_jaccard": round(jac, 4)},
                f"Activity '{work.at[rid, 'Activity_raw']}' looks like a typo/distortion of '{best}' (sim={best_r:.3f})"
            )

        # Synonymous: not too close (avoid typos), but still similar; also token sets differ meaningfully
        if SYNONYM_MIN_RATIO <= best_r <= SYNONYM_MAX_RATIO:
            ta, tb = token_set(a), token_set(best)
            jac = jaccard(ta, tb)
            # synonyms often share fewer exact tokens than typos; require not-too-high overlap
            if jac <= 0.75 and (ta != tb):
                conf = 0.62 + 0.25 * (best_r - SYNONYM_MIN_RATIO) / (SYNONYM_MAX_RATIO - SYNONYM_MIN_RATIO + 1e-9)
                # if overlap is very low, reduce (could be unrelated)
                if jac < 0.2:
                    conf -= 0.12
                conf = max(0.50, min(0.90, conf))

                add_error(
                    row_errors, rid,
                    "synonymous", conf,
                    "ACT_SEMANTIC_VARIANT_HEUR",
                    {"activity": work.at[rid, "Activity_raw"], "normalized": a, "canonical_guess": best,
                     "similarity": round(best_r, 4), "token_jaccard": round(jac, 4),
                     "tokens_activity": sorted(list(ta))[:20], "tokens_canonical": sorted(list(tb))[:20]},
                    f"Activity '{work.at[rid, 'Activity_raw']}' may be a synonym/variant of '{best}' (sim={best_r:.3f}, token_jac={jac:.3f})"
                )

    # --- Homonymous detection (conservative heuristic) ---
    # Idea: same label used in very different local contexts (prev/next activities) across cases.
    # Build context signatures for each occurrence: (prev_activity, next_activity) using Activity_cand within case order.
    # Then for each label with enough occurrences, see if contexts split into two clusters with low overlap.
    # This is heuristic; keep confidence moderate.
    # Prepare per-case ordered events
    work_valid = work[~work["Timestamp_parsed"].isna()].copy()
    work_valid.sort_values(["Case_norm", "Timestamp_parsed"], inplace=True, kind="mergesort")

    prev_act = {}
    next_act = {}
    for case, g in work_valid.groupby("Case_norm", sort=False):
        rids = g.index.to_list()
        acts = g["Activity_cand"].tolist()
        for i, rid in enumerate(rids):
            prev_act[rid] = acts[i - 1] if i > 0 else None
            next_act[rid] = acts[i + 1] if i < len(rids) - 1 else None

    # Collect contexts per label
    label_occ = defaultdict(list)  # label -> list of (rid, prev, next)
    for rid in work_valid.index:
        lab = work.at[rid, "Activity_cand"]
        if not lab:
            continue
        label_occ[lab].append((rid, prev_act.get(rid), next_act.get(rid)))

    for lab, occ in label_occ.items():
        if len(occ) < HOMONYM_MIN_OCCURRENCES:
            continue
        cases = set(work.at[rid, "Case_norm"] for rid, _, _ in occ)
        if len(cases) < HOMONYM_MIN_CASES:
            continue

        # Build context sets
        ctx_sets = []
        for rid, p, nx in occ:
            s = set()
            if p: s.add(f"prev:{p}")
            if nx: s.add(f"next:{nx}")
            ctx_sets.append((rid, s))

        # Find two most dissimilar contexts by jaccard
        # (O(n^2) but only for frequent labels; still manageable)
        max_dist = -1.0
        pair = None
        for i in range(min(len(ctx_sets), 200)):  # cap for speed
            rid_i, si = ctx_sets[i]
            for j in range(i + 1, min(len(ctx_sets), 200)):
                rid_j, sj = ctx_sets[j]
                jac = jaccard(si, sj)
                dist = 1.0 - jac
                if dist > max_dist:
                    max_dist = dist
                    pair = (rid_i, rid_j, si, sj, jac)

        if pair is None:
            continue

        rid_a, rid_b, s_a, s_b, jac_ab = pair
        # If contexts are extremely different, attempt clustering by similarity to these two seeds
        if jac_ab > HOMONYM_CONTEXT_JACCARD_MAX:
            continue

        clusterA, clusterB = [], []
        for rid, s in ctx_sets:
            ja = jaccard(s, s_a)
            jb = jaccard(s, s_b)
            if ja >= jb:
                clusterA.append((rid, s, ja))
            else:
                clusterB.append((rid, s, jb))

        if len(clusterA) < 5 or len(clusterB) < 5:
            continue

        # Compute average within-cluster similarity and between-cluster similarity
        avgA = float(np.mean([x[2] for x in clusterA]))
        avgB = float(np.mean([x[2] for x in clusterB]))
        # between: compare each cluster member to opposite seed
        avg_between = float(np.mean([jaccard(s, s_b) for _, s, _ in clusterA] + [jaccard(s, s_a) for _, s, _ in clusterB]))
        separation = (avgA + avgB) / 2.0 - avg_between

        if separation < HOMONYM_MIN_CLUSTER_SEPARATION:
            continue

        # Flag occurrences that strongly belong to either cluster (high similarity to its seed)
        # Confidence depends on separation and support
        support = min(1.0, len(occ) / 80.0)
        conf = 0.55 + 0.25 * min(1.0, separation) + 0.10 * support
        conf = min(0.85, conf)

        ev = {
            "label": lab,
            "occurrences": int(len(occ)),
            "cases": int(len(cases)),
            "seedA_row": int(rid_a),
            "seedB_row": int(rid_b),
            "seedA_ctx": sorted(list(s_a)),
            "seedB_ctx": sorted(list(s_b)),
            "seed_jaccard": round(float(jac_ab), 4),
            "avgA": round(avgA, 4),
            "avgB": round(avgB, 4),
            "avg_between": round(avg_between, 4),
            "separation": round(float(separation), 4),
        }

        # Flag top members from each cluster (strongest membership)
        clusterA_sorted = sorted(clusterA, key=lambda x: x[2], reverse=True)[:max(10, int(0.15 * len(clusterA)))]
        clusterB_sorted = sorted(clusterB, key=lambda x: x[2], reverse=True)[:max(10, int(0.15 * len(clusterB)))]

        for rid, s, mem in clusterA_sorted + clusterB_sorted:
            add_error(
                row_errors, int(rid),
                "homonymous", conf,
                "ACT_CONTEXT_SPLIT",
                {**ev, "row_ctx": sorted(list(s)), "membership": round(float(mem), 4)},
                f"Activity label '{lab}' appears in two very different contexts (possible homonymous usage)"
            )

    # --- Write output ---
    out_df = finalize_row(row_errors, n)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Flagged rows: {int(out_df['error_flag'].sum())} / {len(out_df)}")


if __name__ == "__main__":
    main()