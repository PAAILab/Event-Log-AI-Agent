#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv

Detects errors using ONLY: Case, Activity, Timestamp, Resource
(Resource being empty is NOT an error.)
"""

import os
import re
import json
import math
import difflib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# --- Tunable thresholds (aggressive by design) ---
COLLATERAL_WINDOW_SECONDS = 3.0          # near-duplicate window
FORMBASED_MIN_GROUP_SIZE = 3             # same-case same-timestamp group size to consider form-based
FORMBASED_MAX_SPAN_SECONDS = 60 * 60 * 6 # if same timestamp group has wide surrounding time span, more suspicious
DISTORTED_SIMILARITY_MIN = 0.86          # typo similarity threshold
DISTORTED_LEN_MIN = 5                    # avoid tiny tokens
SYNONYM_SIMILARITY_MIN = 0.72            # looser than distorted; used with lexical cues
HOMONYM_CONTEXT_DIVERGENCE_MIN = 0.55    # divergence threshold for context vectors
HOMONYM_MIN_OCCURRENCES = 20             # only evaluate labels with enough support


# --- Helpers ---
def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def normalize_activity(a: str) -> str:
    a = safe_str(a).strip()
    a = re.sub(r"\s+", " ", a)
    return a

def normalize_activity_for_match(a: str) -> str:
    a = normalize_activity(a).lower()
    # keep parentheses content but normalize spacing
    a = re.sub(r"\s+", " ", a)
    return a

def parse_timestamp(series: pd.Series) -> pd.Series:
    # robust parsing; keep NaT if invalid
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def ratio(a, b) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def jaccard(a_set, b_set) -> float:
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0

def tokenize(a: str):
    a = normalize_activity_for_match(a)
    # words + keep meaningful tokens
    toks = re.findall(r"[a-z0-9]+", a)
    return toks

def is_polluted(activity: str):
    """
    Detect machine suffix: base + _[5-12 alnum]+_YYYYMMDD HHMMSSmmm...
    Examples show: "Request Info_47xiDPl_20230929 130852312000"
    """
    a = normalize_activity(activity)
    # common pattern: _<5-12 alnum>_<8digits><space><6digits><6+digits>
    m = re.search(r"^(.*)_([A-Za-z0-9]{5,12})_(\d{8})\s(\d{6})(\d{6,})$", a)
    if m:
        base = m.group(1).strip()
        return True, base, {"suffix": m.group(2), "date": m.group(3), "time": m.group(4), "ms": m.group(5)}
    # fallback: base + _<5-12 alnum>_ + long digit tail
    m2 = re.search(r"^(.*)_([A-Za-z0-9]{5,12})_(\d{8}).*$", a)
    if m2:
        base = m2.group(1).strip()
        return True, base, {"suffix": m2.group(2), "date": m2.group(3)}
    return False, None, None

def looks_distorted(a: str, canonical: str):
    """
    Distorted: small edit distance / high similarity but not identical.
    """
    a_n = normalize_activity_for_match(a)
    c_n = normalize_activity_for_match(canonical)
    if a_n == c_n:
        return False, 1.0
    if len(a_n) < DISTORTED_LEN_MIN or len(c_n) < DISTORTED_LEN_MIN:
        return False, ratio(a_n, c_n)
    sim = ratio(a_n, c_n)
    # also catch internal spacing splits like "Revi ew"
    if sim >= DISTORTED_SIMILARITY_MIN:
        return True, sim
    return False, sim

def build_canonical_map(activities):
    """
    Build a data-driven canonical set:
    - Use most frequent "base" for polluted variants.
    - Use frequent exact strings as canonicals.
    """
    norm = [normalize_activity(a) for a in activities]
    counts = Counter(norm)

    # map polluted -> base
    polluted_to_base = {}
    base_counts = Counter()
    for a in counts:
        ok, base, meta = is_polluted(a)
        if ok and base:
            polluted_to_base[a] = base
            base_counts[base] += counts[a]

    # candidate canonicals: frequent exact labels + frequent bases
    canon = set()
    for a, c in counts.most_common():
        if c >= 5:
            canon.add(a)
    for b, c in base_counts.most_common():
        if c >= 5:
            canon.add(b)

    # ensure at least something
    if not canon:
        canon = set(counts.keys())

    return canon, polluted_to_base, counts

def best_canonical(activity, canon_set):
    """
    Find closest canonical by similarity.
    """
    a = normalize_activity(activity)
    a_n = normalize_activity_for_match(a)
    best = None
    best_sim = -1.0
    for c in canon_set:
        c_n = normalize_activity_for_match(c)
        sim = ratio(a_n, c_n)
        if sim > best_sim:
            best_sim = sim
            best = c
    return best, best_sim


def add_error(row_id, etype, conf, tag, evidence, desc, store):
    rec = store[row_id]
    rec["types"].add(etype)
    rec["tags"].add(tag)
    rec["evidence"].append(evidence)
    rec["descriptions"].append(desc)
    rec["confidences"].append(conf)


def main():
    df = pd.read_csv(INPUT_PATH)

    # Validate columns strictly
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")

    # row_id = original row index (0-based)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    # Normalize
    df["Case"] = df["Case"].astype(str)
    df["Activity_raw"] = df["Activity"]
    df["Activity"] = df["Activity"].apply(normalize_activity)
    df["Resource"] = df["Resource"]  # keep as-is; empty is allowed
    df["Timestamp_parsed"] = parse_timestamp(df["Timestamp"])

    # Prepare output store
    store = defaultdict(lambda: {
        "types": set(),
        "tags": set(),
        "evidence": [],
        "descriptions": [],
        "confidences": []
    })

    # --- Basic format/range checks (Timestamp parseability) ---
    # (Not listed as a named error type; we will not flag as one of the 6 types.
    # But we can still include as evidence tag without setting error_flag? Spec wants only those types.
    # So we will NOT flag invalid timestamps as errors unless they trigger other rules.)
    # However, invalid timestamps break logic; we will tag them as evidence under formbased/collateral if needed.

    # --- Build canonical activity set from data ---
    canon_set, polluted_to_base, activity_counts = build_canonical_map(df["Activity"].tolist())

    # --- 1) POLLUTED detection ---
    for i, a in enumerate(df["Activity"].tolist()):
        ok, base, meta = is_polluted(a)
        if ok:
            # confidence: strong if exact pattern with date+time+ms
            conf = 0.95 if meta and {"date", "time", "ms"}.issubset(set(meta.keys())) else 0.85
            add_error(
                row_id=i,
                etype="polluted",
                conf=conf,
                tag="polluted:regex_suffix",
                evidence={"activity": a, "base_candidate": base, "meta": meta},
                desc=f"Activity appears polluted with machine suffix; base label likely '{base}'.",
                store=store
            )

    # --- 2) DISTORTED detection (typos) ---
    # For each activity, compare to best canonical; if very similar but not identical and not just polluted.
    for i, a in enumerate(df["Activity"].tolist()):
        # if polluted, compare base part for distortion too
        ok, base, meta = is_polluted(a)
        target = base if ok and base else a

        best, sim = best_canonical(target, canon_set)
        if best is None:
            continue

        distorted, dsim = looks_distorted(target, best)
        if distorted:
            # avoid flagging pure case differences as distorted
            if normalize_activity_for_match(target) == normalize_activity_for_match(best):
                continue
            # confidence increases with similarity and if canonical is frequent
            freq = activity_counts.get(best, 0)
            conf = min(0.95, 0.70 + 0.25 * (dsim - DISTORTED_SIMILARITY_MIN) / (1 - DISTORTED_SIMILARITY_MIN) + 0.05 * (1 - math.exp(-freq/20)))
            conf = float(max(0.70, conf))
            add_error(
                row_id=i,
                etype="distorted",
                conf=conf,
                tag="distorted:closest_canonical_similarity",
                evidence={"activity": a, "compared_as": target, "canonical": best, "similarity": round(dsim, 4), "canonical_freq": int(freq)},
                desc=f"Activity text likely contains a typo/character distortion; closest canonical '{best}' (sim={dsim:.3f}).",
                store=store
            )

    # --- 3) SYNONYMOUS detection (data-driven, aggressive) ---
    # Heuristic: if an activity is not close enough to any canonical (so not distorted),
    # but shares high token overlap with a frequent canonical and differs by key verb/noun substitutions.
    # We build token sets and compare to frequent canonicals.
    frequent_canon = [c for c, cnt in activity_counts.most_common() if cnt >= 10]
    canon_tokens = {c: set(tokenize(c)) for c in frequent_canon}

    # small synonym cue list (generic, domain-agnostic)
    synonym_cues = [
        ("review", {"assess", "evaluate", "inspect", "check"}),
        ("reject", {"deny", "decline", "refuse"}),
        ("approve", {"grant", "accept", "authorize", "confirm"}),
        ("start", {"begin", "initiate", "launch"}),
        ("diagnose", {"diagnosis", "determine", "establish", "confirm"}),
        ("close", {"terminate", "end", "finalize", "complete"}),
        ("request", {"ask", "solicit"}),
        ("update", {"modify", "change", "revise"}),
        ("send", {"dispatch", "issue", "notify"}),
    ]

    def cue_match(a_toks, c_toks):
        a_set = set(a_toks)
        c_set = set(c_toks)
        for head, alts in synonym_cues:
            if head in c_set and (a_set & alts):
                return True, head, list(a_set & alts)
            if head in a_set and (c_set & alts):
                return True, head, list(c_set & alts)
        return False, None, None

    for i, a in enumerate(df["Activity"].tolist()):
        ok, base, meta = is_polluted(a)
        target = base if ok and base else a

        # skip if already clearly distorted to a canonical
        best, sim = best_canonical(target, canon_set)
        if best and looks_distorted(target, best)[0]:
            continue

        a_toks = tokenize(target)
        if len(a_toks) < 2:
            continue

        best_syn = None
        best_score = -1.0
        best_cue = None

        for c in frequent_canon:
            c_toks = canon_tokens[c]
            jac = jaccard(set(a_toks), c_toks)
            sim2 = ratio(normalize_activity_for_match(target), normalize_activity_for_match(c))
            cue_ok, head, alts = cue_match(a_toks, c_toks)

            # want: moderate string similarity but not too high (else distorted),
            # and decent token overlap or cue match
            score = 0.55 * jac + 0.45 * sim2 + (0.10 if cue_ok else 0.0)
            if score > best_score:
                best_score = score
                best_syn = (c, jac, sim2, cue_ok, head, alts)

        if best_syn and best_score >= 0.78:
            c, jac, sim2, cue_ok, head, alts = best_syn
            # avoid exact match
            if normalize_activity_for_match(target) == normalize_activity_for_match(c):
                continue
            # avoid calling it synonymous when it's just very close typo
            if sim2 >= DISTORTED_SIMILARITY_MIN:
                continue

            conf = 0.70 + 0.20 * min(1.0, (best_score - 0.78) / 0.22) + (0.05 if cue_ok else 0.0)
            conf = float(min(0.92, max(0.70, conf)))
            add_error(
                row_id=i,
                etype="synonymous",
                conf=conf,
                tag="synonymous:token_overlap_and_cues",
                evidence={
                    "activity": a,
                    "compared_as": target,
                    "canonical": c,
                    "score": round(best_score, 4),
                    "jaccard": round(jac, 4),
                    "string_similarity": round(sim2, 4),
                    "cue_match": cue_ok,
                    "cue_head": head,
                    "cue_alts": alts
                },
                desc=f"Activity likely a synonym/variant wording of canonical '{c}' (score={best_score:.3f}).",
                store=store
            )

    # --- 4) COLLATERAL detection (duplicates / near-duplicates) ---
    # Exact duplicates: same Case, Activity, Timestamp, Resource
    key_cols = ["Case", "Activity", "Timestamp", "Resource"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    for i in df.index[dup_mask].tolist():
        add_error(
            row_id=int(i),
            etype="collateral",
            conf=0.95,
            tag="collateral:exact_duplicate",
            evidence={"key": {c: safe_str(df.loc[i, c]) for c in key_cols}},
            desc="Exact duplicate event (same case, activity, timestamp, resource) suggests logging artifact.",
            store=store
        )

    # Near-duplicates within short interval: same Case+Activity+Resource, timestamps within window
    # (Resource may be empty; still valid to compare)
    df_sorted = df.sort_values(["Case", "Activity", "Resource", "Timestamp_parsed", "row_id"], kind="mergesort")
    grp_cols = ["Case", "Activity", "Resource"]
    for _, g in df_sorted.groupby(grp_cols, dropna=False, sort=False):
        if len(g) < 2:
            continue
        ts = g["Timestamp_parsed"].to_numpy()
        rid = g["row_id"].to_numpy()
        # if timestamps invalid, skip near-duplicate check
        if np.all(pd.isna(ts)):
            continue
        # compute consecutive diffs
        for j in range(1, len(g)):
            t1, t0 = ts[j], ts[j-1]
            if pd.isna(t1) or pd.isna(t0):
                continue
            dt = (t1 - t0) / np.timedelta64(1, "s")
            if 0.0 < dt <= COLLATERAL_WINDOW_SECONDS:
                # flag both as collateral (aggressive)
                conf = 0.85 if dt > 0 else 0.95
                for r in (int(rid[j-1]), int(rid[j])):
                    add_error(
                        row_id=r,
                        etype="collateral",
                        conf=conf,
                        tag="collateral:near_duplicate_short_interval",
                        evidence={
                            "case": safe_str(g.iloc[j]["Case"]),
                            "activity": safe_str(g.iloc[j]["Activity"]),
                            "resource": safe_str(g.iloc[j]["Resource"]),
                            "dt_seconds": float(dt),
                            "row_pair": [int(rid[j-1]), int(rid[j])]
                        },
                        desc=f"Near-duplicate events for same case/activity/resource within {dt:.3f}s.",
                        store=store
                    )

    # --- 5) FORMBASED detection (same-case same-timestamp multi-events) ---
    # If within a case, many events share identical timestamp -> likely overwritten form time.
    df_case_ts = df.groupby(["Case", "Timestamp"], dropna=False).size().reset_index(name="n")
    suspicious = df_case_ts[df_case_ts["n"] >= FORMBASED_MIN_GROUP_SIZE]

    # For each suspicious group, check if activities/resources differ (stronger evidence)
    for _, row in suspicious.iterrows():
        case = row["Case"]
        ts_str = row["Timestamp"]
        idx = df.index[(df["Case"] == case) & (df["Timestamp"] == ts_str)].tolist()
        sub = df.loc[idx].copy()
        uniq_act = sub["Activity"].nunique(dropna=False)
        uniq_res = sub["Resource"].nunique(dropna=False)

        # compute surrounding span in that case (if parseable)
        case_sub = df[df["Case"] == case].copy()
        tmin = case_sub["Timestamp_parsed"].min()
        tmax = case_sub["Timestamp_parsed"].max()
        span = None
        if pd.notna(tmin) and pd.notna(tmax):
            span = float((tmax - tmin) / np.timedelta64(1, "s"))

        # confidence: higher if many events share timestamp and differ in activity/resource
        base_conf = 0.70 + 0.08 * min(5, (len(idx) - FORMBASED_MIN_GROUP_SIZE))
        if uniq_act >= 2:
            base_conf += 0.10
        if uniq_res >= 2:
            base_conf += 0.05
        if span is not None and span >= FORMBASED_MAX_SPAN_SECONDS:
            base_conf += 0.05
        conf = float(min(0.95, max(0.70, base_conf)))

        for i in idx:
            add_error(
                row_id=int(i),
                etype="formbased",
                conf=conf,
                tag="formbased:same_case_same_timestamp_cluster",
                evidence={
                    "case": safe_str(case),
                    "timestamp": safe_str(ts_str),
                    "cluster_size": int(len(idx)),
                    "unique_activities": int(uniq_act),
                    "unique_resources": int(uniq_res),
                    "case_time_span_seconds": span
                },
                desc=f"Multiple events in same case share identical timestamp ({len(idx)} events), consistent with form-based overwrite.",
                store=store
            )

    # --- 6) HOMONYMOUS detection (same label used for different meanings) ---
    # Data-driven: for each frequent activity label, build context vectors of (prev_activity, next_activity)
    # and detect multi-modal contexts (high divergence).
    df_case_sorted = df.sort_values(["Case", "Timestamp_parsed", "row_id"], kind="mergesort")
    # build prev/next within case
    df_case_sorted["prev_act"] = df_case_sorted.groupby("Case")["Activity"].shift(1)
    df_case_sorted["next_act"] = df_case_sorted.groupby("Case")["Activity"].shift(-1)

    # only consider labels with enough occurrences
    label_counts = df_case_sorted["Activity"].value_counts(dropna=False)
    candidate_labels = [lab for lab, cnt in label_counts.items() if cnt >= HOMONYM_MIN_OCCURRENCES and safe_str(lab) != ""]

    # Build context distributions
    for lab in candidate_labels:
        sub = df_case_sorted[df_case_sorted["Activity"] == lab]
        # context token = prev|next
        contexts = (sub["prev_act"].fillna("<START>") + "||" + sub["next_act"].fillna("<END>")).tolist()
        ctx_counts = Counter(contexts)
        if len(ctx_counts) <= 3:
            continue

        # Split into two "clusters" by most common contexts vs rest and measure divergence
        most_common = ctx_counts.most_common(3)
        top_ctx = set([c for c, _ in most_common])
        top_mass = sum(ctx_counts[c] for c in top_ctx) / sum(ctx_counts.values())

        # If top contexts don't dominate, label is used in many different surroundings -> possible homonym
        # Also check resource diversity as supporting evidence
        res_div = sub["Resource"].nunique(dropna=False) / max(1, len(sub))
        # divergence proxy
        divergence = 1.0 - top_mass

        if divergence >= HOMONYM_CONTEXT_DIVERGENCE_MIN and top_mass <= 0.60:
            # choose "canonical meaning" as top context; others are suspect
            # flag rows not in top contexts
            suspect_rows = sub[~((sub["prev_act"].fillna("<START>") + "||" + sub["next_act"].fillna("<END>")).isin(top_ctx))]
            if len(suspect_rows) < 3:
                continue

            conf = 0.70 + 0.20 * min(1.0, (divergence - HOMONYM_CONTEXT_DIVERGENCE_MIN) / (1 - HOMONYM_CONTEXT_DIVERGENCE_MIN))
            if res_div > 0.10:
                conf += 0.05
            conf = float(min(0.90, max(0.70, conf)))

            for _, r in suspect_rows.iterrows():
                rid = int(r["row_id"])
                add_error(
                    row_id=rid,
                    etype="homonymous",
                    conf=conf,
                    tag="homonymous:context_divergence_prev_next",
                    evidence={
                        "activity": safe_str(lab),
                        "prev_act": safe_str(r["prev_act"]) if pd.notna(r["prev_act"]) else None,
                        "next_act": safe_str(r["next_act"]) if pd.notna(r["next_act"]) else None,
                        "top_contexts": list(top_ctx),
                        "top_mass": round(top_mass, 4),
                        "divergence": round(divergence, 4),
                        "label_count": int(label_counts[lab]),
                    },
                    desc=f"Activity label '{lab}' appears in divergent process contexts, suggesting homonymous usage.",
                    store=store
                )

    # --- Combine results into output dataframe ---
    out = pd.DataFrame({"row_id": df["row_id"].astype(int)})

    def finalize_row(rid):
        rec = store.get(int(rid))
        if not rec:
            return (False, "", 0.0, "", "", "")
        types = sorted(rec["types"])
        tags = sorted(rec["tags"])
        # confidence: combine as 1 - Π(1-ci) (noisy-or)
        confs = rec["confidences"]
        p_not = 1.0
        for c in confs:
            p_not *= (1.0 - float(c))
        conf = 1.0 - p_not
        conf = float(min(1.0, max(0.0, conf)))

        evidence = rec["evidence"]
        desc = rec["descriptions"]

        return (
            True,
            "|".join(types),
            conf,
            "|".join(tags),
            json.dumps(evidence, ensure_ascii=False),
            " ".join(desc)
        )

    finalized = out["row_id"].apply(finalize_row)
    out["error_flag"] = finalized.apply(lambda x: x[0])
    out["error_types"] = finalized.apply(lambda x: x[1])
    out["error_confidence"] = finalized.apply(lambda x: x[2])
    out["error_tags"] = finalized.apply(lambda x: x[3])
    out["error_evidence"] = finalized.apply(lambda x: x[4])
    out["error_description"] = finalized.apply(lambda x: x[5])

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote: {OUTPUT_PATH}")
    print(out["error_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()