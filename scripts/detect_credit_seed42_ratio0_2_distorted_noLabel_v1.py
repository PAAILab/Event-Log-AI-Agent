#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection (Credit dataset)

Reads:
  /home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_distorted_noLabel.csv

Writes:
  /home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_distorted_noLabel.detected.csv

Detects (ONLY using Case, Activity, Timestamp, Resource):
  - formbased
  - polluted
  - distorted
  - synonymous
  - collateral
  - homonymous (heuristic, evidence-based)

Notes:
  - Empty Resource is NOT an error (do not flag).
"""

import os
import re
import math
import json
from collections import defaultdict, Counter

import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_distorted_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_distorted_noLabel.detected.csv"

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
    """Lowercase + collapse spaces; keep punctuation (helps distinguish labels)."""
    s = norm_space(s).lower()
    return s

def strip_pollution(activity: str):
    """
    Detect and strip machine suffix patterns like:
      "Request Info_47xiDPl_20230929 130852312000"
    Returns (base, polluted_flag, evidence)
    """
    a = norm_space(activity)

    # Common pattern: <base>_<5-12 alnum>_<YYYYMMDD HHMMSS...>
    # Allow optional underscore/space between date and time.
    pat = re.compile(
        r"^(?P<base>.+?)_(?P<code>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}[ _]\d{6,18})$"
    )
    m = pat.match(a)
    if m:
        base = norm_space(m.group("base"))
        code = m.group("code")
        dt = m.group("dt")
        return base, True, {"pattern": "base_code_datetime", "code": code, "suffix_dt": dt}

    # Alternative: <base>_<5-12 alnum> (no datetime)
    pat2 = re.compile(r"^(?P<base>.+?)_(?P<code>[A-Za-z0-9]{5,12})$")
    m2 = pat2.match(a)
    if m2:
        base = norm_space(m2.group("base"))
        code = m2.group("code")
        return base, True, {"pattern": "base_code", "code": code}

    return a, False, None


# -----------------------------
# Utility: edit distance
# -----------------------------
def levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein distance."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    # Ensure a is shorter for memory
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, start=1):
        cur = [i]
        for j, ca in enumerate(a, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def similarity_ratio(a: str, b: str) -> float:
    """1 - normalized edit distance."""
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    d = levenshtein(a, b)
    denom = max(len(a), len(b), 1)
    return 1.0 - (d / denom)


# -----------------------------
# Confidence aggregation
# -----------------------------
def combine_confidences(confs):
    """Noisy-OR combine."""
    confs = [c for c in confs if c is not None]
    if not confs:
        return 0.0
    p_not = 1.0
    for c in confs:
        c = max(0.0, min(1.0, float(c)))
        p_not *= (1.0 - c)
    return 1.0 - p_not

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# -----------------------------
# Main detection
# -----------------------------
def main():
    df = pd.read_csv(INPUT_PATH)

    # Aggressive column validation (but do not use other columns for detection)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Keep only required columns for detection logic
    work = df[REQUIRED_COLS].copy()

    # row_id = original row index (0-based)
    work["row_id"] = work.index.astype(int)

    # Parse timestamps
    # If parsing fails -> treat as strong error signal (format/range) but we must map to allowed types.
    # We'll tag as distorted (timestamp) is not in spec; instead we will treat as "formbased" evidence? No.
    # So we only use it to support other detections; if timestamp invalid, we flag as "collateral" is wrong.
    # Best: flag as "formbased" not correct. Therefore: we will flag as "distorted" on Activity only.
    # For timestamp invalid, we will still flag error as "formbased" is incorrect; so we create a tag
    # but do NOT set error_flag unless another allowed type triggers.
    # (Dataset spec doesn't include timestamp-format error type.)
    work["_ts"] = pd.to_datetime(work["Timestamp"], errors="coerce")

    # Normalize fields
    work["_case"] = work["Case"].astype(str)
    work["_activity_raw"] = work["Activity"].astype(str).map(norm_space)
    work["_resource"] = work["Resource"]  # keep as-is; empty is allowed

    # Polluted detection + base extraction
    bases = []
    polluted_flags = []
    polluted_evs = []
    for a in work["_activity_raw"].tolist():
        base, is_poll, ev = strip_pollution(a)
        bases.append(base)
        polluted_flags.append(is_poll)
        polluted_evs.append(ev)
    work["_activity_base"] = bases
    work["_polluted"] = polluted_flags
    work["_polluted_ev"] = polluted_evs

    # Canonical label set: use most frequent base labels as "clean" anchors
    # (Aggressive: assume majority are clean; polluted/distorted are minority)
    base_counts = Counter(work["_activity_base"].map(norm_key))
    # Keep labels that appear at least 2 times OR are in top-K
    top_k = 80
    canonical = set([k for k, _ in base_counts.most_common(top_k)] + [k for k, v in base_counts.items() if v >= 2])

    # Build mapping from observed base -> nearest canonical (for distorted/synonymous)
    canonical_list = sorted(canonical)

    def nearest_canonical(label_key: str):
        """Return (best_key, best_sim) among canonical labels."""
        if label_key in canonical:
            return label_key, 1.0
        best = None
        best_sim = -1.0
        # prune by length window for speed
        L = len(label_key)
        for ck in canonical_list:
            if abs(len(ck) - L) > 10:
                continue
            sim = similarity_ratio(label_key, ck)
            if sim > best_sim:
                best_sim = sim
                best = ck
        return best, best_sim

    # Synonym heuristic:
    # If two labels are not close by edit distance but share strong token overlap,
    # treat as potential synonym (e.g., "assess application" vs "review application").
    stop = set(["the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "case", "request", "application"])
    def tokens(s):
        s = re.sub(r"[^A-Za-z0-9 ]+", " ", s.lower())
        t = [w for w in s.split() if w and w not in stop]
        return t

    canon_tokens = {ck: set(tokens(ck)) for ck in canonical_list}

    def synonym_candidate(label_key: str):
        t = set(tokens(label_key))
        if not t:
            return None, 0.0
        best = None
        best_score = 0.0
        for ck, ct in canon_tokens.items():
            if not ct:
                continue
            inter = len(t & ct)
            union = len(t | ct)
            j = inter / union if union else 0.0
            # require some overlap but not too similar by edit distance (to separate from distorted)
            if j > best_score:
                best_score = j
                best = ck
        return best, best_score

    # Precompute per-row label keys
    work["_base_key"] = work["_activity_base"].map(norm_key)

    # Distorted detection:
    # If not canonical but very close to a canonical label => distorted
    distorted_target = []
    distorted_sim = []
    for k in work["_base_key"].tolist():
        best, sim = nearest_canonical(k)
        distorted_target.append(best)
        distorted_sim.append(sim)
    work["_nearest_canon"] = distorted_target
    work["_nearest_sim"] = distorted_sim

    # Synonymous detection:
    syn_target = []
    syn_score = []
    for k in work["_base_key"].tolist():
        best, score = synonym_candidate(k)
        syn_target.append(best)
        syn_score.append(score)
    work["_syn_canon"] = syn_target
    work["_syn_score"] = syn_score

    # Collateral detection:
    # 1) exact duplicates: same case, activity_raw, timestamp, resource
    # 2) near duplicates: same case + base_key + resource within <= 3 seconds
    #    (aggressive threshold; typical logging artifacts)
    work["_resource_key"] = work["_resource"].astype(str).fillna("").map(norm_space).map(norm_key)

    # exact dup groups
    exact_cols = ["_case", "_activity_raw", "Timestamp", "_resource_key"]
    exact_dup = work.duplicated(subset=exact_cols, keep=False)

    # near dup: sort by case then time
    work_sorted = work.sort_values(by=["_case", "_ts", "row_id"], kind="mergesort").copy()
    work_sorted["_prev_case"] = work_sorted["_case"].shift(1)
    work_sorted["_prev_base"] = work_sorted["_base_key"].shift(1)
    work_sorted["_prev_res"] = work_sorted["_resource_key"].shift(1)
    work_sorted["_prev_ts"] = work_sorted["_ts"].shift(1)
    dt = (work_sorted["_ts"] - work_sorted["_prev_ts"]).dt.total_seconds()
    near_dup = (
        (work_sorted["_case"] == work_sorted["_prev_case"]) &
        (work_sorted["_base_key"] == work_sorted["_prev_base"]) &
        (work_sorted["_resource_key"] == work_sorted["_prev_res"]) &
        (dt.notna()) & (dt >= 0) & (dt <= 3)
    )
    near_dup_rowids = set(work_sorted.loc[near_dup, "row_id"].tolist())
    # also mark the previous row in the near-dup pair
    near_dup_prev_rowids = set(work_sorted.loc[near_dup, "row_id"].shift(1).dropna().astype(int).tolist())
    near_dup_all = near_dup_rowids | near_dup_prev_rowids

    # Formbased detection:
    # Multiple events in same case share identical timestamp (to millisecond) with different activities/resources.
    # Flag all but the first occurrence at that timestamp within the case (aggressive).
    formbased_rowids = set()
    formbased_groups = defaultdict(list)
    for rid, c, ts in work[["row_id", "_case", "Timestamp"]].itertuples(index=False):
        formbased_groups[(c, ts)].append(rid)
    for (c, ts), rids in formbased_groups.items():
        if len(rids) >= 3:  # aggressive: require at least 3 events overwritten to reduce false positives
            # mark all except earliest row_id as overwritten
            for rid in sorted(rids)[1:]:
                formbased_rowids.add(rid)

    # Homonymous detection (heuristic):
    # Same activity label appears in two distinct "contexts" (prev/next activity base) with strong separation.
    # We compute context signatures per (case, position) and cluster by (prev_base, next_base).
    # If a label has >=2 frequent context signatures with low overlap, mark rows in minority signature as homonymous.
    work_sorted2 = work.sort_values(by=["_case", "_ts", "row_id"], kind="mergesort").copy()
    work_sorted2["_prev_act"] = work_sorted2.groupby("_case")["_base_key"].shift(1)
    work_sorted2["_next_act"] = work_sorted2.groupby("_case")["_base_key"].shift(-1)
    work_sorted2["_ctx"] = list(zip(
        work_sorted2["_prev_act"].fillna("<<START>>"),
        work_sorted2["_next_act"].fillna("<<END>>")
    ))

    label_ctx_counts = defaultdict(Counter)
    for k, ctx in zip(work_sorted2["_base_key"], work_sorted2["_ctx"]):
        label_ctx_counts[k][ctx] += 1

    homonymous_rowids = set()
    homonymous_evidence = {}  # row_id -> dict
    for label, ctx_counter in label_ctx_counts.items():
        if sum(ctx_counter.values()) < 30:
            continue
        common = ctx_counter.most_common(3)
        if len(common) < 2:
            continue
        (ctx1, n1), (ctx2, n2) = common[0], common[1]
        # require both contexts substantial
        if n2 < 8:
            continue
        # require contexts "different enough"
        overlap = (set(ctx1) & set(ctx2))
        if len(overlap) >= 1 and ctx1[0] == ctx2[0] and ctx1[1] == ctx2[1]:
            continue
        # mark rows belonging to the less frequent context as suspicious homonymous
        minority_ctx = ctx2 if n2 <= n1 else ctx1
        rows = work_sorted2.loc[(work_sorted2["_base_key"] == label) & (work_sorted2["_ctx"] == minority_ctx), ["row_id", "_ctx"]]
        for rid, ctx in rows.itertuples(index=False):
            homonymous_rowids.add(int(rid))
            homonymous_evidence[int(rid)] = {
                "label": label,
                "minority_context": minority_ctx,
                "top_contexts": [(str(ctx1), n1), (str(ctx2), n2)]
            }

    # Build output rows
    out = pd.DataFrame({"row_id": work["row_id"].astype(int)})

    error_types = []
    error_tags = []
    error_evidence = []
    error_desc = []
    error_conf = []

    for i, r in work.iterrows():
        rid = int(r["row_id"])
        types = []
        tags = []
        evid = {}
        confs = []
        desc_parts = []

        # POLLUTED
        if bool(r["_polluted"]):
            types.append("polluted")
            tags.append("ACT_POLLUTION_SUFFIX")
            evid["polluted"] = {"raw": r["_activity_raw"], "base": r["_activity_base"], "match": r["_polluted_ev"]}
            confs.append(0.97)
            desc_parts.append(f"Activity has machine-generated suffix; base='{r['_activity_base']}' from raw='{r['_activity_raw']}'.")

        # DISTORTED (typo) - only if not canonical and close to canonical
        # Avoid calling canonical itself distorted; also avoid if it's clearly synonym (token overlap high but edit sim low)
        k = r["_base_key"]
        nearest = r["_nearest_canon"]
        sim = float(r["_nearest_sim"]) if not pd.isna(r["_nearest_sim"]) else 0.0
        if k not in canonical and nearest is not None:
            # distorted threshold: high similarity
            if sim >= 0.86:
                types.append("distorted")
                tags.append("ACT_EDITDIST_CLOSE_TO_CANON")
                evid["distorted"] = {"observed": r["_activity_base"], "nearest_canonical": nearest, "similarity": round(sim, 3)}
                # confidence increases with similarity
                confs.append(clamp01(0.55 + 0.45 * (sim - 0.86) / (1.0 - 0.86)))
                desc_parts.append(f"Activity text likely typo of canonical label '{nearest}' (similarity={sim:.3f}).")

        # SYNONYMOUS - token overlap strong but edit similarity not too high (else distorted)
        syn_c = r["_syn_canon"]
        syn_s = float(r["_syn_score"]) if not pd.isna(r["_syn_score"]) else 0.0
        if k not in canonical and syn_c is not None:
            # require meaningful overlap
            if syn_s >= 0.5 and sim < 0.86:
                types.append("synonymous")
                tags.append("ACT_TOKEN_JACCARD_SYNONYM")
                evid["synonymous"] = {"observed": r["_activity_base"], "canonical_candidate": syn_c, "token_jaccard": round(syn_s, 3)}
                confs.append(clamp01(0.45 + 0.5 * (syn_s - 0.5) / 0.5))
                desc_parts.append(f"Activity wording differs but likely same meaning as '{syn_c}' (token overlap={syn_s:.3f}).")

        # COLLATERAL
        if bool(exact_dup.loc[i]):
            types.append("collateral")
            tags.append("DUP_EXACT_CASE_ACT_TS_RES")
            evid.setdefault("collateral", {})
            evid["collateral"]["exact_duplicate_key"] = {
                "case": r["_case"], "activity": r["_activity_raw"], "timestamp": r["Timestamp"], "resource": norm_space(r["_resource"])
            }
            confs.append(0.98)
            desc_parts.append("Exact duplicate event (same case, activity, timestamp, resource).")
        elif rid in near_dup_all:
            types.append("collateral")
            tags.append("DUP_NEAR_WITHIN_3S_SAME_CASE_ACT_RES")
            evid.setdefault("collateral", {})
            evid["collateral"]["near_duplicate_rule"] = {"window_seconds": 3, "case": r["_case"], "activity_base": r["_activity_base"], "resource": norm_space(r["_resource"])}
            confs.append(0.78)
            desc_parts.append("Near-duplicate event within 3 seconds with same case/activity/resource (logging artifact likely).")

        # FORMBASED
        if rid in formbased_rowids:
            types.append("formbased")
            tags.append("SAME_TS_MULTI_EVENTS_IN_CASE")
            evid.setdefault("formbased", {})
            evid["formbased"]["case_timestamp_group_size"] = len(formbased_groups[(r["_case"], r["Timestamp"])])
            evid["formbased"]["case"] = r["_case"]
            evid["formbased"]["timestamp"] = r["Timestamp"]
            confs.append(0.82)
            desc_parts.append("Multiple events in same case share identical timestamp (possible form overwrite).")

        # HOMONYMOUS (heuristic)
        if rid in homonymous_rowids:
            types.append("homonymous")
            tags.append("LABEL_MULTI_CONTEXT_SIGNATURES")
            evid["homonymous"] = homonymous_evidence.get(rid, {})
            confs.append(0.62)
            desc_parts.append("Same label appears in distinct frequent contexts; this row matches minority context (possible homonym).")

        # Finalize
        types_unique = []
        for t in types:
            if t not in types_unique:
                types_unique.append(t)

        error_types.append("|".join(types_unique))
        error_tags.append("|".join(dict.fromkeys(tags)))
        error_evidence.append(json.dumps(evid, ensure_ascii=False))
        error_desc.append(" ".join(desc_parts) if desc_parts else "")
        error_conf.append(round(combine_confidences(confs), 4))

    out["error_flag"] = out["error_types"].ne("") if "error_types" in out.columns else False
    # Fix: we built error_types list separately
    out["error_types"] = error_types
    out["error_flag"] = out["error_types"].ne("")
    out["error_confidence"] = error_conf
    out["error_tags"] = error_tags
    out["error_evidence"] = error_evidence
    out["error_description"] = error_desc

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote: {OUTPUT_PATH}")
    print("Rows flagged:", int(out["error_flag"].sum()), "of", len(out))


if __name__ == "__main__":
    main()