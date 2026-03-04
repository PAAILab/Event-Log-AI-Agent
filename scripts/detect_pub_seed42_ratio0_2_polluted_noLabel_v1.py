#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_polluted_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_polluted_noLabel.detected.csv

Detects (only using Case, Activity, Timestamp, Resource):
- FORMBASED: same-case repeated identical timestamps across multiple events (likely overwritten form time)
- POLLUTED: activity has machine suffix pattern "_<5-12 alnum>_<YYYYMMDD HHMMSSmmm000>"
- DISTORTED: activity close to a canonical label by edit distance / similarity (typo)
- SYNONYMOUS: activity matches a synonym phrase mapped to a canonical label
- COLLATERAL: exact duplicates or near-duplicates within short interval for same case+activity(+resource)
- HOMONYMOUS: same activity label used in two distinct contexts (different predecessor/successor patterns) within same log

Notes:
- Empty Resource is NOT an error by itself (do not flag).
- Script is conservative where ground truth is unknown; confidence reflects evidence strength.
"""

import os
import re
import math
import json
import argparse
from collections import defaultdict, Counter

import pandas as pd
import numpy as np


# -----------------------------
# Utility: string normalization
# -----------------------------
def norm_text(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_key(s: str) -> str:
    """Aggressive normalization for matching: lowercase, collapse spaces, remove surrounding punctuation."""
    s = norm_text(s).lower()
    s = re.sub(r"[^\w\s\-\(\)]", " ", s)  # keep word chars, spaces, hyphen, parentheses
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Utility: similarity (no extra deps)
# -----------------------------
def levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein distance."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    # ensure a is shorter
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        cur = [j]
        for i, ca in enumerate(a, start=1):
            ins = cur[i - 1] + 1
            dele = prev[i] + 1
            sub = prev[i - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def sim_ratio(a: str, b: str) -> float:
    """1 - normalized edit distance."""
    a = norm_key(a)
    b = norm_key(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = levenshtein(a, b)
    return 1.0 - d / max(len(a), len(b))


# -----------------------------
# Canonical label discovery
# -----------------------------
POLLUTED_RE = re.compile(
    r"^(?P<base>.+?)_(?P<token>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6}\d{3}\d{3})$"
)

def split_polluted(activity: str):
    m = POLLUTED_RE.match(norm_text(activity))
    if not m:
        return None
    base = m.group("base").strip()
    token = m.group("token")
    dt = m.group("dt")
    return base, token, dt


def build_canonical_set(df: pd.DataFrame) -> set:
    """
    Build a set of likely canonical activity labels from the dataset itself:
    - take all non-polluted bases
    - plus bases extracted from polluted labels
    - exclude empty
    """
    canon = set()
    for a in df["Activity"].astype(str).fillna(""):
        a0 = norm_text(a)
        if not a0:
            continue
        sp = split_polluted(a0)
        if sp:
            canon.add(norm_text(sp[0]))
        else:
            canon.add(a0)
    return canon


# -----------------------------
# Synonym mining (data-driven + small seed list)
# -----------------------------
SEED_SYNONYMS = {
    # generic business-process synonyms (small, safe)
    "review case": "Review application",
    "assess application": "Review application",
    "evaluate application": "Review application",
    "inspect application": "Review application",
    "deny request": "Reject request",
    "refuse request": "Reject request",
    "decline application": "Reject request",
    "reject application": "Reject request",
    "grant approval": "Approve request",
    "give approval": "Approve request",
    "approve": "Approve request",
}

def build_synonym_map(canon_set: set) -> dict:
    """
    Keep only seed synonyms whose canonical target exists (or close) in this dataset.
    If target not present, we still allow it but with lower confidence later.
    """
    canon_norm = {norm_key(c): c for c in canon_set}
    syn_map = {}
    for syn, target in SEED_SYNONYMS.items():
        # if exact target exists, keep; else try fuzzy match to nearest canonical
        if target in canon_set:
            syn_map[norm_key(syn)] = target
        else:
            # find nearest canonical to target
            best = None
            best_sim = 0.0
            for c in canon_set:
                s = sim_ratio(target, c)
                if s > best_sim:
                    best_sim = s
                    best = c
            if best is not None and best_sim >= 0.88:
                syn_map[norm_key(syn)] = best
            else:
                syn_map[norm_key(syn)] = target  # keep but will be lower confidence
    return syn_map


# -----------------------------
# Detection rules
# -----------------------------
def detect_polluted(df):
    flags = np.zeros(len(df), dtype=bool)
    base_label = [None] * len(df)
    evidence = [None] * len(df)
    for i, a in enumerate(df["Activity"].astype(str).fillna("")):
        a0 = norm_text(a)
        sp = split_polluted(a0)
        if sp:
            flags[i] = True
            base_label[i] = sp[0]
            evidence[i] = f"activity='{a0}' matches polluted pattern; base='{sp[0]}', token='{sp[1]}', suffix_dt='{sp[2]}'"
    return flags, base_label, evidence


def detect_formbased(df):
    """
    Same Case + same Timestamp repeated across multiple rows (>=3) is strong evidence.
    If repeated exactly twice, still possible but weaker.
    """
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    key = list(zip(df["Case"].astype(str), ts.astype("datetime64[ns]")))
    counts = Counter(key)
    flags = np.zeros(len(df), dtype=bool)
    conf = np.zeros(len(df), dtype=float)
    evidence = [None] * len(df)

    for i, k in enumerate(key):
        c = counts.get(k, 0)
        if pd.isna(k[1]):
            continue
        if c >= 3:
            flags[i] = True
            conf[i] = 0.90
            evidence[i] = f"case={k[0]} has {c} events with identical timestamp={k[1]}"
        elif c == 2:
            flags[i] = True
            conf[i] = 0.65
            evidence[i] = f"case={k[0]} has 2 events with identical timestamp={k[1]} (possible form overwrite)"
    return flags, conf, evidence


def detect_collateral(df):
    """
    - Exact duplicates: same Case, Activity, Timestamp, Resource repeated -> very strong
    - Near duplicates: same Case, Activity, Resource within <=2 seconds -> strong
      (Resource may be empty; still use it as attribute, but emptiness is allowed)
    """
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    case = df["Case"].astype(str)
    act = df["Activity"].astype(str).fillna("").map(norm_text)
    res = df["Resource"].astype(str).fillna("").map(norm_text)

    exact_key = list(zip(case, act, ts.astype("datetime64[ns]"), res))
    exact_counts = Counter(exact_key)

    flags = np.zeros(len(df), dtype=bool)
    conf = np.zeros(len(df), dtype=float)
    evidence = [None] * len(df)

    # exact duplicates
    for i, k in enumerate(exact_key):
        if pd.isna(k[2]):
            continue
        c = exact_counts.get(k, 0)
        if c >= 2:
            flags[i] = True
            conf[i] = max(conf[i], 0.95)
            evidence[i] = f"exact duplicate: (case,activity,timestamp,resource) occurs {c} times: {k}"

    # near duplicates within case+activity+resource
    df_tmp = pd.DataFrame({
        "idx": np.arange(len(df)),
        "Case": case.values,
        "Activity": act.values,
        "Resource": res.values,
        "Timestamp": ts.values
    })
    df_tmp = df_tmp.dropna(subset=["Timestamp"]).sort_values(["Case", "Activity", "Resource", "Timestamp"])

    # compute time deltas to previous within group
    df_tmp["prev_ts"] = df_tmp.groupby(["Case", "Activity", "Resource"])["Timestamp"].shift(1)
    df_tmp["dt_sec"] = (df_tmp["Timestamp"] - df_tmp["prev_ts"]).dt.total_seconds()

    near = df_tmp[(df_tmp["dt_sec"].notna()) & (df_tmp["dt_sec"] >= 0) & (df_tmp["dt_sec"] <= 2.0)]
    for _, r in near.iterrows():
        i = int(r["idx"])
        flags[i] = True
        # if already exact dup, keep higher
        conf[i] = max(conf[i], 0.85 if conf[i] < 0.95 else conf[i])
        ev = f"near-duplicate within {r['dt_sec']:.3f}s for same case+activity+resource; prev_ts={r['prev_ts']}, ts={r['Timestamp']}"
        evidence[i] = evidence[i] + " | " + ev if evidence[i] else ev

    return flags, conf, evidence


def detect_synonymous(df, syn_map):
    flags = np.zeros(len(df), dtype=bool)
    conf = np.zeros(len(df), dtype=float)
    canon = [None] * len(df)
    evidence = [None] * len(df)

    for i, a in enumerate(df["Activity"].astype(str).fillna("")):
        a0 = norm_text(a)
        # if polluted, compare base part too (handled later in multi-error merge)
        k = norm_key(a0)
        if k in syn_map:
            flags[i] = True
            canon[i] = syn_map[k]
            # confidence: higher if canonical appears in dataset frequently
            conf[i] = 0.80
            evidence[i] = f"activity='{a0}' matches synonym phrase -> canonical='{canon[i]}'"
    return flags, conf, canon, evidence


def detect_distorted(df, canonical_set, polluted_base=None):
    """
    Distorted if activity is close to a canonical label but not equal.
    If polluted_base provided, use base for similarity check (to allow POLLUTED+DISTORTED).
    """
    canon_list = list(canonical_set)
    canon_norm = {norm_key(c): c for c in canon_list}

    flags = np.zeros(len(df), dtype=bool)
    conf = np.zeros(len(df), dtype=float)
    canon = [None] * len(df)
    evidence = [None] * len(df)

    for i, a in enumerate(df["Activity"].astype(str).fillna("")):
        raw = norm_text(a)
        base = polluted_base[i] if polluted_base is not None and polluted_base[i] else raw
        if not base:
            continue

        nk = norm_key(base)
        if nk in canon_norm:
            # exact canonical -> not distorted
            continue

        # find best match
        best_c = None
        best_s = 0.0
        for c in canon_list:
            s = sim_ratio(base, c)
            if s > best_s:
                best_s = s
                best_c = c

        # thresholds: aggressive but avoid false positives on short strings
        L = max(len(norm_key(base)), 1)
        if (L >= 8 and best_s >= 0.88) or (L >= 14 and best_s >= 0.84):
            flags[i] = True
            canon[i] = best_c
            # confidence scales with similarity
            conf[i] = float(min(0.92, max(0.60, (best_s - 0.80) / 0.12 * 0.32 + 0.60)))
            evidence[i] = f"base_activity='{base}' not in canonical set; closest='{best_c}' sim={best_s:.3f}"
    return flags, conf, canon, evidence


def detect_homonymous(df):
    """
    Heuristic: same Activity label appears with two (or more) distinct context signatures:
    (prev_activity, next_activity) within same case ordering.
    If an activity has high context entropy and at least two dominant, disjoint contexts,
    flag those occurrences with moderate confidence.
    """
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    tmp = df.copy()
    tmp["__ts"] = ts
    tmp["__act"] = df["Activity"].astype(str).fillna("").map(norm_text)
    tmp["__case"] = df["Case"].astype(str)

    tmp = tmp.dropna(subset=["__ts"]).sort_values(["__case", "__ts", "__act"], kind="mergesort")
    tmp["__prev"] = tmp.groupby("__case")["__act"].shift(1).fillna("<START>")
    tmp["__next"] = tmp.groupby("__case")["__act"].shift(-1).fillna("<END>")

    # context signature
    tmp["__ctx"] = list(zip(tmp["__prev"], tmp["__next"]))

    # build per-activity context distribution
    act_ctx_counts = defaultdict(Counter)
    for a, ctx in zip(tmp["__act"], tmp["__ctx"]):
        act_ctx_counts[a][ctx] += 1

    # decide which activities are homonymous candidates
    homonymous_acts = set()
    for a, ctr in act_ctx_counts.items():
        total = sum(ctr.values())
        if total < 30:
            continue  # need enough evidence
        top = ctr.most_common(3)
        if len(top) < 2:
            continue
        (ctx1, c1), (ctx2, c2) = top[0], top[1]
        p1, p2 = c1 / total, c2 / total
        # if two strong contexts and they differ materially
        if p1 >= 0.25 and p2 >= 0.20 and ctx1 != ctx2:
            homonymous_acts.add(a)

    # flag rows in tmp for those activities where context is one of the dominant ones
    flags = np.zeros(len(df), dtype=bool)
    conf = np.zeros(len(df), dtype=float)
    evidence = [None] * len(df)

    # map back by original index
    for idx, a, prev_a, next_a in zip(tmp.index, tmp["__act"], tmp["__prev"], tmp["__next"]):
        if a in homonymous_acts:
            flags[idx] = True
            conf[idx] = 0.60
            evidence[idx] = f"activity='{a}' shows multiple dominant contexts; this occurrence context=(prev='{prev_a}', next='{next_a}')"
    return flags, conf, evidence


# -----------------------------
# Merge detections
# -----------------------------
def add_error(row, etype, conf, tag, ev, desc, store):
    store["types"].add(etype)
    store["tags"].add(tag)
    if ev:
        store["evidence"].append(ev)
    if desc:
        store["descriptions"].append(desc)
    store["conf"] = max(store["conf"], float(conf))


def main():
    in_path = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_polluted_noLabel.csv"
    out_path = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_polluted_noLabel.detected.csv"

    df = pd.read_csv(in_path)

    # Validate required columns only
    required = ["Case", "Activity", "Timestamp", "Resource"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")

    n = len(df)
    stores = [{
        "types": set(),
        "tags": set(),
        "evidence": [],
        "descriptions": [],
        "conf": 0.0
    } for _ in range(n)]

    # Canonical + synonym resources
    canonical_set = build_canonical_set(df)
    syn_map = build_synonym_map(canonical_set)

    # POLLUTED
    pol_flags, pol_base, pol_ev = detect_polluted(df)
    for i in np.where(pol_flags)[0]:
        add_error(
            i, "polluted", 0.95, "ACT_POLLUTED_REGEX",
            pol_ev[i],
            f"Activity contains machine-generated suffix; base label likely '{pol_base[i]}'.",
            stores[i]
        )

    # FORMBASED
    fb_flags, fb_conf, fb_ev = detect_formbased(df)
    for i in np.where(fb_flags)[0]:
        add_error(
            i, "formbased", fb_conf[i], "TS_SAME_WITHIN_CASE",
            fb_ev[i],
            "Multiple events in the same case share an identical timestamp (possible form overwrite).",
            stores[i]
        )

    # COLLATERAL
    col_flags, col_conf, col_ev = detect_collateral(df)
    for i in np.where(col_flags)[0]:
        add_error(
            i, "collateral", col_conf[i], "DUP_EXACT_OR_NEAR",
            col_ev[i],
            "Duplicate or near-duplicate event instance detected (logging artifact).",
            stores[i]
        )

    # SYNONYMOUS (on raw activity)
    syn_flags, syn_conf, syn_canon, syn_ev = detect_synonymous(df, syn_map)
    for i in np.where(syn_flags)[0]:
        add_error(
            i, "synonymous", syn_conf[i], "ACT_SYNONYM_MAP",
            syn_ev[i],
            f"Activity wording suggests synonym of canonical label '{syn_canon[i]}'.",
            stores[i]
        )

    # DISTORTED (use polluted base if present)
    dis_flags, dis_conf, dis_canon, dis_ev = detect_distorted(df, canonical_set, polluted_base=pol_base)
    for i in np.where(dis_flags)[0]:
        add_error(
            i, "distorted", dis_conf[i], "ACT_FUZZY_CANONICAL",
            dis_ev[i],
            f"Activity appears misspelled/typo; closest canonical label '{dis_canon[i]}'.",
            stores[i]
        )

    # HOMONYMOUS
    hom_flags, hom_conf, hom_ev = detect_homonymous(df)
    for i in np.where(hom_flags)[0]:
        add_error(
            i, "homonymous", hom_conf[i], "CTX_DIVERGENT_PREV_NEXT",
            hom_ev[i],
            "Same activity label appears to be used in multiple distinct process contexts (possible different meanings).",
            stores[i]
        )

    # Build output
    out = pd.DataFrame({
        "row_id": np.arange(n, dtype=int),
        "error_flag": [len(s["types"]) > 0 for s in stores],
        "error_types": ["|".join(sorted(s["types"])) if s["types"] else "" for s in stores],
        "error_confidence": [round(float(s["conf"]), 4) for s in stores],
        "error_tags": ["|".join(sorted(s["tags"])) if s["tags"] else "" for s in stores],
        "error_evidence": [(" || ".join(s["evidence"])) if s["evidence"] else "" for s in stores],
        "error_description": [(" ".join(s["descriptions"])) if s["descriptions"] else "" for s in stores],
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()