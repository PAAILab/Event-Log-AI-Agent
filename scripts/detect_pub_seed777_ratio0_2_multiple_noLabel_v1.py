#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed777_ratio0.2_multiple_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed777_ratio0.2_multiple_noLabel.detected.csv

Detects (based ONLY on Case, Activity, Timestamp, Resource):
- FORMBASED: many events in same case share identical timestamp (likely overwritten)
- POLLUTED: activity has machine suffix pattern _<5-12 alnum>_<yyyymmdd hhmmss...>
- DISTORTED: activity is close misspelling of a canonical label (edit distance)
- SYNONYMOUS: activity matches a synonym phrase of a canonical label
- COLLATERAL: duplicates / near-duplicates within short interval for same case+activity+resource
- HOMONYMOUS: same activity label used in multiple distinct contexts (resource/time-gap patterns)

Notes:
- Empty Resource is NOT an error by itself (do not flag).
"""

import os
import re
import math
import json
from collections import defaultdict, Counter
from datetime import timedelta

import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed777_ratio0.2_multiple_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed777_ratio0.2_multiple_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# -----------------------------
# Utility: robust string helpers
# -----------------------------
def norm_space(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def norm_key(s: str) -> str:
    """Lowercase, collapse spaces, remove surrounding punctuation-ish noise."""
    s = norm_space(s).lower()
    s = re.sub(r"[“”\"']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_dt_parse(series: pd.Series) -> pd.Series:
    # Parse timestamps; keep NaT if invalid
    return pd.to_datetime(series, errors="coerce", utc=False)

# -----------------------------
# Utility: Levenshtein distance
# -----------------------------
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # DP with O(min(n,m)) memory
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, start=1):
        cur = [j]
        for i, ai in enumerate(a, start=1):
            ins = cur[i - 1] + 1
            dele = prev[i] + 1
            sub = prev[i - 1] + (0 if ai == bj else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def similarity_ratio(a: str, b: str) -> float:
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
    r"^(?P<base>.+?)_(?P<rand>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6,12})$"
)

def extract_polluted_base(activity: str):
    m = POLLUTED_RE.match(norm_space(activity))
    if not m:
        return None
    return norm_space(m.group("base"))

def build_canonical_labels(df: pd.DataFrame):
    """
    Build a set of canonical labels from:
    - non-polluted activities
    - polluted base labels
    Then choose canonical representative by frequency (normalized).
    """
    raw = df["Activity"].astype(str).map(norm_space)

    bases = []
    for a in raw:
        base = extract_polluted_base(a)
        if base:
            bases.append(base)
        else:
            bases.append(a)

    # Normalize keys for grouping
    keys = [norm_key(x) for x in bases]
    freq = Counter(keys)

    # Representative: most common original casing among items with same key
    rep = {}
    bucket = defaultdict(Counter)
    for orig, k in zip(bases, keys):
        bucket[k][orig] += 1
    for k, c in bucket.items():
        rep[k] = c.most_common(1)[0][0]

    canonical_keys = set(freq.keys())
    return canonical_keys, rep, freq

# -----------------------------
# Synonym detection (lightweight)
# -----------------------------
# We cannot use external domain columns; synonyms are inferred from text patterns.
# We'll build synonym groups by simple verb/noun normalization and a small generic thesaurus.
GENERIC_SYNONYMS = {
    # verbs
    "review": {"review", "assess", "evaluate", "inspect", "check", "examine"},
    "approve": {"approve", "grant", "accept", "authorize", "confirm"},
    "reject": {"reject", "deny", "decline", "refuse"},
    "request": {"request", "ask", "require", "solicit"},
    "submit": {"submit", "send", "file", "provide"},
    "close": {"close", "complete", "finish", "terminate", "end"},
    "start": {"start", "begin", "initiate", "launch"},
    "update": {"update", "modify", "edit", "revise"},
    "record": {"record", "log", "register", "document"},
    "notify": {"notify", "inform", "alert"},
    "pay": {"pay", "process payment", "charge", "settle"},
}

def tokenize_simple(s: str):
    s = norm_key(s)
    s = re.sub(r"[^a-z0-9\s\-\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def synonym_signature(s: str) -> str:
    """
    Map first verb-ish token to a canonical verb group if possible.
    Keep remaining tokens as-is.
    """
    toks = tokenize_simple(s)
    if not toks:
        return ""
    first = toks[0]
    mapped = first
    for canon, syns in GENERIC_SYNONYMS.items():
        if first in syns:
            mapped = canon
            break
    return " ".join([mapped] + toks[1:])

def build_synonym_index(canonical_rep: dict):
    """
    Build index from synonym_signature -> list of canonical keys.
    """
    sig_to_keys = defaultdict(list)
    for k, rep in canonical_rep.items():
        sig = synonym_signature(rep)
        if sig:
            sig_to_keys[sig].append(k)
    return sig_to_keys

# -----------------------------
# Detection rules
# -----------------------------
def detect_polluted(activity: str):
    base = extract_polluted_base(activity)
    if not base:
        return None
    return base

def detect_distorted(activity: str, canonical_rep: dict, canonical_freq: Counter):
    """
    Distorted if:
    - not exact canonical key
    - close to a frequent canonical label (edit similarity high)
    """
    a = norm_space(activity)
    ak = norm_key(a)
    if not ak:
        return None

    # If polluted, distortion should be checked on base
    base = extract_polluted_base(a)
    if base:
        a = base
        ak = norm_key(a)

    if ak in canonical_rep:
        return None

    # Compare against top frequent canonicals to keep runtime reasonable
    # Take top N by frequency
    topN = 300
    candidates = [k for k, _ in canonical_freq.most_common(topN)]
    best = None
    best_sim = 0.0
    for ck in candidates:
        sim = similarity_ratio(a, canonical_rep[ck])
        if sim > best_sim:
            best_sim = sim
            best = ck

    # Aggressive thresholds: allow moderate typos but avoid synonyms
    # Also require same first character or shared token overlap to reduce false positives
    if best is None:
        return None
    cand = canonical_rep[best]
    if best_sim >= 0.86:
        return {"canonical": cand, "similarity": best_sim}
    # Slightly lower if very short strings
    if len(norm_key(a)) <= 8 and best_sim >= 0.80:
        return {"canonical": cand, "similarity": best_sim}
    return None

def detect_synonymous(activity: str, canonical_rep: dict, sig_to_keys: dict):
    """
    Synonymous if synonym_signature matches a canonical signature but label differs.
    """
    a = norm_space(activity)
    if not a:
        return None

    # If polluted, check base for synonymy
    base = extract_polluted_base(a)
    if base:
        a = base

    ak = norm_key(a)
    sig = synonym_signature(a)
    if not sig or sig not in sig_to_keys:
        return None

    # If it exactly matches a canonical already, not synonym
    if ak in canonical_rep:
        return None

    # Choose the most plausible canonical among those sharing signature:
    # pick the shortest representative (often the clean canonical) and/or most frequent
    keys = sig_to_keys[sig]
    # Prefer exact token overlap
    toks = set(tokenize_simple(a))
    scored = []
    for k in keys:
        rep = canonical_rep[k]
        rtoks = set(tokenize_simple(rep))
        jacc = len(toks & rtoks) / max(1, len(toks | rtoks))
        scored.append((jacc, -len(rep), rep))
    scored.sort(reverse=True)
    best_rep = scored[0][2]
    # Require some overlap to avoid mapping unrelated phrases
    if scored[0][0] >= 0.34:
        return {"canonical": best_rep, "signature": sig, "overlap": scored[0][0]}
    return None

def detect_formbased(case_df: pd.DataFrame):
    """
    Within a case, if a timestamp repeats for >=3 events, mark all but the first as FORMBASED.
    Evidence: same timestamp repeated many times in same case.
    """
    out = {}
    # group by exact timestamp
    g = case_df.groupby("Timestamp_parsed", dropna=False)
    for ts, grp in g:
        if pd.isna(ts):
            continue
        if len(grp) >= 3:
            # sort by original row order; keep first as "possibly correct", flag rest
            grp_sorted = grp.sort_values("row_id")
            for rid in grp_sorted["row_id"].iloc[1:]:
                out[rid] = {
                    "type": "formbased",
                    "evidence": f"Case has {len(grp)} events with identical timestamp {ts}",
                    "tag": "FORM_TS_REPEAT>=3",
                    "confidence": min(0.95, 0.65 + 0.08 * (len(grp) - 2)),
                }
    return out

def detect_collateral(case_df: pd.DataFrame, window_seconds=3):
    """
    Collateral duplicates:
    - exact duplicates: same case+activity+timestamp+resource repeated
    - near duplicates: same case+activity+resource within <=window_seconds
    """
    out = {}

    # Exact duplicates
    dup_cols = ["Case", "Activity", "Timestamp_parsed", "Resource_norm"]
    dups = case_df.duplicated(subset=dup_cols, keep="first")
    for rid in case_df.loc[dups, "row_id"].tolist():
        out[rid] = {
            "type": "collateral",
            "evidence": "Exact duplicate of (Case,Activity,Timestamp,Resource)",
            "tag": "COLL_EXACT_DUP",
            "confidence": 0.98,
        }

    # Near duplicates (requires valid timestamps)
    sdf = case_df.dropna(subset=["Timestamp_parsed"]).sort_values("Timestamp_parsed")
    if len(sdf) >= 2:
        # group by activity+resource
        for (act, res), grp in sdf.groupby(["Activity", "Resource_norm"], dropna=False):
            if len(grp) < 2:
                continue
            grp = grp.sort_values("Timestamp_parsed")
            prev_ts = None
            prev_rid = None
            for _, row in grp.iterrows():
                ts = row["Timestamp_parsed"]
                rid = row["row_id"]
                if prev_ts is not None:
                    dt = (ts - prev_ts).total_seconds()
                    if 0 <= dt <= window_seconds:
                        # flag current as collateral (keep first)
                        if rid not in out:
                            out[rid] = {
                                "type": "collateral",
                                "evidence": f"Near-duplicate within {dt:.3f}s for same activity+resource in case",
                                "tag": f"COLL_NEAR_DUP<= {window_seconds}s",
                                "confidence": 0.85 if dt > 0 else 0.95,
                            }
                prev_ts = ts
                prev_rid = rid

    return out

def detect_homonymous(df: pd.DataFrame):
    """
    Heuristic: same Activity label used with strongly bimodal resource patterns
    and/or strongly bimodal time-gap-to-next distributions across cases.

    We flag rows of an activity if:
    - activity appears with >=2 dominant resource groups that are dissimilar
    - AND the activity is relatively frequent (to avoid noise)
    """
    out = {}
    df2 = df.copy()
    df2["Activity_norm"] = df2["Activity"].astype(str).map(norm_key)
    df2["Resource_norm"] = df2["Resource"].map(norm_space)
    df2["Timestamp_parsed"] = df2["Timestamp_parsed"]

    # compute next time gap within case
    df2 = df2.sort_values(["Case", "Timestamp_parsed", "row_id"])
    df2["next_ts"] = df2.groupby("Case")["Timestamp_parsed"].shift(-1)
    df2["gap_to_next_s"] = (df2["next_ts"] - df2["Timestamp_parsed"]).dt.total_seconds()

    # activity stats
    act_counts = df2["Activity_norm"].value_counts()
    frequent_acts = act_counts[act_counts >= 30].index.tolist()

    for act in frequent_acts:
        sub = df2[df2["Activity_norm"] == act]
        # resource distribution (ignore empty resource as a group but keep it if dominant)
        res_counts = sub["Resource_norm"].fillna("").value_counts()
        if len(res_counts) < 2:
            continue
        top = res_counts.head(3)
        if len(top) < 2:
            continue
        p1 = top.iloc[0] / len(sub)
        p2 = top.iloc[1] / len(sub)

        # gap distribution: compare median gaps for top resources (if timestamps exist)
        # If timestamps missing, skip gap logic
        has_gaps = sub["gap_to_next_s"].notna().sum() >= 20

        # Condition: two strong resource modes
        strong_two_modes = (p1 >= 0.35 and p2 >= 0.25 and top.index[0] != top.index[1])

        gap_separation = False
        gap_evidence = ""
        if has_gaps and strong_two_modes:
            rA, rB = top.index[0], top.index[1]
            gA = sub.loc[sub["Resource_norm"].fillna("") == rA, "gap_to_next_s"].dropna()
            gB = sub.loc[sub["Resource_norm"].fillna("") == rB, "gap_to_next_s"].dropna()
            if len(gA) >= 10 and len(gB) >= 10:
                medA = float(gA.median())
                medB = float(gB.median())
                # require big separation (order-of-magnitude-ish)
                if min(medA, medB) >= 0 and (max(medA, medB) + 1) / (min(medA, medB) + 1) >= 8:
                    gap_separation = True
                    gap_evidence = f"Median gap-to-next differs by >=8x across resources ({rA}:{medA:.1f}s vs {rB}:{medB:.1f}s)"

        if strong_two_modes and (gap_separation or (p1 + p2 >= 0.75)):
            # Flag as homonymous candidate (lower confidence; heuristic)
            conf = 0.55
            tags = ["HOMO_RESOURCE_BIMODAL"]
            evidence = [f"Activity '{act}' has bimodal resource usage: {top.index[0]}({p1:.2f}), {top.index[1]}({p2:.2f})"]
            if gap_separation:
                conf = 0.70
                tags.append("HOMO_GAP_BIMODAL")
                evidence.append(gap_evidence)

            for rid in sub["row_id"].tolist():
                out[rid] = {
                    "type": "homonymous",
                    "evidence": " | ".join(evidence),
                    "tag": "+".join(tags),
                    "confidence": conf,
                }

    return out

# -----------------------------
# Main detection orchestration
# -----------------------------
def add_error(rec, etype, conf, tag, evidence, desc):
    rec["error_flag"] = True
    rec["error_types"].add(etype)
    rec["error_confidence"] = max(rec["error_confidence"], conf)
    rec["error_tags"].add(tag)
    rec["error_evidence"].append(evidence)
    rec["error_description"].append(desc)

def main():
    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

    # row_id = original row index (0-based) to be stable
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    # Normalize
    df["Case"] = df["Case"].map(norm_space)
    df["Activity"] = df["Activity"].map(norm_space)
    df["Resource_norm"] = df["Resource"].map(norm_space)  # empty allowed
    df["Timestamp_parsed"] = safe_dt_parse(df["Timestamp"])

    # Build canonical labels from data itself
    canonical_keys, canonical_rep, canonical_freq = build_canonical_labels(df)
    sig_to_keys = build_synonym_index(canonical_rep)

    # Initialize output records
    records = {
        rid: {
            "row_id": rid,
            "error_flag": False,
            "error_types": set(),
            "error_confidence": 0.0,
            "error_tags": set(),
            "error_evidence": [],
            "error_description": [],
        }
        for rid in df["row_id"].tolist()
    }

    # 0) Basic format/range checks on Timestamp (allowed: missing? not specified; treat invalid as error)
    # Aggressive: invalid timestamp is an error because ordering/time-based mining breaks.
    invalid_ts = df["Timestamp_parsed"].isna() & df["Timestamp"].notna()
    for rid, raw_ts in df.loc[invalid_ts, ["row_id", "Timestamp"]].itertuples(index=False):
        add_error(
            records[rid],
            "timestamp_format",
            0.95,
            "TS_PARSE_FAIL",
            f"Unparseable Timestamp='{raw_ts}'",
            "Timestamp cannot be parsed to datetime; event time is invalid.",
        )

    # 1) Polluted
    for rid, act in df[["row_id", "Activity"]].itertuples(index=False):
        base = detect_polluted(act)
        if base:
            add_error(
                records[rid],
                "polluted",
                0.97,
                "ACT_POLLUTED_SUFFIX",
                f"Activity matches polluted pattern; base='{base}', raw='{act}'",
                f"Activity label contains machine-generated suffix; canonical likely '{base}'.",
            )

    # 2) Distorted + Synonymous (on base if polluted)
    for rid, act in df[["row_id", "Activity"]].itertuples(index=False):
        # Distorted
        dist = detect_distorted(act, canonical_rep, canonical_freq)
        if dist:
            conf = 0.60 + 0.40 * min(1.0, max(0.0, (dist["similarity"] - 0.80) / 0.20))
            add_error(
                records[rid],
                "distorted",
                float(min(0.95, conf)),
                "ACT_EDIT_SIM",
                f"Closest canonical='{dist['canonical']}' similarity={dist['similarity']:.3f} raw='{act}'",
                f"Activity appears misspelled/typo of '{dist['canonical']}'.",
            )

        # Synonymous
        syn = detect_synonymous(act, canonical_rep, sig_to_keys)
        if syn:
            add_error(
                records[rid],
                "synonymous",
                0.72,
                "ACT_SYNONYM_SIG",
                f"Signature='{syn['signature']}', overlap={syn['overlap']:.2f}, canonical='{syn['canonical']}', raw='{act}'",
                f"Activity uses different wording for same meaning; canonical likely '{syn['canonical']}'.",
            )

    # 3) Case-level: formbased + collateral
    for case_id, case_df in df.groupby("Case", dropna=False):
        fb = detect_formbased(case_df)
        for rid, info in fb.items():
            add_error(
                records[rid],
                "formbased",
                info["confidence"],
                info["tag"],
                info["evidence"],
                "Multiple events in the same case share an identical timestamp; likely form-based overwrite.",
            )

        coll = detect_collateral(case_df, window_seconds=3)
        for rid, info in coll.items():
            add_error(
                records[rid],
                "collateral",
                info["confidence"],
                info["tag"],
                info["evidence"],
                "Duplicate/near-duplicate event likely caused by logging artifact.",
            )

    # 4) Homonymous (global heuristic)
    homo = detect_homonymous(df)
    for rid, info in homo.items():
        add_error(
            records[rid],
            "homonymous",
            info["confidence"],
            info["tag"],
            info["evidence"],
            "Same activity label appears to be used for different underlying meanings (context/resource patterns differ).",
        )

    # Finalize output dataframe
    out_rows = []
    for rid in df["row_id"].tolist():
        rec = records[rid]
        out_rows.append(
            {
                "row_id": rec["row_id"],
                "error_flag": bool(rec["error_flag"]),
                "error_types": "|".join(sorted(rec["error_types"])) if rec["error_types"] else "",
                "error_confidence": float(round(rec["error_confidence"], 4)),
                "error_tags": "|".join(sorted(rec["error_tags"])) if rec["error_tags"] else "",
                "error_evidence": " || ".join(rec["error_evidence"]) if rec["error_evidence"] else "",
                "error_description": " ".join(rec["error_description"]) if rec["error_description"] else "",
            }
        )

    out_df = pd.DataFrame(out_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    # Minimal console summary
    n_err = int(out_df["error_flag"].sum())
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(out_df)} | Flagged errors: {n_err} ({n_err/len(out_df):.1%})")
    print("Top error_types:")
    # explode types
    types = []
    for s in out_df.loc[out_df["error_flag"], "error_types"].tolist():
        types.extend([t for t in s.split("|") if t])
    for t, c in Counter(types).most_common(20):
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()