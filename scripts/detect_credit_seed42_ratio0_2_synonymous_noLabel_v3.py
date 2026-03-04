#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection (Credit dataset)

Reads:
  /home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_synonymous_noLabel.csv

Writes:
  /home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_synonymous_noLabel.detected.csv

Detects (ONLY using Case, Activity, Timestamp, Resource):
  - polluted: machine suffix patterns appended to activity labels
  - distorted: typos/spacing/character issues vs canonical labels (derived from data)
  - synonymous: different surface forms mapping to same canonical label (derived from data)
  - formbased: repeated timestamps within a case suggesting overwritten times
  - collateral: exact duplicates and near-duplicates within short interval
  - homonymous: same label used in clearly different contexts (resource/prev/next) suggesting multiple meanings

Notes:
  - Empty Resource is NOT an error.
  - Script is intentionally aggressive (real logs have 5-20% errors).
"""

import os
import re
import math
import json
import unicodedata
from collections import defaultdict, Counter

import pandas as pd
import numpy as np


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_synonymous_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_synonymous_noLabel.detected.csv"

CASE_COL = "Case"
ACT_COL = "Activity"
TS_COL = "Timestamp"
RES_COL = "Resource"


# -----------------------------
# Helpers
# -----------------------------
def norm_ws(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def safe_lower(s: str) -> str:
    return norm_ws(s).lower()

def parse_ts(series: pd.Series) -> pd.Series:
    # robust parsing; keep NaT if invalid
    return pd.to_datetime(series, errors="coerce", utc=False, infer_datetime_format=True)

def jaccard_tokens(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", safe_lower(a)))
    tb = set(re.findall(r"[a-z0-9]+", safe_lower(b)))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def char_similarity(a: str, b: str) -> float:
    # lightweight similarity without external deps
    a = safe_lower(a)
    b = safe_lower(b)
    if a == b:
        return 1.0
    # normalized LCS-ish via SequenceMatcher
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()

def is_polluted(activity: str):
    """
    Polluted pattern: base + '_' + 5-12 alnum + '_' + yyyymmdd hhmmss + optional digits
    Examples: "Request Info_47xiDPl_20230929 130852312000"
    """
    s = norm_ws(activity)
    # allow multiple underscores; capture base
    m = re.match(r"^(?P<base>.+?)_([A-Za-z0-9]{5,12})_(\d{8}\s+\d{6}\d{0,6})$", s)
    if m:
        return True, m.group("base")
    # also catch suffix like _Ab12CdE_20240208 191921970000 with extra spaces
    m2 = re.match(r"^(?P<base>.+?)_([A-Za-z0-9]{5,12})_(\d{8}\s+\d{6}\d+)$", s)
    if m2:
        return True, m2.group("base")
    return False, None

def build_canonical_labels(df: pd.DataFrame):
    """
    Build canonical label set from data itself:
      - If polluted, canonical is extracted base.
      - Else canonical candidates are the most frequent "normalized" forms.
    """
    activities = df[ACT_COL].astype(str).map(norm_ws)

    base_from_polluted = {}
    for idx, act in activities.items():
        ok, base = is_polluted(act)
        if ok and base:
            base_from_polluted[idx] = norm_ws(base)

    # normalized form for frequency: lowercase + collapse spaces + remove punctuation noise
    def norm_for_freq(s):
        s = safe_lower(s)
        s = re.sub(r"[_\-]+", " ", s)
        s = re.sub(r"[^\w\s\(\)]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    normed = activities.map(norm_for_freq)
    freq = Counter(normed)

    # canonical pool: top frequent normalized labels (aggressive but bounded)
    # keep labels with at least 2 occurrences OR in top 200
    most_common = [k for k, v in freq.most_common(300)]
    canon_pool = set([k for k, v in freq.items() if v >= 2] + most_common)

    # also add polluted bases (normalized)
    for base in base_from_polluted.values():
        canon_pool.add(norm_for_freq(base))

    # map each row to a canonical guess:
    # - polluted -> base
    # - else -> nearest canonical by similarity (token+jaro-ish)
    canon_list = sorted(canon_pool, key=lambda x: (-freq.get(x, 0), x))

    def nearest_canon(act_raw: str):
        actn = norm_for_freq(act_raw)
        if actn in canon_pool:
            return actn, 1.0
        # search top N only for speed
        best = None
        best_score = -1.0
        for c in canon_list[:200]:
            # combine token and char similarity
            score = 0.55 * jaccard_tokens(actn, c) + 0.45 * char_similarity(actn, c)
            if score > best_score:
                best_score = score
                best = c
        return best, float(best_score)

    return base_from_polluted, canon_pool, nearest_canon


def add_error(row_errors, rid, etype, conf, tags, evidence, desc):
    rec = row_errors[rid]
    rec["types"].add(etype)
    rec["conf"].append(conf)
    rec["tags"].update(tags if isinstance(tags, (list, set, tuple)) else [tags])
    rec["evidence"].append(evidence)
    rec["desc"].append(desc)


# -----------------------------
# Main detection
# -----------------------------
def main():
    df = pd.read_csv(INPUT_PATH)

    # Validate required columns only
    missing = [c for c in [CASE_COL, ACT_COL, TS_COL, RES_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

    # row_id is original row index
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    # normalize fields
    df["_case"] = df[CASE_COL].astype(str).map(norm_ws)
    df["_act_raw"] = df[ACT_COL].astype(str).map(norm_ws)
    df["_act_low"] = df["_act_raw"].map(safe_lower)
    df["_res"] = df[RES_COL].astype(object)
    df["_res_norm"] = df["_res"].astype(str).map(norm_ws)
    # treat empty/NaN resource as empty string (NOT an error)
    df.loc[df[RES_COL].isna(), "_res_norm"] = ""
    df.loc[df["_res_norm"].isin(["nan", "none", "null"]), "_res_norm"] = ""

    df["_ts"] = parse_ts(df[TS_COL])

    row_errors = defaultdict(lambda: {"types": set(), "conf": [], "tags": set(), "evidence": [], "desc": []})

    # 0) Basic timestamp parse errors (allowed: detect invalid timestamp as error)
    # (Not listed explicitly, but it's a real log error; still only uses Timestamp column.)
    bad_ts = df["_ts"].isna()
    for rid in df.loc[bad_ts, "row_id"].tolist():
        add_error(
            row_errors, rid,
            "timestamp_invalid",
            0.95,
            ["TS_PARSE_FAIL"],
            {"timestamp": str(df.loc[rid, TS_COL])},
            "Timestamp cannot be parsed into a valid datetime."
        )

    # Build canonical labels and nearest matcher from data
    polluted_base_by_idx, canon_pool, nearest_canon = build_canonical_labels(df)

    # 1) Polluted detection
    for i, act in df["_act_raw"].items():
        ok, base = is_polluted(act)
        if ok:
            rid = int(df.at[i, "row_id"])
            add_error(
                row_errors, rid,
                "polluted",
                0.98,
                ["ACT_POLLUTED_SUFFIX"],
                {"activity": act, "extracted_base": base},
                f"Activity appears polluted with machine suffix; extracted base='{base}'."
            )

    # 2) Distorted + Synonymous (data-driven)
    # We treat:
    # - distorted: high char similarity to canonical but not exact match (spacing/typo)
    # - synonymous: low char similarity but high token overlap to canonical (different words)
    # Canonical is inferred from frequent normalized labels and polluted bases.
    def norm_for_compare(s):
        s = safe_lower(s)
        s = re.sub(r"[_\-]+", " ", s)
        s = re.sub(r"[^\w\s\(\)]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for i, act in df["_act_raw"].items():
        rid = int(df.at[i, "row_id"])
        # if polluted, compare base instead of full
        act_cmp = polluted_base_by_idx.get(i, act)
        actn = norm_for_compare(act_cmp)

        canon, score = nearest_canon(act_cmp)
        if canon is None:
            continue

        if actn == canon:
            continue

        tok = jaccard_tokens(actn, canon)
        ch = char_similarity(actn, canon)

        # Distorted: very similar strings but not equal
        # Aggressive thresholds to avoid zero detections
        if ch >= 0.86 and tok >= 0.70:
            conf = min(0.95, 0.55 + 0.45 * ch)
            add_error(
                row_errors, rid,
                "distorted",
                conf,
                ["ACT_NEAR_CANONICAL_TYPO"],
                {"activity": act, "canonical_guess": canon, "char_sim": round(ch, 3), "token_jaccard": round(tok, 3)},
                f"Activity likely distorted/typo variant of canonical='{canon}' (char_sim={ch:.3f}, token_jaccard={tok:.3f})."
            )
        # Synonymous: token overlap moderate but char similarity lower (different wording)
        elif tok >= 0.45 and ch <= 0.82:
            conf = max(0.55, min(0.9, 0.35 + 0.55 * tok))
            add_error(
                row_errors, rid,
                "synonymous",
                conf,
                ["ACT_SEMANTIC_VARIANT"],
                {"activity": act, "canonical_guess": canon, "char_sim": round(ch, 3), "token_jaccard": round(tok, 3)},
                f"Activity likely synonymous variant mapped to canonical='{canon}' (token_jaccard={tok:.3f}, char_sim={ch:.3f})."
            )

    # 3) Collateral duplicates (exact + near)
    # Exact duplicates: same case, activity, timestamp, resource
    key_cols = ["_case", "_act_low", "_ts", "_res_norm"]
    exact_dup_mask = df.duplicated(subset=key_cols, keep=False) & (~df["_ts"].isna())
    for rid in df.loc[exact_dup_mask, "row_id"].tolist():
        add_error(
            row_errors, int(rid),
            "collateral",
            0.97,
            ["DUP_EXACT_CASE_ACT_TS_RES"],
            {"key": df.loc[df["row_id"] == rid, key_cols].iloc[0].to_dict()},
            "Exact duplicate event (same case, activity, timestamp, resource)."
        )

    # Near duplicates: within same case, same activity+resource, within <= 3 seconds
    df_sorted = df.sort_values(["_case", "_ts", "row_id"], kind="mergesort")
    for case, g in df_sorted.groupby("_case", sort=False):
        g = g[~g["_ts"].isna()].copy()
        if len(g) < 2:
            continue
        g["prev_ts"] = g["_ts"].shift(1)
        g["prev_act"] = g["_act_low"].shift(1)
        g["prev_res"] = g["_res_norm"].shift(1)
        g["dt"] = (g["_ts"] - g["prev_ts"]).dt.total_seconds()
        near = (g["dt"].between(0, 3, inclusive="both")) & (g["_act_low"] == g["prev_act"]) & (g["_res_norm"] == g["prev_res"])
        for _, r in g.loc[near].iterrows():
            rid = int(r["row_id"])
            add_error(
                row_errors, rid,
                "collateral",
                0.85,
                ["DUP_NEAR_3S_SAME_ACT_RES"],
                {"case": case, "activity": r["_act_raw"], "resource": r["_res_norm"], "dt_seconds": float(r["dt"])},
                f"Near-duplicate event within {r['dt']:.3f}s for same case/activity/resource."
            )

    # 4) Form-based timestamps: repeated identical timestamps within a case for multiple different activities/resources
    # Heuristic: if a case has a timestamp repeated >=3 times and involves >=2 distinct activities, flag all but first occurrence.
    for case, g in df_sorted.groupby("_case", sort=False):
        g = g[~g["_ts"].isna()].copy()
        if g.empty:
            continue
        counts = g.groupby("_ts").size()
        repeated_ts = counts[counts >= 3].index
        if len(repeated_ts) == 0:
            continue
        for ts in repeated_ts:
            gg = g[g["_ts"] == ts].copy()
            if gg["_act_low"].nunique() >= 2:
                # flag all except the earliest row_id as overwritten
                gg = gg.sort_values("row_id")
                for rid in gg["row_id"].iloc[1:].tolist():
                    add_error(
                        row_errors, int(rid),
                        "formbased",
                        0.78,
                        ["CASE_TS_REPEATED_3PLUS_MULTI_ACT"],
                        {"case": case, "timestamp": str(ts), "n_events_same_ts": int(len(gg)), "distinct_activities": int(gg["_act_low"].nunique())},
                        "Multiple different events share identical timestamp within a case (likely form-based overwrite)."
                    )

    # 5) Homonymous: same label used in clearly different contexts
    # Build context signatures: (prev_act, next_act, resource_bucket)
    # If an activity label has multiple well-separated context clusters, flag minority cluster rows.
    df_ctx = df_sorted.copy()
    df_ctx["prev_act"] = df_ctx.groupby("_case")["_act_low"].shift(1).fillna("__START__")
    df_ctx["next_act"] = df_ctx.groupby("_case")["_act_low"].shift(-1).fillna("__END__")
    # resource bucket: empty vs non-empty + prefix
    def res_bucket(r):
        r = norm_ws(r)
        if r == "":
            return "RES_EMPTY"
        # take prefix before '-' if present
        return "RES_" + r.split("-")[0].upper()

    df_ctx["res_bucket"] = df_ctx["_res_norm"].map(res_bucket)
    df_ctx["ctx_sig"] = df_ctx["prev_act"] + ">>" + df_ctx["_act_low"] + ">>" + df_ctx["next_act"] + ">>" + df_ctx["res_bucket"]

    # For each activity label, see if it appears in >=2 strong context signatures
    for act, g in df_ctx.groupby("_act_low", sort=False):
        if len(g) < 30:
            continue  # need enough evidence
        sig_counts = g["ctx_sig"].value_counts()
        strong = sig_counts[sig_counts >= max(5, int(0.08 * len(g)))]
        if len(strong) >= 2:
            # flag rows belonging to non-dominant strong signatures (minority meaning)
            dominant_sig = strong.index[0]
            for _, r in g.iterrows():
                if r["ctx_sig"] != dominant_sig and r["ctx_sig"] in set(strong.index):
                    rid = int(r["row_id"])
                    add_error(
                        row_errors, rid,
                        "homonymous",
                        0.62,
                        ["ACT_MULTI_CONTEXT_SIGNATURE"],
                        {"activity": r["_act_raw"], "ctx_sig": r["ctx_sig"], "dominant_ctx_sig": dominant_sig,
                         "sig_count": int(sig_counts[r["ctx_sig"]]), "dominant_count": int(sig_counts[dominant_sig])},
                        "Same activity label appears in multiple strong, distinct context signatures (possible homonym)."
                    )

    # -----------------------------
    # Assemble output
    # -----------------------------
    out = pd.DataFrame({"row_id": df["row_id"]})
    out["error_flag"] = out["row_id"].map(lambda rid: rid in row_errors)

    def join_types(rid):
        if rid not in row_errors:
            return ""
        return "|".join(sorted(row_errors[rid]["types"]))

    def conf_score(rid):
        if rid not in row_errors:
            return 0.0
        # combine confidences: 1 - product(1-c)
        cs = row_errors[rid]["conf"]
        p = 1.0
        for c in cs:
            p *= (1.0 - float(c))
        return float(max(0.0, min(1.0, 1.0 - p)))

    def join_tags(rid):
        if rid not in row_errors:
            return ""
        return "|".join(sorted(row_errors[rid]["tags"]))

    def evidence_json(rid):
        if rid not in row_errors:
            return ""
        return json.dumps(row_errors[rid]["evidence"], ensure_ascii=False)

    def description(rid):
        if rid not in row_errors:
            return ""
        # keep concise but specific
        return " ; ".join(row_errors[rid]["desc"][:3]) + ("" if len(row_errors[rid]["desc"]) <= 3 else f" ; (+{len(row_errors[rid]['desc'])-3} more)")

    out["error_types"] = out["row_id"].map(join_types)
    out["error_confidence"] = out["row_id"].map(conf_score)
    out["error_tags"] = out["row_id"].map(join_tags)
    out["error_evidence"] = out["row_id"].map(evidence_json)
    out["error_description"] = out["row_id"].map(description)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    # Print quick stats (helps avoid "zero errors detected")
    n_err = int(out["error_flag"].sum())
    n = len(out)
    dist = Counter()
    for rid in out.loc[out["error_flag"], "row_id"].tolist():
        for t in row_errors[int(rid)]["types"]:
            dist[t] += 1
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Rows: {n}, Errors flagged: {n_err} ({(n_err/n*100 if n else 0):.2f}%)")
    print("Type distribution:", dict(dist))


if __name__ == "__main__":
    main()