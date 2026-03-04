#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_synonymous_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_synonymous_noLabel.detected.csv

Detects (using ONLY Case, Activity, Timestamp, Resource):
- FORMBASED
- POLLUTED
- DISTORTED
- SYNONYMOUS
- COLLATERAL
- HOMONYMOUS (heuristic, conservative)

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


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_synonymous_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_synonymous_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# --- Tunable thresholds (aggressive but evidence-based) ---
COLLATERAL_NEAR_SECONDS = 3.0          # near-duplicate window
FORMBASED_MIN_GROUP_SIZE = 3           # repeated same timestamp within case
FORMBASED_MAX_GROUP_SIZE = 50          # avoid huge batches being auto-flagged without more evidence
FORMBASED_SAME_TS_RATIO_MIN = 0.25     # within-case repeated timestamp ratio threshold
DISTORTED_SIM_MIN = 0.86               # similarity to canonical to call typo-like
DISTORTED_SIM_MAX = 0.97               # if too close, likely just formatting; still can be distorted if has obvious artifacts
SYNONYM_SIM_MIN = 0.72                 # similarity to canonical for synonym-like (lower than distorted)
HOMONYM_CONTEXT_JSD_MIN = 0.35         # divergence threshold for context distributions
HOMONYM_MIN_SUPPORT = 15               # minimum occurrences of label to consider homonymy


# -------------------- Helpers --------------------
def norm_text(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_for_match(s: str) -> str:
    s = norm_text(s).lower()
    # keep alphanumerics and spaces
    s = re.sub(r"[^a-z0-9\s\-\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def js_divergence(p: dict, q: dict, eps=1e-12) -> float:
    # Jensen-Shannon divergence between two discrete distributions
    keys = set(p) | set(q)
    P = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    Q = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    P = P / (P.sum() + eps)
    Q = Q / (Q.sum() + eps)
    M = 0.5 * (P + Q)

    def kl(a, b):
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return np.sum(a * np.log2(a / b))

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)

def add_error(row_errors, etype, conf, tag, evidence, desc):
    row_errors["types"].add(etype)
    row_errors["tags"].add(tag)
    row_errors["evidence"].append(evidence)
    row_errors["descriptions"].append(desc)
    # combine confidence: noisy-or
    row_errors["conf"] = 1.0 - (1.0 - row_errors["conf"]) * (1.0 - conf)

def finalize_row(row_errors):
    types = sorted(row_errors["types"])
    tags = sorted(row_errors["tags"])
    return {
        "error_flag": bool(types),
        "error_types": "|".join(t.lower() for t in types),
        "error_confidence": round(float(row_errors["conf"]), 6),
        "error_tags": "|".join(tags),
        "error_evidence": json.dumps(row_errors["evidence"], ensure_ascii=False),
        "error_description": " ; ".join(row_errors["descriptions"])[:2000],
    }


# -------------------- Main detection logic --------------------
def main():
    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # row_id = original row index (0-based)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    # normalize
    df["Case_n"] = df["Case"].astype(str)
    df["Activity_raw"] = df["Activity"].apply(norm_text)
    df["Activity_n"] = df["Activity_raw"].apply(norm_for_match)
    df["Resource_raw"] = df["Resource"]  # keep as-is; empty is allowed
    df["Resource_n"] = df["Resource"].apply(lambda x: norm_for_match(x) if not (pd.isna(x)) else "")

    # parse timestamps
    df["Timestamp_parsed"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
    # timestamp parse failures are real errors (format/null) but not listed in types.
    # We will treat them as DISTORTED-like? No: keep within allowed types -> flag as DISTORTED (timestamp malformed)
    # because we must output only the defined error types. We'll tag it clearly.
    # (If you prefer, you can remove this block.)
    bad_ts = df["Timestamp_parsed"].isna()

    # Prepare per-row error containers
    errors = {
        rid: {"types": set(), "tags": set(), "evidence": [], "descriptions": [], "conf": 0.0}
        for rid in df["row_id"].tolist()
    }

    # --- 0) Timestamp malformed (mapped to DISTORTED with explicit tag) ---
    for rid in df.loc[bad_ts, "row_id"].tolist():
        ts_val = df.loc[rid, "Timestamp"]
        add_error(
            errors[rid],
            "DISTORTED",
            0.95,
            "ts_parse_fail",
            {"timestamp": str(ts_val)},
            f"Timestamp cannot be parsed: {ts_val!r}."
        )

    # Work only with rows with valid timestamps for time-based rules
    dfv = df.loc[~bad_ts].copy()

    # Sort within case for context features
    dfv = dfv.sort_values(["Case_n", "Timestamp_parsed", "row_id"]).reset_index(drop=True)

    # Build prev/next activity context per case
    dfv["prev_act"] = ""
    dfv["next_act"] = ""
    for case, g in dfv.groupby("Case_n", sort=False):
        idx = g.index.to_numpy()
        acts = g["Activity_n"].to_list()
        prevs = [""] + acts[:-1]
        nexts = acts[1:] + [""]
        dfv.loc[idx, "prev_act"] = prevs
        dfv.loc[idx, "next_act"] = nexts

    # ---------------- POLLUTED detection ----------------
    # Pattern: base + "_" + 5-12 alnum + "_" + yyyymmdd hhmmss + optional millis
    polluted_re = re.compile(
        r"^(?P<base>.+?)_(?P<suffix>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6}\d{0,6})$"
    )

    for i, r in df.iterrows():
        rid = int(r["row_id"])
        act_raw = r["Activity_raw"]
        m = polluted_re.match(act_raw)
        if m:
            base = norm_text(m.group("base"))
            conf = 0.92
            add_error(
                errors[rid],
                "POLLUTED",
                conf,
                "act_polluted_suffix",
                {"activity": act_raw, "base_candidate": base, "suffix": m.group("suffix"), "embedded_dt": m.group("dt")},
                f"Activity has machine-generated suffix pattern: {act_raw!r} (base candidate: {base!r})."
            )

    # Create "canonical candidate" for each row: if polluted, strip to base; else raw normalized
    def strip_polluted(act_raw):
        m = polluted_re.match(act_raw)
        if m:
            return norm_for_match(m.group("base"))
        return norm_for_match(act_raw)

    df["Activity_base_n"] = df["Activity_raw"].apply(strip_polluted)

    # ---------------- COLLATERAL detection ----------------
    # Exact duplicates: same Case, Activity_base_n, Timestamp, Resource_n
    # Near duplicates: same Case, Activity_base_n, Resource_n within COLLATERAL_NEAR_SECONDS
    dfv2 = df.loc[~bad_ts].copy()
    dfv2["Activity_base_n"] = dfv2["Activity_raw"].apply(strip_polluted)
    dfv2["Timestamp_parsed"] = pd.to_datetime(dfv2["Timestamp"], errors="coerce", utc=False)
    dfv2 = dfv2.sort_values(["Case_n", "Activity_base_n", "Resource_n", "Timestamp_parsed", "row_id"])

    # exact duplicates
    dup_cols = ["Case_n", "Activity_base_n", "Timestamp", "Resource_n"]
    dup_mask = dfv2.duplicated(subset=dup_cols, keep=False)
    for rid in dfv2.loc[dup_mask, "row_id"].tolist():
        row = df.loc[rid]
        add_error(
            errors[int(rid)],
            "COLLATERAL",
            0.93,
            "dup_exact_case_act_ts_res",
            {"case": str(row["Case"]), "activity": row["Activity_raw"], "timestamp": str(row["Timestamp"]), "resource": str(row["Resource"])},
            "Exact duplicate event (same case, activity, timestamp, resource)."
        )

    # near duplicates
    for (case, act, res), g in dfv2.groupby(["Case_n", "Activity_base_n", "Resource_n"], sort=False):
        if len(g) < 2:
            continue
        ts = g["Timestamp_parsed"].to_numpy()
        rids = g["row_id"].to_numpy()
        # check consecutive time gaps
        for j in range(1, len(g)):
            dt = (ts[j] - ts[j-1]) / np.timedelta64(1, "s")
            if 0 <= dt <= COLLATERAL_NEAR_SECONDS:
                # flag both rows (aggressive)
                for rid in (int(rids[j-1]), int(rids[j])):
                    row = df.loc[rid]
                    conf = 0.78 + 0.15 * (1.0 - min(dt, COLLATERAL_NEAR_SECONDS) / COLLATERAL_NEAR_SECONDS)
                    add_error(
                        errors[rid],
                        "COLLATERAL",
                        min(conf, 0.9),
                        "dup_near_case_act_res_time",
                        {"case": str(row["Case"]), "activity_base": act, "resource": str(row["Resource"]), "dt_seconds": float(dt)},
                        f"Near-duplicate events for same case/activity/resource within {dt:.3f}s."
                    )

    # ---------------- FORMBASED detection ----------------
    # Within a case: multiple events share identical timestamp (same parsed time).
    # Flag those events if repeated timestamp groups are non-trivial and not just 2 events.
    dfv3 = df.loc[~bad_ts].copy()
    dfv3["Timestamp_parsed"] = pd.to_datetime(dfv3["Timestamp"], errors="coerce", utc=False)
    dfv3["ts_key"] = dfv3["Timestamp_parsed"].astype("datetime64[ns]")

    for case, g in dfv3.groupby("Case_n", sort=False):
        if len(g) < 4:
            continue
        counts = g["ts_key"].value_counts()
        repeated_ts = counts[counts >= 2]
        if repeated_ts.empty:
            continue

        repeated_ratio = repeated_ts.sum() / len(g)
        # only consider if enough repetition
        if repeated_ratio < FORMBASED_SAME_TS_RATIO_MIN:
            continue

        for ts_key, cnt in repeated_ts.items():
            if cnt < FORMBASED_MIN_GROUP_SIZE or cnt > FORMBASED_MAX_GROUP_SIZE:
                continue
            grp = g[g["ts_key"] == ts_key].sort_values("row_id")
            # evidence: show neighboring timestamps in case to justify overwrite
            case_times = g.sort_values("Timestamp_parsed")["Timestamp_parsed"].to_list()
            # propose "canonical" as slightly increasing times after the shared timestamp (unknown true),
            # so we only provide evidence, not a fabricated ground truth.
            for rid in grp["row_id"].tolist():
                row = df.loc[int(rid)]
                conf = 0.72 + 0.2 * min(1.0, (cnt - 2) / 6.0) + 0.08 * min(1.0, repeated_ratio)
                add_error(
                    errors[int(rid)],
                    "FORMBASED",
                    min(conf, 0.92),
                    "case_repeated_timestamp_cluster",
                    {
                        "case": str(row["Case"]),
                        "shared_timestamp": str(row["Timestamp"]),
                        "cluster_size": int(cnt),
                        "case_len": int(len(g)),
                        "repeated_ratio": float(repeated_ratio),
                    },
                    f"Multiple events in same case share identical timestamp ({row['Timestamp']}); likely form-based overwrite (cluster size={cnt})."
                )

    # ---------------- SYNONYMOUS / DISTORTED detection (text-based, unsupervised) ----------------
    # Build canonical label set as the most frequent "base" labels (after stripping polluted suffix).
    # Then map each observed label to nearest canonical; decide distorted vs synonymous by similarity bands.
    base_counts = Counter(df["Activity_base_n"].tolist())
    # remove empties
    if "" in base_counts:
        del base_counts[""]
    # canonical candidates: frequent labels
    canonicals = [a for a, c in base_counts.most_common(2000) if c >= 2]  # aggressive
    if not canonicals:
        canonicals = [a for a, c in base_counts.most_common(2000)]

    # Precompute for speed: for each unique activity_base_n, find best canonical match
    unique_acts = list(base_counts.keys())
    best_match = {}
    for a in unique_acts:
        if a in canonicals:
            best_match[a] = (a, 1.0)
            continue
        # compare to top-N by rough length filter
        candidates = [c for c in canonicals if abs(len(c) - len(a)) <= 10]
        if not candidates:
            candidates = canonicals[:500]
        best_c, best_s = None, -1.0
        for c in candidates[:800]:
            s = seq_ratio(a, c)
            if s > best_s:
                best_s = s
                best_c = c
        best_match[a] = (best_c if best_c is not None else a, float(best_s))

    # Decide distorted vs synonymous:
    # - Distorted: high similarity to canonical but not identical, and contains typical typo artifacts (spaces inside word, swapped chars, etc.)
    # - Synonymous: moderate similarity but shares at least one content token with canonical OR has verb/object pattern difference
    def has_typo_artifacts(raw: str) -> bool:
        if raw is None:
            return False
        s = str(raw)
        # internal split like "Revi ew", "Appr ove"
        if re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", s) and not re.search(r"\b(and|or|of|to|for|in|on)\b", s.lower()):
            return True
        # odd character insertion
        if re.search(r"[A-Za-z][^A-Za-z0-9\s\-\(\)][A-Za-z]", s):
            return True
        return False

    def token_set(s: str):
        s = norm_for_match(s)
        toks = [t for t in re.split(r"[\s\-\(\)]+", s) if t]
        return set(toks)

    for i, r in df.iterrows():
        rid = int(r["row_id"])
        a_base = r["Activity_base_n"]
        if not a_base:
            # empty activity is a real issue; map to DISTORTED
            add_error(
                errors[rid],
                "DISTORTED",
                0.9,
                "act_empty",
                {"activity": r["Activity"]},
                "Activity is empty/null."
            )
            continue

        canon, sim = best_match.get(a_base, (a_base, 1.0))
        if canon == a_base:
            continue

        raw = r["Activity_raw"]
        # If polluted already, don't double-count as distorted/syn unless base itself differs from canonical
        # (we still allow multiple errors if base is also distorted/synonymous)
        artifacts = has_typo_artifacts(raw)

        toks_a = token_set(a_base)
        toks_c = token_set(canon)
        overlap = len(toks_a & toks_c)

        # Distorted: very similar + typo artifacts OR very similar and small edit distance
        if sim >= DISTORTED_SIM_MIN:
            # compute normalized edit distance proxy
            ed = 1.0 - sim
            conf = 0.65 + 0.35 * min(1.0, (sim - DISTORTED_SIM_MIN) / (1.0 - DISTORTED_SIM_MIN))
            if artifacts:
                conf = min(0.95, conf + 0.12)
            if sim <= DISTORTED_SIM_MAX or artifacts or ed >= 0.02:
                add_error(
                    errors[rid],
                    "DISTORTED",
                    min(conf, 0.96),
                    "act_near_canonical_typo",
                    {"activity_base": a_base, "canonical": canon, "similarity": float(sim), "raw": raw, "token_overlap": overlap},
                    f"Activity text looks like a typo/variant of canonical label (similarity={sim:.3f}): {raw!r} ~ {canon!r}."
                )
                continue

        # Synonymous: moderate similarity but not typo-close; require some token overlap or shared head/tail token
        if sim >= SYNONYM_SIM_MIN and sim < DISTORTED_SIM_MIN:
            head_match = (a_base.split(" ")[0:1] == canon.split(" ")[0:1])
            tail_match = (a_base.split(" ")[-1:] == canon.split(" ")[-1:])
            if overlap >= 1 or head_match or tail_match:
                conf = 0.55 + 0.35 * min(1.0, (sim - SYNONYM_SIM_MIN) / (DISTORTED_SIM_MIN - SYNONYM_SIM_MIN))
                add_error(
                    errors[rid],
                    "SYNONYMOUS",
                    min(conf, 0.9),
                    "act_semantic_variant_cluster",
                    {"activity_base": a_base, "canonical": canon, "similarity": float(sim), "token_overlap": overlap},
                    f"Activity label appears to be a synonymous/alternative phrasing of canonical label (similarity={sim:.3f}): {raw!r} -> {canon!r}."
                )

    # ---------------- HOMONYMOUS detection (conservative heuristic) ----------------
    # If the same activity_base_n appears in very different contexts (prev/next activities),
    # it may represent different meanings. We flag only when strong evidence exists.
    df_ctx = dfv.copy()
    df_ctx["Activity_base_n"] = df_ctx["Activity_raw"].apply(strip_polluted)

    # Build context distributions per label: P(prev,next)
    label_groups = df_ctx.groupby("Activity_base_n", sort=False)
    context_dist = {}
    support = {}
    for label, g in label_groups:
        if not label:
            continue
        support[label] = len(g)
        if len(g) < HOMONYM_MIN_SUPPORT:
            continue
        ctx = (g["prev_act"].fillna("") + ">>" + g["next_act"].fillna("")).tolist()
        c = Counter(ctx)
        total = sum(c.values())
        context_dist[label] = {k: v / total for k, v in c.items()}

    # Compare each label's context to the "dominant" context mode vs remainder:
    # Split by most common context; if remainder distribution diverges strongly, flag those in minority context.
    for label, g in label_groups:
        if label not in context_dist:
            continue
        if support.get(label, 0) < HOMONYM_MIN_SUPPORT:
            continue

        ctx_series = (g["prev_act"].fillna("") + ">>" + g["next_act"].fillna(""))
        ctx_counts = ctx_series.value_counts()
        if len(ctx_counts) < 3:
            continue

        dominant_ctx = ctx_counts.index[0]
        dom_mask = (ctx_series == dominant_ctx)
        dom_n = int(dom_mask.sum())
        rem_n = int((~dom_mask).sum())
        if rem_n < max(5, int(0.15 * len(g))):
            continue

        # Build distributions
        dom_counter = Counter(ctx_series[dom_mask].tolist())
        rem_counter = Counter(ctx_series[~dom_mask].tolist())
        dom_dist = {k: v / dom_n for k, v in dom_counter.items()}
        rem_dist = {k: v / rem_n for k, v in rem_counter.items()}

        jsd = js_divergence(dom_dist, rem_dist)
        if jsd < HOMONYM_CONTEXT_JSD_MIN:
            continue

        # Flag minority-context rows as potential homonymous usage
        for rid in g.loc[~dom_mask, "row_id"].tolist():
            row = df.loc[int(rid)]
            conf = 0.55 + 0.4 * min(1.0, (jsd - HOMONYM_CONTEXT_JSD_MIN) / (1.0 - HOMONYM_CONTEXT_JSD_MIN))
            add_error(
                errors[int(rid)],
                "HOMONYMOUS",
                min(conf, 0.9),
                "label_context_divergence",
                {
                    "activity_base": label,
                    "js_divergence": float(jsd),
                    "dominant_context": dominant_ctx,
                    "row_context": str((df_ctx.loc[df_ctx["row_id"] == int(rid), "prev_act"].values[0] if (df_ctx["row_id"] == int(rid)).any() else "")) +
                                   ">>" +
                                   str((df_ctx.loc[df_ctx["row_id"] == int(rid), "next_act"].values[0] if (df_ctx["row_id"] == int(rid)).any() else "")),
                    "support": int(support[label]),
                    "dominant_support": int(dom_n),
                    "minority_support": int(rem_n),
                },
                f"Same activity label appears in strongly divergent surrounding contexts (JSD={jsd:.3f}); possible homonymous meaning for {row['Activity_raw']!r}."
            )

    # ---------------- Output ----------------
    out_rows = []
    for rid in df["row_id"].tolist():
        rec = {"row_id": int(rid)}
        rec.update(finalize_row(errors[int(rid)]))
        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows, columns=[
        "row_id", "error_flag", "error_types", "error_confidence",
        "error_tags", "error_evidence", "error_description"
    ])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(out_df["error_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()