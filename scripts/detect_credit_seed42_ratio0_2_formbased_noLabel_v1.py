#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Mining Event Log Error Detection (credit dataset)

Reads:
  /home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_formbased_noLabel.csv

Writes:
  /home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_formbased_noLabel.detected.csv

Detection uses ONLY: Case, Activity, Timestamp, Resource
(Resource being empty is NOT an error.)
"""

import os
import re
import json
import math
from difflib import SequenceMatcher
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/credit/credit_seed42_ratio0.2_formbased_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/credit_seed42_ratio0.2_formbased_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# --- Tunable thresholds (aggressive by design) ---
COLLATERAL_NEAR_SECONDS = 3.0          # near-duplicate window
FORMBASED_MIN_GROUP_SIZE = 3           # repeated same timestamp within case
FORMBASED_SAME_TS_MAX_SPAN_SEC = 0.0   # exact same timestamp (form overwrite)
DISTORTED_SIM_MIN = 0.86               # similarity to canonical to call typo
DISTORTED_SIM_MAX = 0.97               # avoid flagging exact matches as distorted
SYNONYM_SIM_MAX = 0.82                 # if too similar, it's probably typo not synonym
MIN_ACTIVITY_LEN = 2

# Polluted pattern: base + "_" + 5-12 alnum + "_" + yyyymmdd hhmmss + micro/nano digits
POLLUTED_RE = re.compile(
    r"^(?P<base>.+?)_(?P<suffix>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6}\d{3,6})$"
)

# Normalize whitespace for distorted detection
WS_RE = re.compile(r"\s+")


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_activity(a: str) -> str:
    a = safe_str(a).strip()
    a = WS_RE.sub(" ", a)
    return a


def strip_pollution(a: str):
    """
    If polluted, return (base, suffix, dt_str). Else (None, None, None)
    """
    m = POLLUTED_RE.match(a)
    if not m:
        return None, None, None
    return m.group("base").strip(), m.group("suffix"), m.group("dt")


def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z0-9]+", a.lower()))
    tb = set(re.findall(r"[A-Za-z0-9]+", b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_canonical_vocab(df: pd.DataFrame):
    """
    Build a set of likely canonical activity labels from the dataset itself:
    - Use non-polluted activities as candidates
    - Prefer frequent labels
    """
    acts = df["Activity_norm"].tolist()
    bases = []
    for a in acts:
        base, _, _ = strip_pollution(a)
        if base is not None:
            # polluted -> base candidate
            bases.append(base)
        else:
            bases.append(a)

    # Remove very short / empty
    bases = [b for b in bases if len(b.strip()) >= MIN_ACTIVITY_LEN]
    freq = Counter(bases)

    # Keep all, but we will prefer frequent ones when mapping
    vocab = list(freq.keys())
    return vocab, freq


def best_canonical_match(activity: str, vocab, freq):
    """
    Find best canonical candidate for an activity string.
    Returns (best_label, best_sim, best_jaccard)
    """
    if not activity:
        return None, 0.0, 0.0

    # Speed: compare against top-N frequent labels first, then full if needed
    top = [a for a, _ in freq.most_common(250)]
    candidates = top if len(vocab) > 250 else vocab

    best = None
    best_sim = -1.0
    best_j = -1.0

    for v in candidates:
        s = seq_sim(activity, v)
        if s > best_sim:
            best_sim = s
            best = v
            best_j = token_jaccard(activity, v)

    # If best is weak and we truncated, try full vocab
    if best_sim < 0.80 and len(candidates) != len(vocab):
        for v in vocab:
            s = seq_sim(activity, v)
            if s > best_sim:
                best_sim = s
                best = v
                best_j = token_jaccard(activity, v)

    return best, float(best_sim), float(best_j)


def confidence_from_evidence(base=0.5, boosts=None, penalties=None):
    """
    Combine evidence into [0,1] confidence.
    """
    c = base
    boosts = boosts or []
    penalties = penalties or []
    for b in boosts:
        c += b
    for p in penalties:
        c -= p
    return float(max(0.0, min(1.0, c)))


def main():
    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Keep only required columns for detection (but we don't drop others from df; we just don't use them)
    df_det = df[REQUIRED_COLS].copy()

    # Row id = original row index (0-based)
    df_det["row_id"] = df_det.index.astype(int)

    # Parse timestamp
    df_det["Timestamp_parsed"] = pd.to_datetime(df_det["Timestamp"], errors="coerce", utc=False)

    # Normalize activity
    df_det["Activity_norm"] = df_det["Activity"].apply(normalize_activity)
    df_det["Case_str"] = df_det["Case"].apply(safe_str)
    df_det["Resource_str"] = df_det["Resource"].apply(safe_str)  # empty allowed

    # Prepare output columns
    out = pd.DataFrame({"row_id": df_det["row_id"].values})
    out["error_flag"] = False
    out["error_types"] = ""
    out["error_confidence"] = 0.0
    out["error_tags"] = ""
    out["error_evidence"] = ""
    out["error_description"] = ""

    # Collect per-row findings
    findings = defaultdict(lambda: {
        "types": set(),
        "tags": set(),
        "evidence": [],
        "descriptions": [],
        "conf_parts": []
    })

    # -------------------------
    # 0) Basic format/range checks (Timestamp parse, empty activity/case)
    # (Not listed as a named error type; we will not invent new types.
    #  We will attach as evidence tags but not set error_types unless it supports other types.)
    # -------------------------
    bad_ts = df_det["Timestamp_parsed"].isna()
    for rid in df_det.loc[bad_ts, "row_id"].tolist():
        f = findings[rid]
        f["tags"].add("timestamp_parse_failed")
        f["evidence"].append("Timestamp could not be parsed to datetime")
        f["descriptions"].append("Unparseable Timestamp; downstream temporal rules may be unreliable.")
        f["conf_parts"].append(0.15)  # weak: not a requested error type

    empty_act = df_det["Activity_norm"].str.len().fillna(0) < MIN_ACTIVITY_LEN
    for rid in df_det.loc[empty_act, "row_id"].tolist():
        f = findings[rid]
        f["tags"].add("activity_empty_or_too_short")
        f["evidence"].append("Activity is empty/too short")
        f["descriptions"].append("Activity label is missing or too short to be valid.")
        f["conf_parts"].append(0.20)

    empty_case = df_det["Case_str"].str.strip().eq("")
    for rid in df_det.loc[empty_case, "row_id"].tolist():
        f = findings[rid]
        f["tags"].add("case_missing")
        f["evidence"].append("Case identifier is missing/blank")
        f["descriptions"].append("Case identifier is missing; event cannot be reliably grouped.")
        f["conf_parts"].append(0.20)

    # -------------------------
    # Build canonical vocabulary from dataset itself
    # -------------------------
    vocab, freq = build_canonical_vocab(df_det)

    # -------------------------
    # 1) POLLUTED detection
    # -------------------------
    polluted_rows = []
    for i, row in df_det.iterrows():
        rid = int(row["row_id"])
        a = row["Activity_norm"]
        base, suffix, dt = strip_pollution(a)
        if base is not None:
            polluted_rows.append(rid)
            f = findings[rid]
            f["types"].add("polluted")
            f["tags"].add("polluted_suffix_pattern")
            f["evidence"].append(f"Activity matches polluted pattern: base='{base}', suffix='{suffix}', dt='{dt}'")
            f["descriptions"].append(f"POLLUTED: activity has machine-generated suffix; canonical likely '{base}'.")
            # strong evidence: exact regex match
            f["conf_parts"].append(0.55)

    # -------------------------
    # 2) COLLATERAL detection (exact duplicates + near duplicates)
    # -------------------------
    # Exact duplicates: same Case, Activity_norm, Timestamp (string), Resource_str
    dup_cols = ["Case_str", "Activity_norm", "Timestamp", "Resource_str"]
    dup_mask = df_det.duplicated(subset=dup_cols, keep=False)
    for rid in df_det.loc[dup_mask, "row_id"].tolist():
        f = findings[int(rid)]
        f["types"].add("collateral")
        f["tags"].add("collateral_exact_duplicate")
        # Provide group size evidence
        key = tuple(df_det.loc[df_det["row_id"] == rid, dup_cols].iloc[0].tolist())
        group_size = int((df_det[dup_cols].apply(tuple, axis=1) == key).sum())
        f["evidence"].append(f"Exact duplicate group size={group_size} for (Case,Activity,Timestamp,Resource)={key}")
        f["descriptions"].append("COLLATERAL: exact duplicate event logged multiple times.")
        f["conf_parts"].append(0.60)

    # Near duplicates: within case, same activity + same resource, timestamps within COLLATERAL_NEAR_SECONDS
    # Only for rows with parseable timestamps
    df_ts = df_det[~df_det["Timestamp_parsed"].isna()].copy()
    df_ts.sort_values(["Case_str", "Activity_norm", "Resource_str", "Timestamp_parsed"], inplace=True)

    # Compare consecutive within each (case, activity, resource)
    for (case, act, res), g in df_ts.groupby(["Case_str", "Activity_norm", "Resource_str"], sort=False):
        if len(g) < 2:
            continue
        t = g["Timestamp_parsed"].values
        rids = g["row_id"].values
        # compute deltas between consecutive
        deltas = (t[1:] - t[:-1]) / np.timedelta64(1, "s")
        for idx, dsec in enumerate(deltas):
            if dsec <= COLLATERAL_NEAR_SECONDS:
                # mark both events as collateral (aggressive)
                for rid in (int(rids[idx]), int(rids[idx + 1])):
                    f = findings[rid]
                    f["types"].add("collateral")
                    f["tags"].add("collateral_near_duplicate")
                    f["evidence"].append(
                        f"Near-duplicate within {COLLATERAL_NEAR_SECONDS}s in same (Case,Activity,Resource)=({case},{act},{res}); Δt={float(dsec):.3f}s"
                    )
                    f["descriptions"].append("COLLATERAL: repeated logging of same activity instance within implausibly short interval.")
                    # moderate-strong
                    f["conf_parts"].append(0.45 if dsec > 0 else 0.55)

    # -------------------------
    # 3) FORMBASED detection (same timestamp repeated within a case)
    # -------------------------
    # Within each case, if multiple different activities share exact same timestamp string,
    # likely form overwrite. Flag all but the first occurrence at that timestamp.
    df_case = df_det.copy()
    df_case["Timestamp_str"] = df_case["Timestamp"].apply(safe_str)

    for case, g in df_case.groupby("Case_str", sort=False):
        # group by exact timestamp string
        for ts, gg in g.groupby("Timestamp_str", sort=False):
            if ts.strip() == "":
                continue
            if len(gg) >= FORMBASED_MIN_GROUP_SIZE:
                # If multiple activities at same timestamp, formbased likely.
                # Keep the first as "possibly correct", flag the rest.
                gg_sorted = gg.sort_values("row_id")
                # If all activities identical, this is more collateral than formbased; still can be both.
                all_same_act = gg_sorted["Activity_norm"].nunique() == 1
                # Flag all except first (aggressive)
                to_flag = gg_sorted.iloc[1:] if len(gg_sorted) > 1 else gg_sorted.iloc[0:0]
                for rid in to_flag["row_id"].tolist():
                    f = findings[int(rid)]
                    f["types"].add("formbased")
                    f["tags"].add("formbased_same_timestamp_in_case")
                    f["evidence"].append(
                        f"Case '{case}' has {len(gg_sorted)} events with identical Timestamp='{ts}' (activities_unique={gg_sorted['Activity_norm'].nunique()})"
                    )
                    if all_same_act:
                        f["descriptions"].append(
                            "FORMBASED: multiple events share identical timestamp in same case (could be overwrite); also resembles duplicates."
                        )
                        f["conf_parts"].append(0.35)
                    else:
                        f["descriptions"].append(
                            "FORMBASED: multiple different activities share identical timestamp in same case (form submission/overwrite likely)."
                        )
                        f["conf_parts"].append(0.50)

    # -------------------------
    # 4) DISTORTED + SYNONYMOUS detection (text-based, dataset-driven)
    # -------------------------
    # Strategy:
    # - If polluted, compare base part for distortion/synonymy.
    # - Find best canonical match from vocab.
    # - Distorted: high similarity but not exact.
    # - Synonymous: lower similarity but high token overlap OR known verb variants (light heuristic).
    verb_syn = {
        "review": {"assess", "evaluate", "inspect", "check", "examine"},
        "reject": {"deny", "decline", "refuse"},
        "approve": {"grant", "accept", "authorize"},
        "request": {"ask", "solicit"},
        "submit": {"send", "file", "lodge"},
        "close": {"finish", "complete", "terminate", "end"},
    }

    def verb_variant(a, b):
        ta = re.findall(r"[A-Za-z]+", a.lower())
        tb = re.findall(r"[A-Za-z]+", b.lower())
        if not ta or not tb:
            return False
        # check first verb-ish token
        va, vb = ta[0], tb[0]
        for k, vs in verb_syn.items():
            if (va == k and vb in vs) or (vb == k and va in vs) or (va in vs and vb in vs):
                return True
        return False

    for i, row in df_det.iterrows():
        rid = int(row["row_id"])
        a_raw = row["Activity_norm"]
        if len(a_raw) < MIN_ACTIVITY_LEN:
            continue

        base, _, _ = strip_pollution(a_raw)
        a = base if base is not None else a_raw

        best, sim, jac = best_canonical_match(a, vocab, freq)
        if not best or best.strip() == "" or best == a:
            continue

        # Distorted: very similar strings (typo/spacing)
        if DISTORTED_SIM_MIN <= sim <= DISTORTED_SIM_MAX:
            f = findings[rid]
            f["types"].add("distorted")
            f["tags"].add("distorted_high_string_similarity")
            f["evidence"].append(f"Best canonical='{best}' similarity={sim:.3f} jaccard={jac:.3f} observed='{a}'")
            f["descriptions"].append(f"DISTORTED: activity looks like a typo/variant of '{best}'.")
            # confidence scales with similarity
            f["conf_parts"].append(0.30 + 0.40 * (sim - DISTORTED_SIM_MIN) / (DISTORTED_SIM_MAX - DISTORTED_SIM_MIN + 1e-9))

        # Synonymous: not too similar (avoid typos), but token overlap or verb variant suggests same meaning
        # Aggressive: if jac high and sim moderate, or verb variant and jac moderate.
        if sim < DISTORTED_SIM_MIN and (
            (jac >= 0.60 and sim <= SYNONYM_SIM_MAX) or
            (verb_variant(a, best) and jac >= 0.40)
        ):
            f = findings[rid]
            f["types"].add("synonymous")
            f["tags"].add("synonymous_token_or_verb_variant")
            f["evidence"].append(f"Best canonical='{best}' similarity={sim:.3f} jaccard={jac:.3f} observed='{a}'")
            f["descriptions"].append(f"SYNONYMOUS: activity wording differs but likely same as '{best}'.")
            # confidence: stronger with higher jaccard and moderate sim
            f["conf_parts"].append(0.25 + 0.45 * min(1.0, jac))

    # -------------------------
    # 5) HOMONYMOUS detection (same label used in different semantic contexts)
    # Dataset-driven heuristic:
    # - For each activity label, look at distribution of predecessor/successor activities within cases.
    # - If label has two+ distinct context clusters (very different neighbors), flag as homonymous.
    # This is inherently weak without domain semantics; keep confidence low-moderate.
    # -------------------------
    df_seq = df_det[~df_det["Timestamp_parsed"].isna()].copy()
    df_seq.sort_values(["Case_str", "Timestamp_parsed", "row_id"], inplace=True)

    # Build prev/next context
    prev_act = []
    next_act = []
    for case, g in df_seq.groupby("Case_str", sort=False):
        acts = g["Activity_norm"].tolist()
        prevs = [None] + acts[:-1]
        nexts = acts[1:] + [None]
        prev_act.extend(prevs)
        next_act.extend(nexts)
    df_seq["prev_act"] = prev_act
    df_seq["next_act"] = next_act

    # For each activity, compute context signatures
    # signature = (prev_act, next_act) with None as boundary
    ctx_by_act = defaultdict(list)
    for _, r in df_seq.iterrows():
        act = r["Activity_norm"]
        ctx_by_act[act].append((r["prev_act"], r["next_act"], int(r["row_id"])))

    for act, ctxs in ctx_by_act.items():
        if len(ctxs) < 30:
            continue  # need enough evidence
        # Count top contexts
        cnt = Counter([(p, n) for p, n, _ in ctxs])
        top = cnt.most_common(5)
        if len(top) < 2:
            continue
        (c1, n1), (c2, n2) = top[0][0], top[1][0]
        share1 = top[0][1] / len(ctxs)
        share2 = top[1][1] / len(ctxs)

        # If two dominant contexts and they are very different (both prev and next differ),
        # treat as potential homonym.
        very_diff = (c1 != c2) and (n1 != n2)
        if very_diff and share1 >= 0.25 and share2 >= 0.15:
            # Flag rows belonging to minority context(s) as homonymous (aggressive),
            # but keep confidence modest.
            for p, n, rid in ctxs:
                if (p, n) == (c1, n1):
                    continue
                f = findings[rid]
                f["types"].add("homonymous")
                f["tags"].add("homonymous_context_split")
                f["evidence"].append(
                    f"Activity '{act}' shows multiple dominant contexts: top1(prev='{c1}',next='{n1}',share={share1:.2f}), "
                    f"top2(prev='{c2}',next='{n2}',share={share2:.2f}); this row context(prev='{p}',next='{n}')"
                )
                f["descriptions"].append(
                    "HOMONYMOUS: same activity label appears in distinct process contexts, suggesting different meanings."
                )
                # confidence depends on dominance and difference
                f["conf_parts"].append(0.20 + 0.35 * min(1.0, share1 + share2))

    # -------------------------
    # Finalize per-row output
    # -------------------------
    for rid in out["row_id"].tolist():
        f = findings.get(int(rid))
        if not f:
            continue

        types = sorted([t for t in f["types"]])
        # Only flag as error if it matches one of the requested error types
        requested_types = {"formbased", "polluted", "distorted", "synonymous", "collateral", "homonymous"}
        types = [t for t in types if t in requested_types]

        if types:
            out.loc[out["row_id"] == rid, "error_flag"] = True
            out.loc[out["row_id"] == rid, "error_types"] = "|".join(types)

            # Confidence: combine evidence parts with diminishing returns
            parts = sorted(f["conf_parts"], reverse=True)
            c = 0.0
            for p in parts:
                c = 1 - (1 - c) * (1 - min(0.95, max(0.0, p)))
            # small penalty if only weak homonymous
            if types == ["homonymous"]:
                c = min(c, 0.65)
            out.loc[out["row_id"] == rid, "error_confidence"] = float(max(0.05, min(1.0, c)))

            out.loc[out["row_id"] == rid, "error_tags"] = "|".join(sorted(f["tags"]))
            out.loc[out["row_id"] == rid, "error_evidence"] = json.dumps(f["evidence"], ensure_ascii=False)
            out.loc[out["row_id"] == rid, "error_description"] = " ".join(f["descriptions"])
        else:
            # keep clean (do not flag based on non-requested checks)
            pass

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote: {OUTPUT_PATH}")
    print(out["error_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()