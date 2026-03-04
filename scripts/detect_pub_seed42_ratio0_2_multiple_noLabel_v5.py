#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection (precision-oriented)

Fix for ITERATION 5 issue:
- Previous version flagged 60.7% rows (too aggressive) mainly due to over-flagging
  "homonymous" and "synonymous".
- This script is intentionally conservative:
  * SYNONYMOUS: only if activity matches a high-precision synonym pattern AND the
    canonical label exists in the dataset AND similarity is high.
  * HOMONYMOUS: only if the same normalized label appears in clearly different
    contexts (prev/next activity) with strong evidence and enough support.
  * DISTORTED: only if very close to an existing canonical label (high similarity)
    and not already explained by polluted/synonym.
  * FORM-BASED: only for timestamp collisions within a case with >=3 events sharing
    exact timestamp and evidence of "should be different" (different activities/resources).
  * COLLATERAL: exact duplicates OR near-duplicates within 2 seconds for same case+activity+resource.

Input:  /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv
"""

import os
import re
import math
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict, Counter

INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_multiple_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_multiple_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# -----------------------------
# Helpers
# -----------------------------
def norm_space(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_activity_base(a: str) -> str:
    """Normalize activity for comparisons (casefold, collapse spaces)."""
    a = norm_space(a).casefold()
    return a

def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def safe_bool(x):
    return bool(x) and str(x).lower() not in ("nan", "none")

def add_error(row_id, etype, conf, tag, evidence, desc,
              errors, confs, tags, evidences, descs):
    errors[row_id].add(etype)
    confs[row_id] = max(confs[row_id], conf)
    tags[row_id].add(tag)
    evidences[row_id].append(evidence)
    descs[row_id].append(desc)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(INPUT_PATH)

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

# Keep only required columns for detection (per instruction)
work = df[REQUIRED_COLS].copy()
work["row_id"] = work.index.astype(int)

# Parse timestamps
work["Timestamp_parsed"] = pd.to_datetime(work["Timestamp"], errors="coerce", utc=False)

# Normalize fields
work["Case_n"] = work["Case"].astype(str)
work["Activity_raw"] = work["Activity"].astype(str)
work["Activity_n"] = work["Activity_raw"].map(norm_activity_base)
work["Resource_n"] = work["Resource"].astype(str).map(norm_space)  # empty allowed; not an error

# -----------------------------
# Output containers
# -----------------------------
errors = defaultdict(set)
confs = defaultdict(float)
tags = defaultdict(set)
evidences = defaultdict(list)
descs = defaultdict(list)

# -----------------------------
# 0) Basic format/range checks (conservative)
# -----------------------------
# Timestamp unparsable is a real error (format)
bad_ts = work["Timestamp_parsed"].isna()
for rid, ts in work.loc[bad_ts, ["row_id", "Timestamp"]].itertuples(index=False):
    add_error(
        rid, "timestamp_format", 0.95, "TS_PARSE_FAIL",
        f"Timestamp='{ts}' could not be parsed",
        "Unparsable timestamp format.",
        errors, confs, tags, evidences, descs
    )

# Empty/NULL Case or Activity are errors (but Resource empty is NOT an error)
bad_case = work["Case"].isna() | (work["Case_n"].str.strip() == "") | (work["Case_n"].str.lower() == "nan")
for rid, casev in work.loc[bad_case, ["row_id", "Case"]].itertuples(index=False):
    add_error(
        rid, "case_missing", 0.98, "CASE_MISSING",
        f"Case='{casev}'",
        "Missing/empty case identifier.",
        errors, confs, tags, evidences, descs
    )

bad_act = work["Activity"].isna() | (work["Activity_raw"].str.strip() == "") | (work["Activity_raw"].str.lower() == "nan")
for rid, actv in work.loc[bad_act, ["row_id", "Activity"]].itertuples(index=False):
    add_error(
        rid, "activity_missing", 0.98, "ACT_MISSING",
        f"Activity='{actv}'",
        "Missing/empty activity label.",
        errors, confs, tags, evidences, descs
    )

# -----------------------------
# 1) POLLUTED detection (high precision)
# Pattern: "<base>_<5-12 alnum>_<YYYYMMDD HHMMSSmmm...>"
# -----------------------------
polluted_re = re.compile(
    r"^(?P<base>.+?)_(?P<suffix>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6}\d{3,6})$"
)

polluted_rows = []
for rid, act in work[["row_id", "Activity_raw"]].itertuples(index=False):
    m = polluted_re.match(norm_space(act))
    if m:
        base = norm_space(m.group("base"))
        polluted_rows.append((rid, base, m.group("suffix"), m.group("dt")))

# Only flag as polluted if base label exists elsewhere in dataset (evidence of canonical)
activity_set_n = set(work["Activity_n"].unique())
for rid, base, suf, dt in polluted_rows:
    base_n = norm_activity_base(base)
    if base_n in activity_set_n:
        add_error(
            rid, "polluted", 0.92, "ACT_POLLUTED_PATTERN",
            f"Activity='{work.loc[rid,'Activity_raw']}' matches polluted pattern; base='{base}' exists in log",
            f"Activity label contains machine-generated suffix '{suf}' and embedded datetime '{dt}'.",
            errors, confs, tags, evidences, descs
        )

# -----------------------------
# Build canonical activity candidates (for distorted/synonymous)
# Canonical = activities that are NOT polluted-looking and are frequent enough.
# -----------------------------
def looks_polluted(a: str) -> bool:
    return bool(polluted_re.match(norm_space(a)))

# frequency on normalized activity
freq = Counter(work["Activity_n"].tolist())
# canonical candidates: not polluted, length>=3, and appears >= 3 times (reduces noise)
canonical = [a for a in freq if (not looks_polluted(a)) and len(a) >= 3 and freq[a] >= 3]
canonical_set = set(canonical)

# -----------------------------
# 2) DISTORTED detection (very conservative)
# Only if:
# - not polluted
# - not exact canonical
# - best match in canonical has similarity >= 0.92
# - and edit-like evidence: same first letter OR shares >=60% tokens
# -----------------------------
def token_set(s: str):
    return set([t for t in re.split(r"[^a-z0-9]+", s) if t])

def best_canonical_match(a_n: str):
    best = (None, 0.0)
    for c in canonical:
        sim = seq_sim(a_n, c)
        if sim > best[1]:
            best = (c, sim)
    return best

for rid, a_raw, a_n in work[["row_id", "Activity_raw", "Activity_n"]].itertuples(index=False):
    if not safe_bool(a_raw):
        continue
    if looks_polluted(a_raw):
        continue
    if a_n in canonical_set:
        continue

    c, sim = best_canonical_match(a_n)
    if c is None:
        continue

    # extra guards to avoid over-flagging
    toks_a = token_set(a_n)
    toks_c = token_set(c)
    jacc = (len(toks_a & toks_c) / len(toks_a | toks_c)) if (toks_a or toks_c) else 0.0
    first_ok = (a_n[:1] == c[:1]) if (a_n and c) else False

    if sim >= 0.92 and (first_ok or jacc >= 0.60):
        add_error(
            rid, "distorted", clamp01(0.70 + 0.30 * sim), "ACT_DISTORTED_SIMILARITY",
            f"Activity='{a_raw}' ~ canonical='{c}' (similarity={sim:.3f}, jaccard={jacc:.3f})",
            f"Activity label likely contains a typo; closest canonical label is '{c}'.",
            errors, confs, tags, evidences, descs
        )

# -----------------------------
# 3) SYNONYMOUS detection (high precision, dictionary + strong similarity)
# Only flag if:
# - activity matches a known synonym phrase pattern
# - mapped canonical exists in dataset canonical_set
# - similarity to canonical >= 0.80 (guards against random matches)
# -----------------------------
# Minimal high-precision synonym map (extendable)
SYN_MAP = {
    # review application
    "review case": "review application",
    "assess application": "review application",
    "evaluate application": "review application",
    "inspect application": "review application",
    # reject request
    "reject application": "reject request",
    "deny request": "reject request",
    "decline application": "reject request",
    "refuse request": "reject request",
    # diagnose patient
    "make diagnosis": "diagnose patient",
    "establish diagnosis": "diagnose patient",
    "determine diagnosis": "diagnose patient",
    "confirm diagnosis": "diagnose patient",
    # start production
    "start manufacturing": "start production",
    "begin production": "start production",
    "initiate production run": "start production",
    "launch production": "start production",
    # approve request
    "grant approval": "approve request",
    "approve application": "approve request",
}

for rid, a_raw, a_n in work[["row_id", "Activity_raw", "Activity_n"]].itertuples(index=False):
    if not safe_bool(a_raw):
        continue
    if looks_polluted(a_raw):
        continue

    if a_n in SYN_MAP:
        canon = SYN_MAP[a_n]
        if canon in canonical_set:
            sim = seq_sim(a_n, canon)
            if sim >= 0.80:
                add_error(
                    rid, "synonymous", clamp01(0.75 + 0.20 * sim), "ACT_SYNONYM_DICT",
                    f"Activity='{a_raw}' mapped to canonical='{canon}' (similarity={sim:.3f})",
                    f"Activity label is a synonym/variant of canonical activity '{canon}'.",
                    errors, confs, tags, evidences, descs
                )

# -----------------------------
# 4) COLLATERAL detection (duplicates / near-duplicates)
# Exact duplicate: same Case, Activity_n, Timestamp_parsed, Resource_n
# Near duplicate: same Case, Activity_n, Resource_n within 2 seconds (and same day)
# -----------------------------
# Exact duplicates
dup_cols = ["Case_n", "Activity_n", "Timestamp_parsed", "Resource_n"]
dup_mask = work.duplicated(subset=dup_cols, keep=False) & work["Timestamp_parsed"].notna()
for rid, casev, actv, tsv, resv in work.loc[dup_mask, ["row_id", "Case_n", "Activity_raw", "Timestamp", "Resource_n"]].itertuples(index=False):
    add_error(
        rid, "collateral", 0.93, "DUP_EXACT",
        f"Exact duplicate on (Case,Activity,Timestamp,Resource)=({casev},{actv},{tsv},{resv})",
        "Duplicate event record (exact match).",
        errors, confs, tags, evidences, descs
    )

# Near duplicates within 2 seconds
near_window_s = 2
work_sorted = work.sort_values(["Case_n", "Activity_n", "Resource_n", "Timestamp_parsed", "row_id"])
grp = work_sorted.groupby(["Case_n", "Activity_n", "Resource_n"], dropna=False, sort=False)

for (casev, actn, resn), g in grp:
    g = g[g["Timestamp_parsed"].notna()]
    if len(g) < 2:
        continue
    ts = g["Timestamp_parsed"].values
    rids = g["row_id"].values
    # check consecutive deltas
    prev_t = None
    prev_rid = None
    for t, rid in zip(ts, rids):
        if prev_t is not None:
            delta = (pd.Timestamp(t) - pd.Timestamp(prev_t)).total_seconds()
            if 0 < delta <= near_window_s:
                # flag both rows as collateral (near-duplicate burst)
                for rr in (prev_rid, rid):
                    add_error(
                        int(rr), "collateral", 0.80, "DUP_NEAR_2S",
                        f"Near-duplicate within {delta:.3f}s for Case='{casev}', Activity='{actn}', Resource='{resn}'",
                        "Repeated logging of same activity instance within an implausibly short interval.",
                        errors, confs, tags, evidences, descs
                    )
        prev_t, prev_rid = t, rid

# -----------------------------
# 5) FORM-BASED detection (timestamp collisions within case)
# Conservative: require >=3 events in same case with identical timestamp AND
# at least 2 distinct activities among them.
# -----------------------------
case_ts = work[work["Timestamp_parsed"].notna()].groupby(["Case_n", "Timestamp_parsed"])
for (casev, tsv), g in case_ts:
    if len(g) < 3:
        continue
    distinct_acts = g["Activity_n"].nunique()
    distinct_res = g["Resource_n"].nunique()
    if distinct_acts >= 2:
        # flag all but the first event at that timestamp (keep one as "could be real")
        g2 = g.sort_values("row_id")
        for rid in g2["row_id"].iloc[1:].tolist():
            add_error(
                int(rid), "form-based", 0.78, "CASE_TS_COLLISION_3PLUS",
                f"Case='{casev}' has {len(g)} events at same Timestamp='{tsv}' (distinct_activities={distinct_acts}, distinct_resources={distinct_res})",
                "Multiple events share identical timestamp within a case, consistent with form-based overwrite.",
                errors, confs, tags, evidences, descs
            )

# -----------------------------
# 6) HOMONYMOUS detection (very conservative)
# Only if same Activity_n appears with strongly different surrounding context:
# - compute (prev_activity, next_activity) context per event within case order
# - for an activity label, if it has >=2 context clusters each with support>=10
#   and the top contexts are very dissimilar (Jaccard <= 0.1 on context tokens),
#   then flag events belonging to minority cluster(s).
# This avoids flagging most rows.
# -----------------------------
# Build per-case ordered sequences
work_seq = work.sort_values(["Case_n", "Timestamp_parsed", "row_id"])
work_seq["prev_act"] = work_seq.groupby("Case_n")["Activity_n"].shift(1).fillna("")
work_seq["next_act"] = work_seq.groupby("Case_n")["Activity_n"].shift(-1).fillna("")
work_seq["ctx"] = (work_seq["prev_act"] + ">>" + work_seq["next_act"]).astype(str)

# For each activity, count contexts
act_ctx_counts = work_seq.groupby(["Activity_n", "ctx"]).size().reset_index(name="n")
# Only consider activities with enough occurrences
act_total = work_seq.groupby("Activity_n").size().to_dict()

def ctx_tokens(ctx: str):
    # tokens from prev and next
    return set([t for t in re.split(r"[^a-z0-9]+", ctx) if t])

# Determine homonymous candidates
HOMO_MIN_TOTAL = 80
HOMO_MIN_CLUSTER = 10

homo_activities = []
for actn, total in act_total.items():
    if total < HOMO_MIN_TOTAL:
        continue
    sub = act_ctx_counts[act_ctx_counts["Activity_n"] == actn].sort_values("n", ascending=False)
    if len(sub) < 2:
        continue
    # take top 2 contexts
    top1 = sub.iloc[0]
    top2 = sub.iloc[1]
    if top2["n"] < HOMO_MIN_CLUSTER:
        continue
    t1 = ctx_tokens(top1["ctx"])
    t2 = ctx_tokens(top2["ctx"])
    j = (len(t1 & t2) / len(t1 | t2)) if (t1 or t2) else 1.0
    # require very different contexts
    if j <= 0.10:
        homo_activities.append((actn, top1["ctx"], int(top1["n"]), top2["ctx"], int(top2["n"]), j))

# Flag minority contexts for those activities
for actn, ctx1, n1, ctx2, n2, j in homo_activities:
    # define majority context as ctx1 (highest count)
    majority = ctx1
    # flag events not in majority, but only if they are in a sizable alternative cluster
    alt_contexts = act_ctx_counts[(act_ctx_counts["Activity_n"] == actn) & (act_ctx_counts["ctx"] != majority)]
    alt_contexts = alt_contexts[alt_contexts["n"] >= HOMO_MIN_CLUSTER]
    alt_set = set(alt_contexts["ctx"].tolist())
    if not alt_set:
        continue

    affected = work_seq[(work_seq["Activity_n"] == actn) & (work_seq["ctx"].isin(alt_set))]
    for rid, casev, a_raw, ctx in affected[["row_id", "Case_n", "Activity_raw", "ctx"]].itertuples(index=False):
        add_error(
            int(rid), "homonymous", 0.70, "ACT_HOMONYM_CONTEXT_SPLIT",
            f"Activity='{a_raw}' has divergent context '{ctx}' vs majority '{majority}' (context_jaccard={j:.3f})",
            "Same activity label appears in clearly different process contexts, suggesting different underlying meanings.",
            errors, confs, tags, evidences, descs
        )

# -----------------------------
# Compose output
# -----------------------------
out = pd.DataFrame({"row_id": work["row_id"].astype(int)})

def types_for(rid):
    if rid not in errors:
        return ""
    # stable order
    order = ["polluted", "distorted", "synonymous", "form-based", "collateral", "homonymous",
             "timestamp_format", "case_missing", "activity_missing"]
    present = [t for t in order if t in errors[rid]] + sorted([t for t in errors[rid] if t not in order])
    return "|".join(present)

def tags_for(rid):
    if rid not in tags:
        return ""
    return "|".join(sorted(tags[rid]))

def evidence_for(rid):
    if rid not in evidences:
        return ""
    # keep short-ish but specific
    ev = evidences[rid]
    return " || ".join(ev[:5]) + (f" || (+{len(ev)-5} more)" if len(ev) > 5 else "")

def desc_for(rid):
    if rid not in descs:
        return ""
    d = descs[rid]
    return " ".join(d[:3]) + (f" (+{len(d)-3} more)" if len(d) > 3 else "")

out["error_flag"] = out["row_id"].map(lambda rid: rid in errors)
out["error_types"] = out["row_id"].map(types_for)
out["error_confidence"] = out["row_id"].map(lambda rid: round(clamp01(confs.get(rid, 0.0)), 3))
out["error_tags"] = out["row_id"].map(tags_for)
out["error_evidence"] = out["row_id"].map(evidence_for)
out["error_description"] = out["row_id"].map(desc_for)

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out.to_csv(OUTPUT_PATH, index=False)

# Print summary to stdout (optional)
type_counts = Counter()
for rid, ts in errors.items():
    for t in ts:
        type_counts[t] += 1

print(f"Wrote: {OUTPUT_PATH}")
print(f"Flagged rows: {out['error_flag'].sum()} / {len(out)} ({out['error_flag'].mean()*100:.2f}%)")
print("Type distribution:", dict(type_counts))