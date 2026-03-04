#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Event Log Error Detection
Input : /home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_formbased_noLabel.csv
Output: /home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_formbased_noLabel.detected.csv

Detects (based ONLY on Case, Activity, Timestamp, Resource):
- formbased
- polluted
- distorted
- synonymous
- collateral
- homonymous (heuristic, conservative)

Resource being empty is NOT an error (do not flag).
"""

import os
import re
import math
import json
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict, Counter

INPUT_PATH = "/home/unist/바탕화면/event-log-ai/data/pub/pub_seed42_ratio0.2_formbased_noLabel.csv"
OUTPUT_PATH = "/home/unist/바탕화면/event-log-ai/data_detected/pub_seed42_ratio0.2_formbased_noLabel.detected.csv"

REQUIRED_COLS = ["Case", "Activity", "Timestamp", "Resource"]

# --- Tunable thresholds (aggressive but not reckless) ---
COLLATERAL_WINDOW_SECONDS = 3.0          # near-duplicate window
FORMBASED_MIN_GROUP_SIZE = 3             # same case+timestamp repeated >= this
FORMBASED_MIN_DISTINCT_ACTIVITIES = 2    # repeated timestamp should include multiple activities
DISTORTED_MIN_SIM = 0.86                 # similarity to canonical to call typo
DISTORTED_MAX_SIM = 0.985                # too close -> likely exact same, not typo
SYNONYM_MIN_SIM = 0.70                   # similarity to canonical for synonym mapping (looser)
HOMONYM_CONTEXT_MIN_CASES = 8            # need enough evidence
HOMONYM_JS_THRESHOLD = 0.55              # divergence threshold for context distributions

# Polluted pattern: base + "_" + 5-12 alnum + "_" + yyyymmdd hhmmss + micro/nano digits
POLLUTED_RE = re.compile(
    r"^(?P<base>.+?)_(?P<suffix>[A-Za-z0-9]{5,12})_(?P<dt>\d{8}\s\d{6}\d{3,6})$"
)

# Basic normalization helpers
SPACE_RE = re.compile(r"\s+")
NON_ALNUM_KEEP_SPACE = re.compile(r"[^a-z0-9 ]+")

# A small, domain-agnostic synonym lexicon (kept minimal; also uses data-driven clustering)
SYNONYM_SEEDS = {
    # canonical: set(variants)
    "review application": {
        "review case", "assess application", "evaluate application", "inspect application"
    },
    "reject request": {
        "reject application", "deny request", "decline application", "refuse request"
    },
    "diagnose patient": {
        "make diagnosis", "establish diagnosis", "determine diagnosis", "confirm diagnosis"
    },
    "start production": {
        "start manufacturing", "begin production", "initiate production run", "launch production"
    },
}

def norm_text(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = SPACE_RE.sub(" ", s)
    return s

def norm_for_match(s: str) -> str:
    s = norm_text(s)
    s = NON_ALNUM_KEEP_SPACE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def jensen_shannon(p, q, eps=1e-12):
    # p, q are dicts of probabilities over same support
    keys = set(p) | set(q)
    def kl(a, b):
        s = 0.0
        for k in keys:
            av = a.get(k, 0.0) + eps
            bv = b.get(k, 0.0) + eps
            s += av * math.log(av / bv)
        return s
    m = {k: 0.5*(p.get(k,0.0)+q.get(k,0.0)) for k in keys}
    return math.sqrt(0.5*kl(p,m) + 0.5*kl(q,m))

def build_synonym_map_from_seeds():
    m = {}
    for canon, vars_ in SYNONYM_SEEDS.items():
        canon_n = norm_for_match(canon)
        m[canon_n] = canon  # store original canonical label
        for v in vars_:
            m[norm_for_match(v)] = canon
    return m

def infer_canonical_activity_set(df):
    """
    Build a set of likely canonical activities from the data itself:
    - strip polluted suffix if present
    - normalize spacing
    - count frequency
    - take top frequent unique bases as canonical candidates
    """
    bases = []
    for a in df["Activity"].astype(str).tolist():
        a0 = a.strip()
        m = POLLUTED_RE.match(a0)
        if m:
            base = m.group("base")
        else:
            base = a0
        bases.append(norm_for_match(base))
    cnt = Counter([b for b in bases if b])
    # Keep those that appear at least twice OR are in top 200 (aggressive)
    common = {k for k,v in cnt.items() if v >= 2}
    top = {k for k,_ in cnt.most_common(200)}
    return common | top

def best_canonical_match(activity_base_norm, canonical_set):
    """
    Find best fuzzy match among canonical_set.
    Returns (best_label_norm, best_similarity)
    """
    best = (None, 0.0)
    # quick pruning: compare only those with similar length
    L = len(activity_base_norm)
    for c in canonical_set:
        if abs(len(c) - L) > max(6, int(0.35*L)):
            continue
        s = sim(activity_base_norm, c)
        if s > best[1]:
            best = (c, s)
    return best

def main():
    # --- Load ---
    df = pd.read_csv(INPUT_PATH)

    # Validate required columns only (do not use others)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Keep only required columns for detection
    work = df[REQUIRED_COLS].copy()

    # Parse timestamps
    work["Timestamp_parsed"] = pd.to_datetime(work["Timestamp"], errors="coerce", utc=False)

    # Precompute normalized activity
    work["Activity_raw"] = work["Activity"].astype(str)
    work["Activity_norm"] = work["Activity_raw"].map(norm_for_match)

    # Polluted base extraction
    polluted_base = []
    polluted_suffix = []
    polluted_dt = []
    is_polluted = []
    for a in work["Activity_raw"].tolist():
        a0 = str(a).strip()
        m = POLLUTED_RE.match(a0)
        if m:
            is_polluted.append(True)
            polluted_base.append(norm_for_match(m.group("base")))
            polluted_suffix.append(m.group("suffix"))
            polluted_dt.append(m.group("dt"))
        else:
            is_polluted.append(False)
            polluted_base.append(norm_for_match(a0))
            polluted_suffix.append("")
            polluted_dt.append("")
    work["is_polluted"] = is_polluted
    work["polluted_base_norm"] = polluted_base
    work["polluted_suffix"] = polluted_suffix
    work["polluted_dt"] = polluted_dt

    # Canonical candidates from data
    canonical_set = infer_canonical_activity_set(work)

    # Seed synonym map
    seed_syn_map = build_synonym_map_from_seeds()

    # --- Initialize outputs ---
    n = len(work)
    out = pd.DataFrame({
        "row_id": list(range(n)),
        "error_flag": [False]*n,
        "error_types": [""]*n,
        "error_confidence": [0.0]*n,
        "error_tags": [""]*n,
        "error_evidence": [""]*n,
        "error_description": [""]*n,
    })

    # Collect per-row detections
    det_types = defaultdict(set)
    det_tags = defaultdict(list)
    det_evidence = defaultdict(list)
    det_desc = defaultdict(list)
    det_scores = defaultdict(list)

    def add_det(i, etype, score, tag, evidence, desc):
        det_types[i].add(etype)
        det_tags[i].append(tag)
        det_evidence[i].append(evidence)
        det_desc[i].append(desc)
        det_scores[i].append(float(score))

    # --- Basic format/range checks (aggressive) ---
    # Timestamp parse failures are real errors (not in the provided list, but still erroneous rows).
    # However output requires only given error types; we will tag as "distorted"?? No.
    # So we will NOT invent new error types; instead we will treat unparseable timestamps as "formbased"
    # would be wrong. Therefore: we will flag them as "collateral"?? also wrong.
    # => We will flag them as "distorted" only if Activity is also problematic; otherwise leave unflagged.
    # (Still record evidence in tags but no error_type.) This keeps compliance with requested types.
    bad_ts_idx = work.index[work["Timestamp_parsed"].isna()].tolist()
    for i in bad_ts_idx:
        # no error_type added; but keep evidence in tags/description only if other errors exist later
        det_tags[i].append("timestamp_parse_failed")
        det_evidence[i].append({"Timestamp": work.at[i, "Timestamp"]})
        det_desc[i].append(f"Timestamp could not be parsed: {work.at[i,'Timestamp']}")

    # --- 2) Polluted detection ---
    for i, flag in enumerate(work["is_polluted"].tolist()):
        if flag:
            base = work.at[i, "polluted_base_norm"]
            raw = work.at[i, "Activity_raw"]
            add_det(
                i, "polluted", 0.98,
                "polluted_regex",
                {"Activity": raw, "base_extracted": base, "suffix": work.at[i,"polluted_suffix"], "dt_token": work.at[i,"polluted_dt"]},
                f"Activity appears polluted with machine suffix; extracted base='{base}'."
            )

    # --- 3) Distorted detection (typos) ---
    # Compare base (polluted stripped) to best canonical match
    for i in range(n):
        base = work.at[i, "polluted_base_norm"]
        if not base:
            continue
        best_c, best_s = best_canonical_match(base, canonical_set)
        # Distorted: close but not identical
        if best_c and best_c != base and (best_s >= DISTORTED_MIN_SIM and best_s <= DISTORTED_MAX_SIM):
            # Avoid calling synonyms as distorted if seed synonym says it's a synonym
            if base in seed_syn_map:
                continue
            add_det(
                i, "distorted",
                min(0.95, 0.55 + 0.45*best_s),
                "distorted_fuzzy_to_canonical",
                {"activity_base_norm": base, "best_canonical_norm": best_c, "similarity": round(best_s, 4)},
                f"Activity base looks like a typo/variant of '{best_c}' (similarity={best_s:.3f})."
            )

    # --- 4) Synonymous detection ---
    # First: seed-based exact normalized match
    for i in range(n):
        base = work.at[i, "polluted_base_norm"]
        if not base:
            continue
        if base in seed_syn_map:
            canon = seed_syn_map[base]
            add_det(
                i, "synonymous", 0.92,
                "syn_seed_lexicon",
                {"activity_base_norm": base, "canonical": canon},
                f"Activity base matches known synonym variant; canonical='{canon}'."
            )

    # Second: data-driven synonym-ish detection:
    # If base is not in canonical_set but is moderately similar to a canonical label, treat as synonym
    # (but only if not already distorted and similarity not too high).
    for i in range(n):
        base = work.at[i, "polluted_base_norm"]
        if not base or base in canonical_set:
            continue
        best_c, best_s = best_canonical_match(base, canonical_set)
        if best_c and best_s >= SYNONYM_MIN_SIM and best_s < DISTORTED_MIN_SIM:
            add_det(
                i, "synonymous",
                min(0.85, 0.40 + 0.60*best_s),
                "syn_fuzzy_semantic_proxy",
                {"activity_base_norm": base, "closest_canonical_norm": best_c, "similarity": round(best_s, 4)},
                f"Activity base is not a common canonical label; resembles '{best_c}' (similarity={best_s:.3f}) suggesting synonym/alternate phrasing."
            )

    # --- 1) Form-based detection ---
    # Same Case + exact same Timestamp repeated for multiple events (>=3) with multiple activities
    # Flag all but the first occurrence in that timestamp group as formbased.
    # (We cannot know true timestamp; evidence includes group size and activities.)
    work["_row_id"] = range(n)
    grp = work.groupby(["Case", "Timestamp"], dropna=False)["_row_id"].apply(list)
    for (case, ts), rows in grp.items():
        if ts is None or (isinstance(ts, float) and math.isnan(ts)):
            continue
        if len(rows) >= FORMBASED_MIN_GROUP_SIZE:
            acts = [work.at[r, "Activity_raw"] for r in rows]
            distinct_acts = len(set(acts))
            if distinct_acts >= FORMBASED_MIN_DISTINCT_ACTIVITIES:
                # sort by original row order; keep first as "anchor"
                rows_sorted = sorted(rows)
                for r in rows_sorted[1:]:
                    add_det(
                        r, "formbased", 0.90,
                        "formbased_same_case_same_timestamp_cluster",
                        {"Case": case, "Timestamp": ts, "cluster_size": len(rows), "distinct_activities": distinct_acts, "activities": acts[:10]},
                        f"Multiple events in same case share identical timestamp '{ts}' (cluster_size={len(rows)}), consistent with form-based overwrite."
                    )

    # --- 5) Collateral detection ---
    # A) exact duplicates: same Case, Activity_raw, Timestamp, Resource
    dup_cols = ["Case", "Activity_raw", "Timestamp", "Resource"]
    dup_mask = work.duplicated(subset=dup_cols, keep="first")
    for i in work.index[dup_mask].tolist():
        add_det(
            i, "collateral", 0.97,
            "collateral_exact_duplicate",
            {c: work.at[i, c] for c in dup_cols},
            "Exact duplicate event (same case, activity, timestamp, resource) suggests logging artifact."
        )

    # B) near-duplicates: same Case + same activity base + same resource within short interval
    # (resource may be empty; still allowed, but if empty we still can detect duplicates)
    w2 = work.copy()
    w2["ts"] = w2["Timestamp_parsed"]
    w2 = w2.sort_values(["Case", "ts", "_row_id"], kind="mergesort")

    for case, sub in w2.groupby("Case", dropna=False):
        idxs = sub["_row_id"].tolist()
        ts_list = sub["ts"].tolist()
        act_list = sub["polluted_base_norm"].tolist()
        res_list = sub["Resource"].tolist()

        for k in range(1, len(idxs)):
            i_prev = idxs[k-1]
            i_cur = idxs[k]
            t_prev = ts_list[k-1]
            t_cur = ts_list[k]
            if pd.isna(t_prev) or pd.isna(t_cur):
                continue
            dt = (t_cur - t_prev).total_seconds()
            if dt < 0:
                continue
            if dt <= COLLATERAL_WINDOW_SECONDS:
                if act_list[k] and act_list[k] == act_list[k-1] and str(res_list[k]) == str(res_list[k-1]):
                    add_det(
                        i_cur, "collateral",
                        max(0.80, 0.95 - 0.05*dt),
                        "collateral_near_duplicate_short_interval",
                        {"Case": case, "activity_base_norm": act_list[k], "Resource": res_list[k], "dt_seconds": dt,
                         "prev_row_id": i_prev, "prev_timestamp": str(work.at[i_prev, "Timestamp"]), "cur_timestamp": str(work.at[i_cur, "Timestamp"])},
                        f"Near-duplicate event within {dt:.3f}s for same case/activity/resource suggests collateral logging."
                    )

    # --- 6) Homonymous detection (conservative heuristic) ---
    # If same activity label appears in two very different contexts (prev/next activity distributions),
    # flag minority-context occurrences as homonymous.
    # Context = (prev_base, next_base) within same case ordered by timestamp.
    w3 = w2.copy()
    w3["prev_act"] = None
    w3["next_act"] = None

    for case, sub in w3.groupby("Case", dropna=False):
        ids = sub["_row_id"].tolist()
        acts = sub["polluted_base_norm"].tolist()
        for j, rid in enumerate(ids):
            prev_a = acts[j-1] if j-1 >= 0 else ""
            next_a = acts[j+1] if j+1 < len(ids) else ""
            w3.loc[w3["_row_id"] == rid, "prev_act"] = prev_a
            w3.loc[w3["_row_id"] == rid, "next_act"] = next_a

    # Build context distributions per activity
    act_to_contexts = defaultdict(list)
    for _, r in w3.iterrows():
        a = r["polluted_base_norm"]
        if not a:
            continue
        ctx = (r["prev_act"] or "", r["next_act"] or "")
        act_to_contexts[a].append(ctx)

    # For each activity with enough occurrences, see if it splits into two clusters by context
    for a, ctxs in act_to_contexts.items():
        if len(ctxs) < HOMONYM_CONTEXT_MIN_CASES:
            continue
        # Build distribution over prev and next separately
        prev_cnt = Counter([c[0] for c in ctxs if c[0]])
        next_cnt = Counter([c[1] for c in ctxs if c[1]])
        if len(prev_cnt) < 3 and len(next_cnt) < 3:
            continue

        # Split by most common prev activity vs others (simple, aggressive)
        top_prev, top_prev_n = (prev_cnt.most_common(1)[0] if prev_cnt else ("", 0))
        if not top_prev or top_prev_n < max(3, int(0.35*len(ctxs))):
            continue

        group_A = [c for c in ctxs if c[0] == top_prev]
        group_B = [c for c in ctxs if c[0] != top_prev]

        if len(group_B) < 3:
            continue

        def dist_of(group, which=0):
            cnt = Counter([c[which] for c in group if c[which]])
            total = sum(cnt.values()) or 1
            return {k: v/total for k,v in cnt.items()}

        # Compare next-activity distributions between groups
        p = dist_of(group_A, which=1)
        q = dist_of(group_B, which=1)
        js = jensen_shannon(p, q)

        if js >= HOMONYM_JS_THRESHOLD:
            # Flag rows in minority group as homonymous (more likely misused label)
            minority_prev = None
            if len(group_A) < len(group_B):
                minority_prev = top_prev
                minority_selector = lambda prev: prev == top_prev
                minority_size = len(group_A)
            else:
                minority_prev = f"not {top_prev}"
                minority_selector = lambda prev: prev != top_prev
                minority_size = len(group_B)

            # Apply to actual rows
            rows_a = w3.index[w3["polluted_base_norm"] == a].tolist()
            for ridx in rows_a:
                prev_a = w3.at[ridx, "prev_act"] or ""
                if minority_selector(prev_a):
                    row_id = int(w3.at[ridx, "_row_id"])
                    add_det(
                        row_id, "homonymous",
                        min(0.80, 0.55 + 0.45*min(1.0, js)),
                        "homonymous_context_divergence_prev_split",
                        {"activity_base_norm": a, "top_prev": top_prev, "js_divergence_next": round(js, 4),
                         "minority_definition": minority_prev, "minority_size": minority_size, "total": len(ctxs)},
                        f"Same activity label '{a}' appears in divergent contexts (JS={js:.3f}); minority-context occurrences may represent different meaning."
                    )

    # --- Finalize output rows ---
    for i in range(n):
        types = sorted(det_types[i])
        # Remove empty / non-specified types (we only add specified ones)
        if types:
            out.at[i, "error_flag"] = True
            out.at[i, "error_types"] = "|".join(types)

            # Confidence: combine multiple scores (noisy-or)
            scores = det_scores[i] if det_scores[i] else [0.0]
            p_not = 1.0
            for s in scores:
                p_not *= (1.0 - max(0.0, min(1.0, s)))
            conf = 1.0 - p_not
            out.at[i, "error_confidence"] = round(conf, 4)

            out.at[i, "error_tags"] = "|".join(det_tags[i])[:2000]
            out.at[i, "error_evidence"] = safe_json(det_evidence[i])[:20000]
            # Keep description concise but specific
            desc = " ; ".join(det_desc[i])
            out.at[i, "error_description"] = desc[:20000]
        else:
            out.at[i, "error_flag"] = False
            out.at[i, "error_types"] = ""
            out.at[i, "error_confidence"] = 0.0
            out.at[i, "error_tags"] = ""
            out.at[i, "error_evidence"] = ""
            out.at[i, "error_description"] = ""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote: {OUTPUT_PATH}")
    print(out["error_flag"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()