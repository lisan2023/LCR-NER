#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze n-best predictions dumped by predict_nbest().

Reads a test_nbest.json (or any path passed as argv[1]) and prints:
  1. Distribution of `best_path` (which rank wins F1-wise per sample)
  2. Per-rank average Precision / Recall / F1

Usage:
  python analyze_nbest.py                          # default path
  python analyze_nbest.py guwen_outputs/test_nbest.json

Reads the JSON shape produced by run_ner_crf.predict_nbest:
{
  "predictions": [
    {"id": ..., "tokens": ..., "best_path": int, "prfs": [{precision,recall,f1}, ...],
     "paths": [...]}
    ...
  ]
}
"""
import json
import sys
from collections import Counter

# Optional plotting: if matplotlib is available we render the best_path
# distribution as a horizontal bar chart and (optional) donut, saved next to the
# input JSON.
try:
    import matplotlib
    matplotlib.use("Agg")                      # headless backend, no X required
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _parse_bios_entities(tags):
    """Parse a list of BIO(S) tag strings into entity spans using the SAME
    logic as ``run_ner_crf.py`` (via ``processors.utils_ner.get_entity_bios``).

    Returns list of ``(start, end_inclusive, type)`` tuples. Imported lazily
    to avoid a hard dependency on the project's transformers / torch stack
    when this script is used standalone for ad-hoc analysis.
    """
    try:
        from processors.utils_ner import get_entity_bios
        # get_entity_bios takes (seq, id2label) and returns [(type, start, end), ...]
        # Build an id2label identity mapping since our seq is already string labels.
        id2label = {i: t for i, t in enumerate(tags)}
        raw = get_entity_bios(tags, id2label)
        # convert [type, start, end] -> (start, end, type)
        return [(s, e, tp) for tp, s, e in raw]
    except ImportError:
        # Fallback: identical behaviour to get_entity_bios (state machine).
        return _parse_bios_entities_fallback(tags)


def _parse_bios_entities_fallback(tags):
    """Faithful re-implementation of ``get_entity_bios`` for environments where
    ``processors.utils_ner`` cannot be imported (keeps semantics bit-for-bit).
    """
    chunks = []
    chunk = [-1, -1, -1]
    seq = tags
    for indx, tag in enumerate(seq):
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunks.append([tag.split('-')[1], indx, indx])
            chunk = [-1, -1, -1]
            continue
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [tag.split('-')[1], indx, -1]
        elif (tag.startswith("I-") or tag.startswith("E-")) and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    # End-of-sequence flush (mirrors the original `if indx == len(seq)-1` logic)
    if chunk[2] != -1:
        chunks.append(chunk)
    return [(s, e, tp) for tp, s, e in chunks]


def _entities_in_range(entities, start, end):
    """Filter entities that intersect [start, end] (inclusive), clip spans."""
    out = []
    for s, e, t in entities:
        if e < start or s > end:
            continue
        out.append((max(s, start), min(e, end), t))
    return out


def _build_conflict_spans(is_conflict, path_ents, L):
    """Build conflict spans per the paper description:

    1. **Initial fragments**: maximal contiguous runs of positions where
       candidates disagree on tags.
    2. **Extend on both sides**: for each fragment, look at every candidate
       entity that intersects it; expand the fragment's left/right boundaries
       to fully cover those entities (避免实体被截断).
    3. **Merge overlapping / adjacent spans**: extensions of neighbouring
       initial fragments may now overlap — coalesce them into one span.
    """
    # step 1: initial fragments
    initial, i = [], 0
    while i < L:
        if is_conflict[i]:
            j = i
            while j < L and is_conflict[j]:
                j += 1
            initial.append((i, j - 1))       # [i, j-1] inclusive
            i = j
        else:
            i += 1
    if not initial:
        return []

    # step 2: extend each fragment by candidate entity boundaries
    extended = []
    for s, e in initial:
        ext_s, ext_e = s, e
        for ents in path_ents:
            for es, ee, _ in ents:
                if ee < s or es > e:        # no intersection with [s, e]
                    continue
                ext_s = min(ext_s, es)
                ext_e = max(ext_e, ee)
        extended.append((ext_s, ext_e))

    # step 3: merge overlapping or directly-adjacent spans
    extended.sort()
    merged = [extended[0]]
    for s, e in extended[1:]:
        if s <= merged[-1][1] + 1:          # overlap or adjacent
            last_s, last_e = merged[-1]
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def analyze_conflict_units(predictions):
    """Local conflict unit statistics (paper Table 5).

    Implements the paper's pipeline:
      1. Character-level alignment of top-K candidate paths
         (token == char for Chinese, so we work token-level).
      2. Initial conflict fragments at positions where any two candidates
         disagree on the tag.
      3. **Extend** each fragment on both sides to fully cover any
         candidate entity that intersects it (避免实体边界被截断).
      4. **Merge** overlapping / adjacent spans.
      5. Per final span: build candidate solution set; merge identical
         entity sets from different paths into one candidate solution.
      6. **Entity-level** Top-K coverage: for each gold entity inside a
         conflict unit, check whether any of the top-K candidates
         propose an entity with the same (start, end, type). Aggregate
         over all such gold entities.
      7. Judgment accuracy at the entity level: of the gold entities
         covered by ALL candidates, fraction that best_path also covers.

    Metrics:
      - total_units                : # conflict units
      - total_entities             : # gold entities whose start falls in a
                                     conflict unit (coverage denominator)
      - avg_per_sentence           : # units / # sentences
      - avg_length                 : average unit length (tokens)
      - avg_distinct_solutions     : avg # distinct entity solutions per unit
      - coverage_at_k              : list[float], fraction of gold entities
                                     covered when only top-K candidates
                                     are considered
      - judgment_accuracy          : of gold entities covered by ALL candidates,
                                     fraction that best_path also covers
    """
    n = len(predictions)
    nbest = len(predictions[0]["paths"]) if predictions else 0
    total_units        = 0
    total_length       = 0
    total_distinct     = 0
    total_gold_ents    = 0
    n_correct_ents     = 0   # gold ents covered by best_path (= judgment hits)
    n_samples_conflict = 0
    covered_at_k       = [0] * nbest   # covered_at_k[k-1] = # gold ents covered by top-K

    for pred in predictions:
        tokens    = pred["tokens"].split()
        L         = len(tokens)
        path_tags = [p["tags"].split() for p in pred["paths"]]
        gold_tags = pred.get("gold_tags", "").split()
        if len(gold_tags) != L:
            continue                         # safety: skip mismatched samples

        # Per-position conflict flag (any pair of candidates disagree)
        is_conflict = [
            len({tl[pos] for tl in path_tags}) > 1
            for pos in range(L)
        ]

        # Pre-parse entity spans for each path + gold
        path_ents = [_parse_bios_entities(t) for t in path_tags]
        gold_ents = _parse_bios_entities(gold_tags)

        # Build conflict spans with entity-boundary extension + merging
        spans = _build_conflict_spans(is_conflict, path_ents, L)
        if spans:
            n_samples_conflict += 1

        best_idx = pred.get("best_path", 0)

        for start, end in spans:
            total_units  += 1
            total_length += end - start + 1

            # Candidate solutions in this span (per-path entity sets)
            path_sets = [
                frozenset(_entities_in_range(ents, start, end))
                for ents in path_ents
            ]
            # Gold entities that **overlap** the conflict span.
            # Use intersection semantics (e[0] <= end AND e[1] >= start)
            # rather than "start in unit", because gold entities can extend
            # past the unit's left boundary (the extension is by *candidate*
            # entity boundaries, not gold's).
            gold_ents_in_unit = [
                e for e in gold_ents if e[0] <= end and e[1] >= start
            ]
            total_gold_ents += len(gold_ents_in_unit)

            distinct = set(path_sets)
            total_distinct += len(distinct)

            # Entity-level top-K coverage:
            # for each K, count gold entities that appear in at least one
            # of the top-K candidate entity sets.
            for k in range(1, nbest + 1):
                top_k_ents = set()
                for ps in path_sets[:k]:
                    top_k_ents.update(ps)
                for ge in gold_ents_in_unit:
                    if ge in top_k_ents:
                        covered_at_k[k - 1] += 1

            # Judgment (entity-level): does best_path also cover each gold entity?
            best_set = path_sets[best_idx]
            for ge in gold_ents_in_unit:
                if ge in best_set:
                    n_correct_ents += 1

    if total_units == 0 or nbest == 0:
        return dict(
            n=n, total_units=0, total_entities=0, n_samples_conflict=n_samples_conflict,
            avg_per_sentence=0, avg_length=0, avg_distinct_solutions=0,
            n_covered_entities=0, coverage_rate=0, n_correct=0, judgment_accuracy=0,
            coverage_at_k=[0.0] * nbest,
        )
    coverage_at_k_rate = [
        c / total_gold_ents if total_gold_ents else 0.0
        for c in covered_at_k
    ]
    full_covered_ents = covered_at_k[-1]    # covered by all nbest
    return dict(
        n                      = n,
        nbest                  = nbest,
        total_units            = total_units,
        total_entities         = total_gold_ents,
        n_samples_conflict     = n_samples_conflict,
        avg_per_sentence       = total_units / n,
        avg_length             = total_length / total_units,
        avg_distinct_solutions = total_distinct / total_units,
        n_covered_entities     = full_covered_ents,
        coverage_rate          = coverage_at_k_rate[-1],     # = full-set entity coverage
        n_correct              = n_correct_ents,
        judgment_accuracy      = (n_correct_ents / full_covered_ents) if full_covered_ents else 0,
        coverage_at_k          = coverage_at_k_rate,
        covered_at_k_counts    = covered_at_k,
    )


def oracle_at_k_prf(predictions, K):
    """Per sample, pick the path with highest F1 among the first K, then
    macro-average Precision / Recall / F1 across samples.

    Also returns ``rank_counts``: distribution of which rank was chosen
    (e.g., did top-K ever actually pick a candidate other than rank 0?).
    """
    sum_p = sum_r = sum_f1 = 0
    n = 0
    rank_chosen_counts = Counter()

    for pred in predictions:
        prfs = pred.get("prfs", [])
        if not prfs:
            continue
        top_k = prfs[:K]
        if not top_k:
            continue
        # Argmax by f1 (ties -> lowest rank wins)
        best_idx, best_f1 = 0, -1.0
        for i, prf in enumerate(top_k):
            if prf["f1"] > best_f1:
                best_f1, best_idx = prf["f1"], i
        best = top_k[best_idx]
        sum_p   += best["precision"]
        sum_r   += best["recall"]
        sum_f1  += best["f1"]
        n       += 1
        rank_chosen_counts[best_idx] += 1

    if n == 0:
        return {"precision": 0, "recall": 0, "f1": 0,
                "rank_counts": Counter(), "n": 0}
    return {"precision": sum_p / n, "recall": sum_r / n, "f1": sum_f1 / n,
            "rank_counts": rank_chosen_counts, "n": n}


def _print_oracle_at_k_table(predictions, nbest):
    print("=" * 64)
    print("Top-K oracle (MACRO)  -  per sample: pick the candidate with highest F1 in top-K")
    print("  Aggregation: mean(F1_sample) over sentences (every sentence weighted equally)")
    print("  Equivalent: arithmetic mean of per-sentence F1 scores")
    print("=" * 64)
    print(f"  {'K':<6}{'precision':>10}{'recall':>10}{'F1':>10}"
          f"  {'most-picked rank':<22}")
    print("  " + "-" * 60)
    for K in range(1, nbest + 1):
        m = oracle_at_k_prf(predictions, K)
        if m["rank_counts"]:
            mode_rank, mode_cnt = m["rank_counts"].most_common(1)[0]
            most_label = f"rank {mode_rank + 1}  ({mode_cnt}/{m['n']})"
        else:
            most_label = "-"
        print(f"  top-{K:<3}"
              f"{m['precision']:>9.2%}"
              f"{m['recall']:>9.2%}"
              f"{m['f1']:>9.2%}  {most_label:<22}")
    print()
    print("  vs MICRO version (printed next):")
    print("    - MACRO weights each sentence equally -> reflects 'per-user' F1")
    print("    - MICRO weights each ENTITY equally       -> reflects 'per-token' F1")
    print("    - Long sentences (>10 entities) dominate MICRO, not MACRO")
    print("=" * 64)


def micro_oracle_at_k_prf(predictions, K):
    """Micro-averaged entity-level P/R/F1 across the corpus.

    Per sample: pick the candidate with the highest F1 among the first K.
    Then aggregate TP / FP / FN at the entity level (via tag-sequence
    parsing), and derive P/R/F1 from the totals.
    """
    total_tp = total_fp = total_fn = 0
    n = 0

    for pred in predictions:
        gold_tags = pred.get("gold_tags", "").split()
        prfs      = pred.get("prfs", [])
        if not prfs or not gold_tags:
            continue
        top_k = prfs[:K]
        if not top_k:
            continue
        # Argmax by f1 (ties -> lowest rank wins)
        best_idx, best_f1 = 0, -1.0
        for i, prf in enumerate(top_k):
            if prf["f1"] > best_f1:
                best_f1, best_idx = prf["f1"], i
        best_tags = pred["paths"][best_idx]["tags"].split()
        if len(best_tags) != len(gold_tags):
            continue

        gold_set = set(_parse_bios_entities(gold_tags))
        pred_set = set(_parse_bios_entities(best_tags))
        total_tp += len(gold_set & pred_set)
        total_fp += len(pred_set - gold_set)
        total_fn += len(gold_set - pred_set)
        n += 1

    p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f1,
                tp=total_tp, fp=total_fp, fn=total_fn, n=n)


def _print_micro_oracle_table(predictions, nbest):
    print("=" * 80)
    print("Top-K oracle (MICRO)  -  TP/FP/FN accumulated across the corpus")
    print("  Aggregation: sum(TP), sum(FP), sum(FN) across sentences, then F1")
    print("  Equivalent: F1 = 2 * sum(TP) / (sum(TP) + sum(FP) + sum(TP) + sum(FN))")
    print("=" * 80)
    print(f"  {'K':<6}{'precision':>10}{'recall':>10}{'F1':>10}"
          f"  {'TP':>6}{'FP':>6}{'FN':>6}{'N':>6}")
    print("  " + "-" * 72)
    for K in range(1, nbest + 1):
        m = micro_oracle_at_k_prf(predictions, K)
        print(f"  top-{K:<3}"
              f"{m['precision']:>9.2%}"
              f"{m['recall']:>9.2%}"
              f"{m['f1']:>9.2%}"
              f"{m['tp']:>6}"
              f"{m['fp']:>6}"
              f"{m['fn']:>6}"
              f"{m['n']:>6}")
    print()
    print("  Why this differs from MACRO:")
    print("    - MACRO: each sentence has equal weight (F1_sentence averaged)")
    print("    - MICRO: each ENTITY has equal weight  (TP/FP/FN summed first)")
    print("    - On long sentences, MICRO is usually slightly HIGHER than MACRO")
    print("      because long sentences typically have more entities and the model")
    print("      tends to do better on them; aggregating at the entity level")
    print("      amplifies that signal.")
    print("=" * 80)


def micro_oracle_per_conflict_unit(predictions, K):
    """Micro P/R/F1 across all conflict units, picking top-K-best per unit.

    For each conflict unit (built by ``_build_conflict_spans``):
      - pick the candidate path with the highest F1 within the top-K,
      - parse the candidate's tags inside the unit span into an entity set,
      - parse the gold tags inside the unit span into an entity set,
      - accumulate TP / FP / FN over all conflict units.

    This is the entity-level "per-unit oracle" — measures how well
    top-K candidates can jointly cover the gold inside conflict regions.
    """
    total_tp = total_fp = total_fn = 0
    units_count = 0

    for pred in predictions:
        tokens    = pred["tokens"].split()
        L         = len(tokens)
        path_tags = [p["tags"].split() for p in pred["paths"]]
        gold_tags = pred.get("gold_tags", "").split()
        if len(gold_tags) != L:
            continue

        is_conflict = [
            len({tl[pos] for tl in path_tags}) > 1
            for pos in range(L)
        ]
        path_ents = [_parse_bios_entities(t) for t in path_tags]
        gold_ents = _parse_bios_entities(gold_tags)

        spans = _build_conflict_spans(is_conflict, path_ents, L)
        best_idx = pred.get("best_path", 0)
        # Restrict "candidate pool" to top-K
        candidate_indices = list(range(min(K, len(path_tags))))

        for start, end in spans:
            units_count += 1
            # Pick the path in top-K whose entity set inside this unit
            # best matches the gold's (fallback: best_path if none better).
            gold_set = set(_entities_in_range(gold_ents, start, end))
            # Pick the path in top-K whose entity set inside this unit
            # has the most TP (intersection with gold); tie-break by the
            # candidate's overall F1 on the full sentence.
            chosen_idx = best_idx
            best_tp = -1
            best_f1_fallback = -1.0
            for i in candidate_indices:
                cand_set = set(_entities_in_range(path_ents[i], start, end))
                tp = len(cand_set & gold_set)
                f1_i = pred["prfs"][i]["f1"] if i < len(pred["prfs"]) else -1.0
                if tp > best_tp or (tp == best_tp and f1_i > best_f1_fallback):
                    best_tp = tp
                    best_f1_fallback = f1_i
                    chosen_idx = i

            pred_set = set(_entities_in_range(path_ents[chosen_idx], start, end))
            total_tp += len(pred_set & gold_set)
            total_fp += len(pred_set - gold_set)
            total_fn += len(gold_set - pred_set)

    p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f1,
                tp=total_tp, fp=total_fp, fn=total_fn,
                n_units=units_count)


def _print_micro_per_conflict_unit_table(predictions, nbest):
    """Per-conflict-unit oracle (MICRO P/R/F1, pick top-K best inside each unit)."""
    print("=" * 84)
    print("Per-conflict-unit oracle  (MICRO P/R/F1, pick top-K best inside each unit)")
    print("=" * 84)
    print(f"  {'K':<6}{'precision':>10}{'recall':>10}{'F1':>10}"
          f"  {'TP':>6}{'FP':>6}{'FN':>6}{'#units':>8}")
    print("  " + "-" * 76)
    for K in range(1, nbest + 1):
        m = micro_oracle_per_conflict_unit(predictions, K)
        print(f"  top-{K:<3}"
              f"{m['precision']:>9.2%}"
              f"{m['recall']:>9.2%}"
              f"{m['f1']:>9.2%}"
              f"{m['tp']:>6}"
              f"{m['fp']:>6}"
              f"{m['fn']:>6}"
              f"{m['n_units']:>8}")
    print("=" * 84)


def _build_unit_oracle_choice(predictions, K):
    """For each conflict unit, decide which candidate (in top-K) is the
    *oracle* — i.e., whose entity set inside this unit best matches gold.

    Selection rule:
      1. maximise TP overlap with the unit's gold_set
      2. tie-break by the candidate's full-sentence F1

    Returns a per-sample list of dicts:
        { start, end, oracle_idx, gold_set, cand_sets, candidate_indices,
          wrong_indices }
    where ``cand_sets`` is the list of per-candidate entity sets (each is a
    set of ``(s, e, type)`` tuples clipped to [start, end]).

    """
    samples = []
    for pred in predictions:
        tokens    = pred["tokens"].split()
        L         = len(tokens)
        path_tags = [p["tags"].split() for p in pred["paths"]]
        gold_tags = pred.get("gold_tags", "").split()
        if len(gold_tags) != L:
            samples.append(None); continue

        is_conflict = [
            len({tl[pos] for tl in path_tags}) > 1
            for pos in range(L)
        ]
        path_ents = [_parse_bios_entities(t) for t in path_tags]
        gold_ents = _parse_bios_entities(gold_tags)
        spans = _build_conflict_spans(is_conflict, path_ents, L)

        best_idx = pred.get("best_path", 0)
        candidate_indices = list(range(min(K, len(path_tags))))

        unit_records = []
        for s, e in spans:
            gold_set = set(_entities_in_range(gold_ents, s, e))
            cand_sets = [
                set(_entities_in_range(path_ents[i], s, e))
                for i in candidate_indices
            ]
            oracle_idx = best_idx
            best_tp = -1
            best_f1 = -1.0
            for ci, i in enumerate(candidate_indices):
                cand_set = cand_sets[ci]
                tp = len(cand_set & gold_set)
                f1_i = pred["prfs"][i]["f1"] if i < len(pred["prfs"]) else -1.0
                if tp > best_tp or (tp == best_tp and f1_i > best_f1):
                    best_tp = tp
                    best_f1 = f1_i
                    oracle_idx = i
            wrong_indices = [
                candidate_indices[ci]
                for ci, cs in enumerate(cand_sets) if cs != gold_set
            ]
            unit_records.append({
                "start":     s,
                "end":       e,
                "oracle_idx": oracle_idx,
                "gold_set":  gold_set,
                "cand_sets": cand_sets,
                "candidate_indices": candidate_indices,
                "wrong_indices":     wrong_indices,
            })
        samples.append(unit_records)
    return samples


def _aggregate_with_judge(predictions, samples, choose_unit):
    """Aggregate corpus-level micro TP/FP/FN with a per-unit ``choose_unit(u)``
    function that returns the candidate index picked by the "judge".

    Non-conflict positions in every sentence use best_path tags (deterministic).
    The judge only chooses inside conflict units.
    """
    tp = fp = fn = 0
    for pred, units in zip(predictions, samples):
        if units is None:
            continue
        gold_tags = pred.get("gold_tags", "").split()
        if not gold_tags:
            continue
        path_tags = [p["tags"].split() for p in pred["paths"]]
        best_idx  = pred.get("best_path", 0)
        L = len(gold_tags)

        # In-unit mask
        in_unit = [False] * L
        for u in units:
            for p in range(u["start"], u["end"] + 1):
                in_unit[p] = True

        merged = list(path_tags[best_idx])
        for u in units:
            i = choose_unit(u)
            chosen_tags = path_tags[i]
            # chosen_tags is the FULL candidate tag sequence aligned with the
            # sentence, so its index p already corresponds to position p of
            # the original sentence. Don't subtract u["start"].
            for p in range(u["start"], u["end"] + 1):
                merged[p] = chosen_tags[p]

        pred_ents = _parse_bios_entities(merged)
        gold_ents = _parse_bios_entities(gold_tags)
        pred_set, gold_set = set(pred_ents), set(gold_ents)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    return tp, fp, fn


def micro_full_with_topk_in_conflicts(predictions, K):
    """Oracle baseline: pick the top-K-best candidate inside each conflict unit,
    use best_path outside. Equivalent to LLM judge with 100% accuracy.

    """
    samples = _build_unit_oracle_choice(predictions, K)
    tp, fp, fn = _aggregate_with_judge(
        predictions, samples,
        choose_unit=lambda u: u["oracle_idx"],
    )
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=p, recall=r, f1=f1,
                tp=tp, fp=fp, fn=fn)


def _print_micro_full_with_topk_in_conflicts_table(predictions, nbest):
    print("=" * 88)
    print("Full-sentence micro P/R/F1, with top-K oracle applied inside conflict units")
    print("  (non-conflict positions use best_path unchanged)")
    print("=" * 88)
    print(f"  {'K':<6}{'precision':>10}{'recall':>10}{'F1':>10}"
          f"  {'TP':>6}{'FP':>6}{'FN':>6}{'#units':>8}")
    print("  " + "-" * 80)
    for K in range(1, nbest + 1):
        m = micro_full_with_topk_in_conflicts(predictions, K)
        print(f"  top-{K:<3}"
              f"{m['precision']:>9.2%}"
              f"{m['recall']:>9.2%}"
              f"{m['f1']:>9.2%}"
              f"{m['tp']:>6}"
              f"{m['fp']:>6}"
              f"{m['fn']:>6}"
              f"{m.get('n_units', 0):>8}")
    print("=" * 88)




def _print_conflict_table(m):
    print("=" * 60)
    print("Table 5.  Local conflict units and adjudication (entity-level)")
    print("=" * 60)
    print(f"  {'Metric':<32}{'Value':>22}")
    print("  " + "-" * 54)
    print(f"  {'# local conflict units':<32}{m['total_units']:>22}")
    print(f"  {'# gold entities in conflict units':<32}{m['total_entities']:>22}")
    print(f"  {'# samples with at least one unit':<32}{m['n_samples_conflict']:>22}")
    print(f"  {'Avg units per sentence':<32}{m['avg_per_sentence']:>22.3f}")
    print(f"  {'Avg conflict fragment length':<32}{m['avg_length']:>19.2f} tokens")
    print(f"  {'Avg # distinct candidate solutions':<32}{m['avg_distinct_solutions']:>22.3f}")
    print(f"  {'# gold ents covered by best_path':<32}{m['n_correct']:>22}")
    print(f"  {'Judgment accuracy (entity-level)':<32}{100*m['judgment_accuracy']:>20.2f} %")
    print("=" * 60)
    # Top-K coverage rate at the entity level
    cov_at_k     = m.get("coverage_at_k") or []
    counts_at_k  = m.get("covered_at_k_counts") or []
    total_ents   = m.get("total_entities", 0)
    if cov_at_k:
        print()
        print("  Candidate-set coverage rate by Top-K  (entity-level)")
        print("  " + "-" * 54)
        for k, rate in enumerate(cov_at_k, start=1):
            cnt = counts_at_k[k - 1] if k - 1 < len(counts_at_k) else 0
            bar_len = int(round(rate * 30))
            bar = "#" * bar_len
            label = f"top-{k}"
            print(f"    {label:<7}{100*rate:>6.2f}%   ({cnt:>4}/{total_ents})  {bar}")
    print("=" * 60)


def _plot_best_path(paths_json_path, predictions, nbest):
    """Render the best_path distribution as a horizontal bar chart + donut.

    Files are saved next to ``paths_json_path`` with suffixes
    ``_best_path_bar.png`` and ``_best_path_pie.png``. Bar chart is the
    primary view; donut is provided for visual comparison.
    """
    if not _HAS_MPL:
        print(f"  (matplotlib not installed -> skipped chart for {paths_json_path})")
        return

    counter    = Counter(p["best_path"] for p in predictions)
    ranks      = list(range(nbest))
    counts     = [counter.get(r, 0) for r in ranks]
    total      = sum(counts)
    percents   = [100 * c / total if total else 0 for c in counts]
    labels     = [f"rank {r + 1}" for r in ranks]   # display 1..N (data still 0..N-1)
    # rank 0 dark, others faded
    color_main = "#4C72B0"
    palette    = [color_main] + ["#C8D4E3"] * (nbest - 1)

    base = paths_json_path.rsplit(".", 1)[0]

    # ---- horizontal bar ----
    fig, ax = plt.subplots(figsize=(7, max(3, nbest * 0.45 + 1)))
    bars = ax.barh(labels, percents, color=palette, edgecolor="white")
    ax.invert_yaxis()                              # rank 0 on top
    ax.set_xlabel("Share of samples (%)")
    ax.set_title(f"best_path distribution  (n={total})")
    ax.set_xlim(0, max(percents) * 1.15 + 1)
    for bar, pct, c in zip(bars, percents, counts):
        ax.text(bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"  {pct:.1f}%  (n={c})",
                va="center", fontsize=9, color="#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_bar = base + "_best_path_bar.png"
    fig.savefig(out_bar, dpi=130)
    plt.close(fig)

    # ---- donut (companion) ----
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    ax2.pie(
        counts,
        labels=[f"rank {r + 1}\n({p:.1f}%)" for r, p in enumerate(percents)],
        colors=palette,
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    ax2.set_title(f"best_path distribution  (n={total})")
    out_pie = base + "_best_path_pie.png"
    fig2.savefig(out_pie, dpi=130)
    plt.close(fig2)

    print()
    print("Charts saved:")
    print(f"  {out_bar}")
    print(f"  {out_pie}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "guwen_outputs/bert/test_nbest.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    n = len(predictions)
    if n == 0:
        print("No predictions in input.")
        return

    nbest = len(predictions[0].get("prfs", []))
    print(f"Input     : {path}")
    print(f"Samples   : {n}")
    print(f"Nbest     : {nbest}")
    print(f"Task      : {data.get('task_name', '?')}")
    print(f"Model     : {data.get('model_name', '?')}")
    print()

    # ---------- best_path 分布 ----------
    best_paths = [p["best_path"] for p in predictions]
    counter = Counter(best_paths)
    print("=" * 56)
    print("best_path distribution  (rank chosen by max F1 per sample)")
    print("=" * 56)
    print(f"  {'rank':<6}{'count':>8}{'percent':>10}    bar")
    print("  " + "-" * 50)
    for rank in range(nbest):
        c = counter.get(rank, 0)
        pct = 100 * c / n if n else 0
        bar = "#" * int(round(pct / 2))
        marker = "  <-- top-1 by F1" if rank == 0 else ""
        print(f"  {rank:<6}{c:>8}{pct:>9.1f}%   {bar}{marker}")
    print("  " + "-" * 50)
    nonzero = sum(1 for v in counter.values() if v > 0)
    print(f"  ({nonzero}/{nbest} ranks actually carry samples with F1>0)")
    print()

    # ---------- 每条路径平均 P/R/F1 ----------
    print("=" * 56)
    print("Per-rank average Precision / Recall / F1")
    print("=" * 56)
    print(f"  {'rank':<6}{'avg P':>9}{'avg R':>9}{'avg F1':>10}{'count':>8}")
    print("  " + "-" * 44)
    sums = {"p": [0.0] * nbest, "r": [0.0] * nbest, "f1": [0.0] * nbest}
    cnt  = [0] * nbest
    for p in predictions:
        for r, prf in enumerate(p.get("prfs", [])):
            if r >= nbest:
                break
            sums["p"][r]   += prf["precision"]
            sums["r"][r]   += prf["recall"]
            sums["f1"][r]  += prf["f1"]
            cnt[r]        += 1
    for r in range(nbest):
        if cnt[r] == 0:
            print(f"  {r:<6}{'-':>9}{'-':>9}{'-':>10}{0:>8}")
            continue
        print(f"  {r:<6}"
              f"{sums['p'][r]/cnt[r]:>8.2%}"
              f"{sums['r'][r]/cnt[r]:>8.2%}"
              f"{sums['f1'][r]/cnt[r]:>9.2%}"
              f"{cnt[r]:>8}")
    print()

    # ---------- Top-1 vs Oracle ----------
    print("=" * 56)
    print("Top-1  vs  Oracle  (best-of-{nbest})".format(nbest=nbest))
    print("=" * 56)
    top1  = [p["prfs"][0]["f1"]                       for p in predictions]
    oracle = [max(p["prfs"][r]["f1"]
                    for r in range(len(p["prfs"])))   for p in predictions]
    avg_top1   = sum(top1)   / n
    avg_oracle = sum(oracle) / n
    print(f"  rank-0 average F1:          {avg_top1:.4f}")
    print(f"  best-of-{nbest} average F1: {avg_oracle:.4f}")
    print(f"  nbest ceiling lift:         {avg_oracle - avg_top1:+.4f}")
    print()
    n_perfect_top1  = sum(1 for v in top1   if v >= 1 - 1e-9)
    n_perfect_any   = sum(1 for v in oracle if v >= 1 - 1e-9)
    print(f"  F1=1.0 on rank-0:           {n_perfect_top1}/{n}"
          f"  ({100*n_perfect_top1/n:.1f}%)")
    print(f"  F1=1.0 somewhere in top-{nbest}:"
          f" {n_perfect_any}/{n}"
          f"  ({100*n_perfect_any/n:.1f}%)")

    # ---------- 分布图 ----------
    _plot_best_path(path, predictions, nbest)

    # ---------- Top-K oracle (per sample: pick best-F1 path in top-K) ----------
    _print_oracle_at_k_table(predictions, nbest)

    # ---------- Top-K oracle 微观统计 ----------
    _print_micro_oracle_table(predictions, nbest)

    # ---------- 每个冲突单元的 Top-K 微观 oracle ----------
    _print_micro_per_conflict_unit_table(predictions, nbest)

    # ---------- 整句级别(non-conflict 用 best_path,conflict 用 top-K oracle) ----------
    _print_micro_full_with_topk_in_conflicts_table(predictions, nbest)


    # ---------- 表5:局部冲突单元与裁决效果 ----------
    conflict_metrics = analyze_conflict_units(predictions)
    if conflict_metrics["total_units"] > 0:
        _print_conflict_table(conflict_metrics)
    else:
        print("(No conflict units detected; check that gold_tags is in the JSON.)")


if __name__ == "__main__":
    main()
