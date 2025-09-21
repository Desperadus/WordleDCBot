#!/usr/bin/python
import os
import itertools
from collections import Counter, defaultdict
from math import log2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Files:
DICT_FILE_all = 'all_words.txt'   # all allowed guesses (for scoring)
DICT_FILE = 'words.txt'       # official solution list (remaining candidates)

# ---------- Pattern helpers (fast, compact) ----------

POW3 = (1, 3, 9, 27, 81)  # for base-3 encoding of 5 trits


def encode_pattern(trits):
    """(t0..t4) with values in {0,1,2} -> int in [0..242]."""
    return trits[0]*POW3[0] + trits[1]*POW3[1] + trits[2]*POW3[2] + trits[3]*POW3[3] + trits[4]*POW3[4]


def calculate_pattern_id(guess, true):
    """
    Return Wordle feedback as a compact int (0..242).
    2 pass algo, 26-letter counts (a-z), faster than Counter.
    """
    # First pass: greens
    pattern = [0]*5
    rem = [0]*26
    for i, (g, t) in enumerate(zip(guess, true)):
        if g == t:
            pattern[i] = 2
        else:
            rem[ord(t) - 97] += 1

    # Second pass: yellows/greys
    for i, g in enumerate(guess):
        if pattern[i] == 0:
            idx = ord(g) - 97
            if rem[idx] > 0:
                pattern[i] = 1
                rem[idx] -= 1
            else:
                pattern[i] = 0
    return encode_pattern(pattern)

# ---------- IO ----------


def load_wordlists():
    with open(DICT_FILE_all) as f:
        all_dictionary = [w.strip() for w in f if w.strip()]
    with open(DICT_FILE) as f:
        solution_list = [w.strip() for w in f if w.strip()]
    lens = {len(w) for w in all_dictionary}
    assert len(lens) == 1, "Dictionary contains different length words."
    word_len = next(iter(lens))
    return all_dictionary, set(solution_list), word_len

# ---------- Entropy (parallel) ----------


def _entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    # scipy.stats.entropy would be fine, but avoid heavy deps: use base-2 Shannon entropy.
    H = 0.0
    for c in counts:
        if c:
            p = c / total
            H -= p * log2(p)
    return H


def _counts_for_guess(args):
    """
    Worker: given (guess, remaining_list) return (guess, entropy).
    We only compute patterns vs the *remaining* solutions (not every allowed word).
    """
    guess, remaining_list = args
    # 243 buckets for 5-trit Wordle patterns
    buckets = [0]*243
    for w in remaining_list:
        pid = calculate_pattern_id(guess, w)
        buckets[pid] += 1
    return guess, _entropy_from_counts(buckets)


def calculate_entropies_parallel(guesses, remaining, n_jobs=None, show_progress=True):
    """
    Entropy of each candidate in `guesses`, given remaining solution set `remaining`.
    Parallelized across guesses with multiprocessing.
    """
    remaining_list = list(remaining)
    if not remaining_list:
        return {g: 0.0 for g in guesses}

    if n_jobs is None:
        n_jobs = max(1, (cpu_count() or 1) - 1)

    ent = {}
    with Pool(processes=n_jobs) as pool:
        it = pool.imap_unordered(_counts_for_guess, ((
            g, remaining_list) for g in guesses), chunksize=64)
        if show_progress:
            for g, H in tqdm(it, total=len(guesses), desc="Scoring (entropy)"):
                ent[g] = H
        else:
            for g, H in it:
                ent[g] = H
    return ent

# ---------- Remaining filter ----------


def reduce_remaining(guess, target_feedback_id, remaining):
    """
    Keep only words in `remaining` that would yield `target_feedback_id` for this `guess`.
    One linear scan – no giant caches.
    """
    out = []
    for w in remaining:
        if calculate_pattern_id(guess, w) == target_feedback_id:
            out.append(w)
    return set(out)

# ---------- Percentile ----------


def percentile_rank(values, x):
    n = len(values)
    if n == 0:
        return 0.0
    less = sum(v < x for v in values)
    equal = sum(v == x for v in values)
    return 100.0 * (less + 0.5 * equal) / n

# ---------- Main ----------


def main(guesses, target, verbose=True, n_jobs=None, score_from_all_allowed=True):
    """
    Critique your Wordle play (memory-light, multicore).
    - Entropy is computed vs the *current remaining solutions* only.
    - We score every candidate in:
        * all allowed guesses (default), or
        * just the remaining solutions (set score_from_all_allowed=False for speed).
    """
    all_dictionary, solution_set, WORD_LEN = load_wordlists()
    target = target.strip().lower()
    assert len(target) == WORD_LEN, f"Target must be {WORD_LEN} letters."
    # Optionally: assert target in solution_set

    remaining = set(solution_set)
    results = []

    # For quick membership validation:
    allowed = set(all_dictionary)

    for round_idx, guess in enumerate(guesses, start=1):
        guess = guess.strip().lower()
        if guess not in allowed:
            raise ValueError(
                f"Guess '{guess}' is not in the allowed guesses list (all_words.txt).")

        # Which pool of words to rank this turn?
        pool_to_score = all_dictionary if score_from_all_allowed else list(
            remaining)

        # Compute entropies in parallel
        ent = calculate_entropies_parallel(
            pool_to_score, remaining, n_jobs=n_jobs, show_progress=verbose)

        # Best available move and your guess stats
        best_guess, best_entropy = max(ent.items(), key=lambda kv: kv[1])
        guess_entropy = ent.get(guess, 0.0)
        pct = percentile_rank(list(ent.values()), guess_entropy)

        # Actual feedback vs target, then reduce remaining
        feedback_id = calculate_pattern_id(guess, target)
        new_remaining = reduce_remaining(guess, feedback_id, remaining)

        info = {
            "round": round_idx,
            "guess": guess,
            "pattern_id": feedback_id,         # compact 0..242
            "percentile": pct,                 # higher is better
            "guess_entropy": guess_entropy,    # bits
            "best_guess": best_guess,
            "best_entropy": best_entropy,
            "remaining_after": len(new_remaining),
        }
        results.append(info)

        if verbose:
            # If you want the 5-trit tuple back, decode feedback_id here for display.
            print(f"Round {round_idx} — Guess: {guess}")
            print(
                f"  Your guess percentile: {pct:.2f}%  (entropy={guess_entropy:.3f} bits)")
            print(
                f"  Best available guess:  {best_guess}  (entropy={best_entropy:.3f} bits)")
            print(f"  Remaining possible solutions: {len(new_remaining)}\n")

        remaining = new_remaining

    return results


if __name__ == "__main__":
    # Example usage:
    main(['crane', 'tares', 'hater', 'water', 'mater', 'dater'],
         'later', n_jobs=None, score_from_all_allowed=True)
