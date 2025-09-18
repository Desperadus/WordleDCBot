#!/usr/bin/python
import os
import itertools
import pickle
from collections import defaultdict, Counter
from scipy.stats import entropy
from tqdm import tqdm

# Files:
# - DICT_FILE_all: all allowed guesses (used to compute entropies / partitions)
# - DICT_FILE: official solution list (used as "possible solutions")
DICT_FILE_all = 'all_words.txt'
DICT_FILE = 'words.txt'
CACHE_FILE = 'pattern_dict.p'


def calculate_pattern(guess, true):
    """Return Wordle feedback pattern as a tuple of 5 ints in {0,1,2}."""
    wrong = [i for i, v in enumerate(guess) if v != true[i]]
    counts = Counter(true[i] for i in wrong)
    pattern = [2] * 5
    for i in wrong:
        v = guess[i]
        if counts[v] > 0:
            pattern[i] = 1
            counts[v] -= 1
        else:
            pattern[i] = 0
    return tuple(pattern)


def generate_pattern_dict(dictionary):
    """
    For each word and (five-cell) pattern, precompute the set of words that yield that pattern.
    Returns: dict[word][pattern] -> set(words)
    """
    pattern_dict = defaultdict(lambda: defaultdict(set))
    for w1 in tqdm(dictionary, desc="Precomputing patterns (outer)"):
        for w2 in dictionary:
            pattern = calculate_pattern(w1, w2)
            pattern_dict[w1][pattern].add(w2)
    return {k: dict(v) for k, v in pattern_dict.items()}


def calculate_entropies(words, possible_words, pattern_dict, all_patterns):
    """Entropy of each candidate in `words`, given remaining `possible_words` (solutions)."""
    entropies = {}
    for w in words:
        # How the remaining solution set would split for each possible feedback pattern:
        counts = []
        pmap = pattern_dict[w]
        for pattern in all_patterns:
            bucket = pmap.get(pattern, set())
            counts.append(len(bucket & possible_words))
        entropies[w] = entropy(counts)
    return entropies


def percentile_rank(values, x):
    """
    Percentile rank of x among `values` (0..100, higher is better).
    Uses mid-rank for ties.
    """
    n = len(values)
    if n == 0:
        return 0.0
    less = sum(v < x for v in values)
    equal = sum(v == x for v in values)
    return 100.0 * (less + 0.5 * equal) / n


def load_wordlists():
    with open(DICT_FILE_all) as f:
        all_dictionary = [w.strip() for w in f if w.strip()]
    with open(DICT_FILE) as f:
        solution_list = [w.strip() for w in f if w.strip()]
    # Sanity checks
    lens = {len(w) for w in all_dictionary}
    assert len(lens) == 1, "Dictionary contains different length words."
    word_len = next(iter(lens))
    return all_dictionary, set(solution_list), word_len


def load_or_build_pattern_cache(all_dictionary):
    if os.path.exists(CACHE_FILE):
        pattern_dict = pickle.load(open(CACHE_FILE, 'rb'))
        # quick integrity check: ensure some overlap with current list
        if not set(pattern_dict.keys()).issuperset(set(all_dictionary[:10])):
            # cache likely mismatched -> rebuild
            pattern_dict = generate_pattern_dict(all_dictionary)
            pickle.dump(pattern_dict, open(CACHE_FILE, 'wb+'))
    else:
        pattern_dict = generate_pattern_dict(all_dictionary)
        pickle.dump(pattern_dict, open(CACHE_FILE, 'wb+'))
    return pattern_dict


def main(guesses, target, verbose=True):
    """
    Critique your Wordle play.
    Args:
        guesses: list[str] — guesses you actually played, in order.
        target:  str       — the true answer.
    Prints (and returns) per-guess stats:
        - percentile of your guess by entropy (higher is better),
        - remaining possible solutions after applying the feedback,
        - and the best-entropy guess available at that step.
    """
    all_dictionary, solution_set, WORD_LEN = load_wordlists()
    target = target.strip().lower()

    assert len(target) == WORD_LEN, f"Target must be {WORD_LEN} letters."
    # If you want to enforce "target in official solution list", uncomment below:
    # assert target in solution_set, "Target not found in solution list."

    # Precompute pattern mapping over all allowed guesses
    pattern_dict = load_or_build_pattern_cache(all_dictionary)

    # All possible feedback patterns (3^WORD_LEN)
    all_patterns = list(itertools.product([0, 1, 2], repeat=WORD_LEN))

    # Start with the official solution set as the remaining possibilities
    remaining = set(solution_set)

    results = []
    for round_idx, guess in enumerate(guesses, start=1):
        guess = guess.strip().lower()
        if guess not in pattern_dict:
            raise ValueError(
                f"Guess '{guess}' is not in the allowed guesses list (all_words.txt)."
            )

        # Compute entropies for all allowed guesses *given the current remaining solutions*
        ent = calculate_entropies(
            all_dictionary, remaining, pattern_dict, all_patterns)

        # Best possible move right now
        best_guess, best_entropy = max(ent.items(), key=lambda kv: kv[1])

        if guess not in ent:
            raise ValueError(f"No entropy computed for guess '{guess}'.")

        guess_entropy = ent[guess]
        pct = percentile_rank(list(ent.values()), guess_entropy)

        # Apply feedback for the actual target, then reduce remaining solutions
        pattern = calculate_pattern(guess, target)
        bucket = pattern_dict[guess].get(pattern, set())
        # Intersect with the *solution* set so "remaining" counts solutions only
        new_remaining = remaining & (bucket & solution_set)

        # Report
        info = {
            "round": round_idx,
            "guess": guess,
            "pattern": pattern,
            "percentile": pct,                 # 0..100 (higher = better)
            "guess_entropy": guess_entropy,    # for reference
            "best_guess": best_guess,
            "best_entropy": best_entropy,
            "remaining_after": len(new_remaining),
        }
        results.append(info)

        if verbose:
            print(f"Round {round_idx} — Guess: {guess}")
            print(f"  Pattern vs target: {pattern}")
            print(
                f"  Your guess percentile: {pct:.2f}%  (entropy={guess_entropy:.3f})")
            print(
                f"  Best available guess:  {best_guess}  (entropy={best_entropy:.3f})")
            print(f"  Remaining possible solutions: {len(new_remaining)}\n")

        # Update remaining for the next round
        remaining = new_remaining

    return results


if __name__ == "__main__":
    # Example usage:
    main(['slate', 'pound', 'grime', 'chine', 'knife'], 'knife')
    pass
