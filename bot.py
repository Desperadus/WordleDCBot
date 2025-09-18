import os
import io
import re
import json
import math
import pickle
import asyncio
import itertools
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv

from scipy.stats import entropy
from tqdm import tqdm

# ---------- Config ----------
DICT_FILE_ALL = "all_words.txt"
DICT_FILE_SOL = "words.txt"
CACHE_FILE = "pattern_dict.p"
GEMINI_MODEL = "gemini-2.5-flash"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")


from google import genai
from google.genai import types as genai_types

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def calculate_pattern(guess: str, true: str) -> Tuple[int, int, int, int, int]:
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

def generate_pattern_dict(dictionary: List[str]) -> Dict[str, Dict[Tuple[int, ...], set]]:
    pattern_dict = defaultdict(lambda: defaultdict(set))
    for w1 in tqdm(dictionary, desc="Precomputing patterns", disable=True):
        for w2 in dictionary:
            pattern = calculate_pattern(w1, w2)
            pattern_dict[w1][pattern].add(w2)
    return {k: dict(v) for k, v in pattern_dict.items()}

def calculate_entropies_bits(
    words: List[str],
    possible_words: set,
    pattern_dict,
    all_patterns
) -> Tuple[Dict[str, float], Dict[str, Dict[Tuple[int, ...], int]]]:
    ent_bits = {}
    pattern_counts = {}
    for w in words:
        pmap = pattern_dict[w]
        counts = []
        counts_map = {}
        for pattern in all_patterns:
            bucket = pmap.get(pattern, set())
            c = len(bucket & possible_words)
            counts.append(c)
            counts_map[pattern] = c
        ent_bits[w] = entropy(counts, base=2)
        pattern_counts[w] = counts_map
    return ent_bits, pattern_counts

def percentile_rank(values, x, eps=1e-9):
    # 100% if you tie the maximum (within tolerance)
    mx = max(values)
    if abs(x - mx) <= eps:
        return 100.0
    n = len(values)
    less = sum(v < x - eps for v in values)
    equal = sum(abs(v - x) <= eps for v in values)
    return 100.0 * (less + 0.5 * equal) / n

def load_wordlists():
    with open(DICT_FILE_ALL) as f:
        all_dictionary = [w.strip().lower() for w in f if w.strip()]
    with open(DICT_FILE_SOL) as f:
        solution_list = [w.strip().lower() for w in f if w.strip()]
    lens = {len(w) for w in all_dictionary}
    assert len(lens) == 1, "Dictionary contains different length words."
    word_len = next(iter(lens))
    return all_dictionary, set(solution_list), word_len

def load_or_build_pattern_cache(all_dictionary: List[str]):
    if os.path.exists(CACHE_FILE):
        try:
            pattern_dict = pickle.load(open(CACHE_FILE, "rb"))
            if not set(all_dictionary[:5]).issubset(pattern_dict.keys()):
                raise ValueError("Cache mismatch; rebuilding.")
            return pattern_dict
        except Exception:
            pass
    pattern_dict = generate_pattern_dict(all_dictionary)
    pickle.dump(pattern_dict, open(CACHE_FILE, "wb+"))
    return pattern_dict

def convert_pattern_to_emoji_string(pattern: Tuple[int, ...]) -> str:
    return "".join({
        0: "⬜",
        1: "🟨",
        2: "🟩"
    }[i] for i in pattern)

def analyze_play(guesses: List[str], target: str):
    all_dictionary, solution_set, WORD_LEN = load_wordlists()
    if target is None or target.strip().lower() == "None":
        target = guesses[-1]
    assert len(target) == WORD_LEN, f"Target must be {WORD_LEN} letters."

    pattern_dict = load_or_build_pattern_cache(all_dictionary)
    all_patterns = list(itertools.product([0, 1, 2], repeat=WORD_LEN))
    remaining = set(solution_set)
    results = []

    for round_idx, guess in enumerate(guesses, start=1):
        print(f"Round {round_idx}: Guessing '{guess}'")
        guess = guess.strip().lower()
        if guess not in pattern_dict:
            raise ValueError(f"Guess '{guess}' not in allowed list.")

        ent_bits, pattern_counts = calculate_entropies_bits(
            all_dictionary, remaining, pattern_dict, all_patterns
        )
        best_guess, best_entropy_bits = max(ent_bits.items(), key=lambda kv: kv[1])
        guess_entropy_bits = ent_bits[guess]
        pct_expected = percentile_rank(list(ent_bits.values()), guess_entropy_bits)

        pattern = calculate_pattern(guess, target)
        observed_bucket_size = pattern_counts[guess][pattern]
        total_before = len(remaining)
        received_info_bits = (
            float("inf") if observed_bucket_size < 1
            else math.log2(total_before / observed_bucket_size)
        )
        luck_bits = received_info_bits - guess_entropy_bits

        # shrink remaining to solutions in the observed bucket
        bucket_all = pattern_dict[guess].get(pattern, set())
        new_remaining = remaining & (bucket_all & solution_set)

        # How "lucky" among all patterns for this guess?
        info_distribution = []
        for _, size in pattern_counts[guess].items():
            if size > 0:
                info_distribution.append(math.log2(total_before / size))
        received_info_percentile = percentile_rank(info_distribution, received_info_bits) if info_distribution else 0.0
        if guess_entropy_bits == 0:
            received_info_percentile = 0.0
        if len(remaining) == 1:
            best_guess = target

        results.append({
            "round": round_idx,
            "guess": guess,
            "pattern": convert_pattern_to_emoji_string(pattern),
            "percentile_expected": pct_expected,
            "guess_entropy_bits": guess_entropy_bits,
            "received_info_bits": received_info_bits,
            "luck_bits": luck_bits,
            "received_info_percentile": received_info_percentile,
            "best_guess": best_guess,
            "best_entropy_bits": best_entropy_bits,
            "remaining_after": len(new_remaining),
            "win": guess == target
        })
        remaining = new_remaining

    return results

WORDLE_JSON_PROMPT = """
You are an OCR + parser for Wordle screenshots.

Return ONLY JSON with this schema (no prose):
{
  "guesses": ["word1","word2", "..."],   // 5-letter lowercase, in play order
  "target": "crane" | null,              // if the solution is visible; else null
  "notes": string                        // brief note, e.g. "from 'The word was CRANE'"
}

Rules:
- If letters are partially visible, infer the whole 5-letter word if unambiguous.
- If the puzzle is unfinished and target is not shown, set "target": null.
- Normalize to lowercase ASCII.
"""

async def extract_words_with_gemini(image_bytes: bytes, mime_type: str) -> Dict:
    # Prefer structured JSON output; see google-genai "structured output" docs.
    # https://ai.google.dev/gemini-api/docs/structured-output
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                WORDLE_JSON_PROMPT
            ],
            config={"response_mime_type": "application/json"},
        )
        txt = response.text
        return json.loads(txt)
    except Exception:
        # Fallback: ask for plain text, then try to recover JSON blob with regex
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                WORDLE_JSON_PROMPT + "\nReturn ONLY JSON."
            ],
        )
        t = response.text or ""
        match = re.search(r"\{.*\}", t, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"guesses": [], "target": None, "notes": "parse_failed"}

# ---------- Helpers ----------
def format_bits(x: float) -> str:
    if x == float("inf"):
        return "∞"
    return f"{x:.3f}"

def ensure_list_of_words(raw) -> List[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for w in raw:
        if isinstance(w, str) and len(w.strip()) == 5 and w.isalpha():
            out.append(w.strip().lower())
    return out

def detect_mime(attachment: discord.Attachment) -> str:
    if attachment.content_type and attachment.content_type.startswith("image/"):
        return attachment.content_type
    # Fallback based on filename
    name = attachment.filename.lower()
    if name.endswith(".png"): return "image/png"
    if name.endswith(".jpg") or name.endswith(".jpeg"): return "image/jpeg"
    if name.endswith(".webp"): return "image/webp"
    return "image/png"

@bot.command(name="wanal")
@commands.cooldown(1, 8, commands.BucketType.user)
async def wordleanalysis(ctx, target_arg: Optional[str] = None):
    """
    Usage: !wordleanalysis [optional_target]
    - attach a Wordle screenshot to your message
    - optionally pass the target word if you already know it
    """
    print("Wordle analysis initiated.")
    if not ctx.message.attachments:
        return await ctx.reply("Please attach a Wordle screenshot to your command.", mention_author=False)

    #Reply to user that the analysis is underway
    response = await ctx.reply("Analyzing your Wordle screenshot...", mention_author=False)

    # Take the first image attachment
    att = ctx.message.attachments[0]
    mime = detect_mime(att)
    image_bytes = await att.read()

    # OCR + parse with Gemini
    parsed = await extract_words_with_gemini(image_bytes, mime)
    guesses = ensure_list_of_words(parsed.get("guesses", []))
    target = parsed.get("target")
    notes = parsed.get("notes", "")
    print(f"Parsed guesses: {guesses}, target: {target}, notes: {notes}")

    # Allow explicit override via command arg
    if target_arg and isinstance(target_arg, str) and len(target_arg) == 5 and target_arg.isalpha():
        target = target_arg.lower()

    if not guesses:
        return await ctx.reply("I couldn't detect any guesses in the screenshot. Please try a clearer image.", mention_author=False)

    if not target:
        await ctx.reply("I couldn't see the solution in the screenshot. DM me again with `!wordleanalysis TARGETWORD` (attach the same image or skip the image).", mention_author=False)
        try:
            await ctx.author.send("Heads up: I couldn't detect the target word. If you know it, DM me:\n`!wordleanalysis TARGETWORD`.")
        except discord.Forbidden:
            pass
        return

    # Delete the initial message and response
    try:
        await ctx.message.delete()
        await response.delete()
    except Exception:
        print("Failed to delete messages.")

    # Run analysis
    try:
        results = analyze_play(guesses, target)
    except Exception as e:
        return await ctx.reply(f"Analysis error: {e}", mention_author=False)

    # Build a pretty embed for DM
    emb = discord.Embed(
        title="Wordle Solve Critique",
        description=f"**Target:** `{target}`\n**Detected guesses:** {', '.join(f'`{g}`' for g in guesses)}\n{('Note: ' + notes) if notes else ''}",
        color=discord.Color.green()
    )
    # Per-round fields
    for r in results:
        name = f"Round {r['round']}: `{r['guess']}` → pattern `{r['pattern']}`"
        if r.get("win"):
            val = "• **Result:** ✅ Correct my nigga!"
        else:
            val = (
                f"• **Expected entropy:** {format_bits(r['guess_entropy_bits'])} bits "
                f"(pct {r['percentile_expected']:.1f}%)\n"
                f"• **Received entropy:** {format_bits(r['received_info_bits'])} bits "
                f"(luck Δ {format_bits(r['luck_bits'])}, pct {r['received_info_percentile']:.1f}%)\n"
                f"• **Best available guess:** `{r['best_guess']}` ({format_bits(r['best_entropy_bits'])} bits)\n"
                f"• **Remaining solutions:** {r['remaining_after']}"
            )
        emb.add_field(name=name, value=val, inline=False)

    emb.set_footer(text="Powered by Pikachuj")

    # DM the user
    try:
        await ctx.author.send(embed=emb)
    except discord.Forbidden:
        await ctx.reply("I can't DM you (privacy settings). Please DM me first, then re-run the command.", mention_author=False)


# ---------- Run ----------
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN not set")
    bot.run(DISCORD_TOKEN)
