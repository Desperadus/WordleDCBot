import os
import io
import re
import json
import math
import pickle
import asyncio
import itertools
import logging
import uuid
from logging.handlers import RotatingFileHandler
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv

from scipy.stats import entropy
from tqdm import tqdm

# CONFIG
DICT_FILE_ALL = "all_words.txt"
DICT_FILE_SOL = "words.txt"
CACHE_FILE = "pattern_dict.p"
GEMINI_MODEL = "gemini-2.5-flash"

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("wordlebot")
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    # File (rotating) handler
    fh = RotatingFileHandler(
        "logs/bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    # Console handler (useful for local/dev)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    # Avoid duplicate handlers if reloaded
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

LOGGER = setup_logging()

def log_event(event: str, **payload):
    """Log a JSON line for easier analysis later."""
    try:
        LOGGER.info(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str))
    except Exception:
        # Fallback if something isn't JSON-serializable
        LOGGER.info(f"{event} {payload}")

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

@bot.event
async def on_ready():
    try:
        user = str(bot.user) if bot.user else None
        guilds = [g.id for g in bot.guilds]
        log_event("bot_ready", bot_user=user, guild_count=len(guilds), guild_ids=guilds)
    except Exception:
        LOGGER.exception("Failed to log bot_ready")

@bot.event
async def on_command_error(ctx, error):
    user_name = str(ctx.author) if ctx and getattr(ctx, "author", None) else None
    user_id = getattr(ctx.author, "id", None) if ctx and getattr(ctx, "author", None) else None
    guild_id = getattr(ctx.guild, "id", None) if ctx and getattr(ctx, "guild", None) else None
    channel_id = getattr(ctx.channel, "id", None) if ctx and getattr(ctx, "channel", None) else None
    if isinstance(error, commands.CommandOnCooldown):
        log_event(
            "command_cooldown",
            command=getattr(ctx.command, "name", None),
            user_name=user_name,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            retry_after=getattr(error, "retry_after", None),
        )
        try:
            await ctx.reply(f"You're on cooldown. Try again in {error.retry_after:.1f}s.", mention_author=False)
        except Exception:
            pass
    else:
        log_event(
            "command_error",
            command=getattr(ctx.command, "name", None),
            user_name=user_name,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
            error=str(error),
        )
        raise error

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
    remaining_solutions = set(solution_set)
    remaining_allowed = set(all_dictionary)
    results = []

    for round_idx, guess in enumerate(guesses, start=1):
        LOGGER.debug(f"Round {round_idx}: Guessing '{guess}'")
        guess = guess.strip().lower()
        if guess not in pattern_dict:
            raise ValueError(f"Guess '{guess}' not in allowed list.")

        ent_bits, pattern_counts = calculate_entropies_bits(
            all_dictionary, remaining_solutions, pattern_dict, all_patterns
        )
        best_guess, best_entropy_bits = max(ent_bits.items(), key=lambda kv: kv[1])
        guess_entropy_bits = ent_bits[guess]
        pct_expected = percentile_rank(list(ent_bits.values()), guess_entropy_bits)

        pattern = calculate_pattern(guess, target)
        observed_bucket_size = pattern_counts[guess][pattern]
        total_before = len(remaining_solutions)
        received_info_bits = (
            float("inf") if observed_bucket_size < 1
            else math.log2(total_before / observed_bucket_size)
        )
        luck_bits = received_info_bits - guess_entropy_bits

        # shrink remaining to solutions in the observed bucket
        bucket_all = pattern_dict[guess].get(pattern, set())
        bucket_solutions = bucket_all & solution_set

        remaining_solutions &= bucket_solutions
        remaining_allowed &= bucket_all
        # How "lucky" among all patterns for this guess?
        info_distribution = []
        for _, size in pattern_counts[guess].items():
            if size > 0:
                info_distribution.append(math.log2(total_before / size))
        received_info_percentile = percentile_rank(info_distribution, received_info_bits) if info_distribution else 0.0
        if guess_entropy_bits == 0:
            received_info_percentile = 0.0
        if len(remaining_allowed) == 1:
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
            "remaining_solutions": len(remaining_solutions),
            "remaining_allowed": len(remaining_allowed),
            "win": guess == target
        })
        

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
    Usage: !wanal [optional_target]
    - attach a Wordle screenshot to your message
    - optionally pass the target word if you already know it
    """
    user_name = str(ctx.author)
    user_id = getattr(ctx.author, "id", None)
    guild_id = getattr(ctx.guild, "id", None) if ctx.guild else None
    channel_id = getattr(ctx.channel, "id", None)
    request_id = str(uuid.uuid4())
    log_event(
        "command_invoked",
        command="!wanal",
        user_name=user_name,
        user_id=user_id,
        guild_id=guild_id,
        channel_id=channel_id,
        request_id=request_id,
        attachments=[a.filename for a in ctx.message.attachments],
    )
    if not ctx.message.attachments:
        return await ctx.reply("Please attach a Wordle screenshot to your command.", mention_author=False)

    # Reply to user that the analysis is underway
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
    log_event(
        "parsed_input",
        user_name=user_name,
        user_id=user_id,
    request_id=request_id,
        guesses=guesses,
        target=target,
        notes=notes,
        mime=mime,
        attachment_name=att.filename,
    )

    # Allow explicit override via command arg
    if target_arg and isinstance(target_arg, str) and len(target_arg) == 5 and target_arg.isalpha():
        target = target_arg.lower()
        log_event("target_overridden", user_name=user_name, user_id=user_id, request_id=request_id, target=target)

    if not guesses:
        log_event("no_guesses_detected", user_name=user_name, user_id=user_id, request_id=request_id, target_arg=target_arg)
        return await ctx.reply("I couldn't detect any guesses in the screenshot. Please try a clearer image.", mention_author=False)

    if not target:
        log_event("no_target_detected", user_name=user_name, user_id=user_id, request_id=request_id, guesses=guesses)
        await ctx.reply("I couldn't see the solution in the screenshot. DM me again with `!wanal TARGETWORD` (attach the same image or skip the image).", mention_author=False)
        try:
            await ctx.author.send("Heads up: I couldn't detect the target word. If you know it, DM me:\n`!wanal TARGETWORD`.")
        except discord.Forbidden:
            pass
        return

    # Delete the initial message and response
    try:
        await ctx.message.delete()
        await response.delete()
    except Exception:
        LOGGER.warning("Failed to delete messages.")

    # Run analysis
    try:
        results = analyze_play(guesses, target)
    except Exception as e:
        log_event("analysis_error", user_name=user_name, user_id=user_id, request_id=request_id, error=str(e))
        return await ctx.reply(f"Analysis error: {e}", mention_author=False)

    # Log per-round entropy and progress stats
    for r in results:
        log_event(
            "analysis_round",
            user_name=user_name,
            user_id=user_id,
            request_id=request_id,
            target=target,
            **r,
        )

    # Summary log
    total_rounds = len(results)
    won = any(r.get("win") for r in results)
    log_event(
        "analysis_completed",
        user_name=user_name,
        user_id=user_id,
        request_id=request_id,
        target=target,
        total_rounds=total_rounds,
        won=won,
        guesses=guesses,
    )

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
            val = "• **Result:** ✅ Correct my dude!"
        else:
            val = (
                f"• **Expected entropy:** {format_bits(r['guess_entropy_bits'])} bits "
                f"(pct {r['percentile_expected']:.1f}%)\n"
                f"• **Received entropy:** {format_bits(r['received_info_bits'])} bits "
                f"(luck Δ {format_bits(r['luck_bits'])}, pct {r['received_info_percentile']:.1f}%)\n"
                f"• **Best available guess:** `{r['best_guess']}` ({format_bits(r['best_entropy_bits'])} bits)\n"
                f"• **Remaining solutions:** {r['remaining_solutions']}"
                f"• **Remaining allowed:** {r['remaining_allowed']}"
            )
        emb.add_field(name=name, value=val, inline=False)

    emb.set_footer(text="Powered by Pikachuj")

    # DM the user
    try:
        await ctx.author.send(embed=emb)
        log_event("dm_sent", user_name=user_name, user_id=user_id, request_id=request_id, destination="dm", rounds=len(results))
    except discord.Forbidden:
        # send the message into the channel
        await ctx.channel.send(embed=emb)
        await ctx.channel.send("I can't DM you (privacy settings). Please DM me first, then re-run the command.")
        log_event("dm_failed_privacy", user_name=user_name, user_id=user_id, request_id=request_id, destination="channel", rounds=len(results))


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN not set")
    bot.run(DISCORD_TOKEN)
