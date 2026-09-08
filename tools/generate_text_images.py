#!/usr/bin/env python3
"""Generate 15k 600x600 white-bg black-text PNG images of random ASCII strings."""

from __future__ import annotations

import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Canvas
IMG_SIZE = 600
MARGIN = 100
DRAW_SIZE = IMG_SIZE - 2 * MARGIN  # 500x500 writable region
MAX_CHARS_PER_LINE = 6
MAX_LINES = 2
MAX_TOTAL_CHARS = MAX_CHARS_PER_LINE * MAX_LINES

NUM_IMAGES = 15_000
ENGLISH_RATIO = 0.40
OUT_DIR = Path(__file__).resolve().parent.parent / "AS_VALs" / "outputs" / "text_images"


# Printable ASCII excluding space-only / control; keep space for phrases
PRINTABLE = string.digits + string.ascii_letters + string.punctuation + " "
PRINTABLE_NO_SPACE = string.digits + string.ascii_letters + string.punctuation

# Short English words (<= 12 chars) suitable for 1–2 lines of 6
WORDS = [
    "a", "I", "ok", "hi", "go", "up", "on", "in", "at", "to", "of", "or",
    "be", "do", "we", "me", "my", "by", "if", "it", "is", "as", "an", "no",
    "yes", "the", "and", "for", "you", "all", "not", "can", "had", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
    "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did",
    "its", "let", "put", "say", "she", "too", "use", "dad", "mom", "cat",
    "dog", "run", "sun", "car", "bus", "map", "key", "box", "cup", "pen",
    "red", "big", "hot", "cold", "warm", "cool", "fast", "slow", "good",
    "best", "love", "hate", "hope", "fear", "time", "year", "week", "hour",
    "home", "work", "play", "read", "write", "talk", "walk", "jump", "open",
    "close", "start", "stop", "help", "need", "want", "like", "make", "take",
    "give", "find", "know", "think", "look", "come", "gone", "here", "there",
    "where", "when", "what", "which", "this", "that", "these", "those", "with",
    "from", "into", "over", "under", "about", "after", "before", "again",
    "always", "never", "often", "once", "twice", "first", "last", "next",
    "right", "left", "above", "below", "inside", "outside", "people", "person",
    "friend", "family", "mother", "father", "sister", "brother", "school",
    "office", "market", "street", "city", "country", "world", "earth", "water",
    "fire", "light", "dark", "night", "morning", "evening", "summer", "winter",
    "spring", "autumn", "happy", "sad", "angry", "tired", "hungry", "thirsty",
    "strong", "weak", "brave", "quiet", "loud", "soft", "hard", "easy",
    "simple", "complex", "pretty", "ugly", "clean", "dirty", "fresh", "stale",
    "sweet", "sour", "bitter", "salty", "apple", "bread", "milk", "coffee",
    "tea", "juice", "pizza", "pasta", "salad", "fruit", "sugar", "salt",
    "phone", "laptop", "tablet", "mouse", "screen", "keyboard", "window",
    "door", "table", "chair", "bed", "room", "house", "garden", "park",
    "river", "ocean", "mountain", "forest", "flower", "tree", "leaf", "bird",
    "fish", "horse", "sheep", "tiger", "lion", "bear", "wolf", "fox",
    "music", "song", "dance", "movie", "book", "paper", "pen", "pencil",
    "color", "black", "white", "green", "blue", "yellow", "orange", "purple",
    "brown", "gray", "silver", "gold", "metal", "wood", "glass", "stone",
    "number", "letter", "word", "phrase", "sentence", "story", "poem",
    "code", "data", "file", "folder", "image", "photo", "video", "audio",
    "error", "debug", "build", "test", "deploy", "server", "client", "user",
    "admin", "login", "logout", "signup", "password", "token", "cache",
    "queue", "stack", "array", "list", "dict", "tuple", "class", "object",
    "method", "function", "module", "import", "export", "return", "yield",
    "true", "false", "null", "none", "void", "type", "value", "index",
    "count", "total", "sum", "avg", "max", "min", "sort", "filter",
    "search", "query", "result", "output", "input", "param", "arg", "flag",
    "hello", "world", "python", "script", "random", "string", "ascii",
    "pixel", "margin", "canvas", "font", "draw", "render", "save", "load",
]

# Short phrases that fit in at most 2 lines x 6 chars after wrapping
PHRASES = [
    "hi there", "ok go", "yes sir", "no way", "oh no", "oh yes", "my god",
    "good job", "well done", "thank you", "bye now", "see you", "come on",
    "go away", "get out", "sit down", "stand up", "look up", "look at",
    "turn on", "turn off", "wake up", "go home", "at home", "at work",
    "on time", "in time", "too bad", "so good", "so far", "so what",
    "why not", "who am I", "how are", "what if", "as if", "if so",
    "not yet", "not now", "right now", "just now", "once more", "once upon",
    "all day", "all night", "next day", "last day", "new year", "old man",
    "big deal", "no idea", "bad luck", "good luck", "high five", "low key",
    "keep it", "let go", "hold on", "hang on", "give up", "take it",
    "make it", "do it", "try it", "use it", "fix it", "set it",
    "I am ok", "I love u", "I need u", "you can", "we can", "they can",
    "it is ok", "it was", "he said", "she said", "we said", "go for",
    "wait up", "hurry up", "slow down", "calm down", "cheer up", "shut up",
    "open up", "close it", "save me", "help me", "call me", "text me",
    "email me", "find me", "meet me", "join us", "follow me", "trust me",
    "believe", "for sure", "of course", "no doubt", "no issue", "no thanks",
    "yes please", "after all", "before me", "above all", "in fact", "in short",
    "at last", "at least", "at most", "by far", "by now", "for now",
    "from now", "up to", "out of", "into it", "on it", "off it",
    "red car", "blue sky", "green tea", "black tea", "hot dog", "ice tea",
    "new book", "old song", "fast car", "slow bus", "big city", "small town",
    "dark night", "bright day", "cold wind", "warm sun", "fresh air",
    "pure water", "sweet cake", "sour milk", "salt fish", "soft bed",
    "hard work", "easy task", "best friend", "true love", "real life",
    "code time", "bug fix", "unit test", "pull req", "push all", "git add",
    "log in", "log out", "sign up", "sign in", "check out", "check in",
]


def wrap_text(text: str) -> list[str] | None:
    """Wrap text into at most 2 lines of <=6 chars. Returns None if impossible."""
    text = text.strip()
    if not text or len(text) > MAX_TOTAL_CHARS:
        return None
    if " " not in text:
        if len(text) <= MAX_CHARS_PER_LINE:
            return [text]
        if len(text) <= MAX_TOTAL_CHARS:
            return [text[:MAX_CHARS_PER_LINE], text[MAX_CHARS_PER_LINE:]]
        return None

    # Prefer wrapping at spaces when possible
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if len(w) > MAX_CHARS_PER_LINE:
            # hard-split long token
            remaining = w
            while remaining:
                chunk = remaining[:MAX_CHARS_PER_LINE]
                remaining = remaining[MAX_CHARS_PER_LINE:]
                if current:
                    lines.append(current)
                    current = ""
                if len(lines) >= MAX_LINES:
                    return None
                if remaining:
                    lines.append(chunk)
                    if len(lines) >= MAX_LINES and remaining:
                        return None
                else:
                    current = chunk
            continue
        candidate = w if not current else f"{current} {w}"
        if len(candidate) <= MAX_CHARS_PER_LINE:
            current = candidate
        else:
            if current:
                lines.append(current)
            if len(lines) >= MAX_LINES:
                return None
            current = w
    if current:
        lines.append(current)
    if len(lines) > MAX_LINES:
        return None
    if any(len(ln) > MAX_CHARS_PER_LINE for ln in lines):
        return None
    return lines


def random_ascii_string() -> str:
    """Random printable ASCII, 1–12 chars, valid for wrap."""
    n = random.randint(1, MAX_TOTAL_CHARS)
    # Prefer no leading/trailing spaces; allow internal spaces sometimes
    if n == 1:
        return random.choice(PRINTABLE_NO_SPACE)
    chars = [random.choice(PRINTABLE_NO_SPACE)]
    for _ in range(n - 2):
        chars.append(random.choice(PRINTABLE if random.random() < 0.12 else PRINTABLE_NO_SPACE))
    chars.append(random.choice(PRINTABLE_NO_SPACE))
    s = "".join(chars)
    # Collapse accidental multi-spaces and ensure wrap works
    while "  " in s:
        s = s.replace("  ", " ")
    s = s.strip()
    if not s:
        return random.choice(PRINTABLE_NO_SPACE)
    if wrap_text(s) is None:
        # fall back to contiguous no-space string
        s = "".join(random.choices(PRINTABLE_NO_SPACE, k=n))
    return s


def random_english_text() -> str:
    """Pick a word or short phrase that fits the layout rules."""
    for _ in range(50):
        if random.random() < 0.55:
            candidate = random.choice(WORDS)
        else:
            candidate = random.choice(PHRASES)
        # Occasionally join two short words into a phrase
        if random.random() < 0.15:
            a, b = random.choice(WORDS), random.choice(WORDS)
            if len(a) + 1 + len(b) <= MAX_TOTAL_CHARS:
                candidate = f"{a} {b}"
        if wrap_text(candidate) is not None:
            return candidate
    # guaranteed short fallback
    return random.choice([w for w in WORDS if len(w) <= MAX_CHARS_PER_LINE])


_FONT_PATH: str | None = None
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _resolve_font_path() -> str | None:
    global _FONT_PATH
    if _FONT_PATH is not None:
        return _FONT_PATH or None
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            _FONT_PATH = path
            return path
    _FONT_PATH = ""
    return None


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    path = _resolve_font_path()
    if path:
        try:
            font = ImageFont.truetype(path, size=size)
            _FONT_CACHE[size] = font
            return font
        except OSError:
            pass
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def fit_font(lines: list[str], draw: ImageDraw.ImageDraw) -> ImageFont.ImageFont:
    """Largest font that fits text inside the 500x500 drawable area."""
    # leave a little padding inside the drawable box
    max_w = DRAW_SIZE - 20
    max_h = DRAW_SIZE - 20
    lo, hi = 12, 220
    best = get_font(48)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = get_font(mid)
        widths = []
        heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        line_gap = max(4, mid // 8)
        total_h = sum(heights) + line_gap * (len(lines) - 1)
        if max(widths) <= max_w and total_h <= max_h:
            best = font
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_image(text: str, path: Path) -> None:
    lines = wrap_text(text)
    if lines is None:
        raise ValueError(f"Cannot wrap text: {text!r}")

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    font = fit_font(lines, draw)

    # Measure block
    line_gap = max(4, getattr(font, "size", 48) // 8)
    metrics = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        metrics.append((bbox[2] - bbox[0], bbox[3] - bbox[1], -bbox[1]))
    total_h = sum(h for _, h, _ in metrics) + line_gap * (len(lines) - 1)

    # Center within the 500x500 drawable region (margin 100)
    y = MARGIN + (DRAW_SIZE - total_h) // 2
    for line, (w, h, ascent_fix) in zip(lines, metrics):
        x = MARGIN + (DRAW_SIZE - w) // 2
        draw.text((x, y + ascent_fix), line, fill="black", font=font)
        y += h + line_gap

    img.save(path, format="PNG")


def build_string_list(n: int) -> tuple[list[str], int]:
    n_english = int(n * ENGLISH_RATIO)
    if n_english / n < ENGLISH_RATIO:
        n_english += 1
    n_random = n - n_english
    texts = [random_english_text() for _ in range(n_english)]
    texts += [random_ascii_string() for _ in range(n_random)]
    random.shuffle(texts)
    return texts, n_english


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    print(f"Generating {NUM_IMAGES} images (>= {ENGLISH_RATIO:.0%} English)...")

    texts, n_english = build_string_list(NUM_IMAGES)
    print(f"English/phrase: {n_english} ({n_english / NUM_IMAGES:.1%})")
    print(f"Random ASCII: {NUM_IMAGES - n_english}")

    manifest = OUT_DIR / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as mf:
        for i, text in enumerate(texts):
            name = f"{i:05d}.png"
            render_image(text, OUT_DIR / name)
            safe = text.replace("\\", "\\\\").replace("\n", "\\n")
            mf.write(f"{name}\t{safe}\n")
            if (i + 1) % 500 == 0 or i + 1 == NUM_IMAGES:
                print(f"  saved {i + 1}/{NUM_IMAGES}")

    print(f"Done. Images + manifest in: {OUT_DIR}")


if __name__ == "__main__":
    main()
