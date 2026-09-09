"""Parse OCR text lines into menu dish candidates."""

from __future__ import annotations

import re
import uuid
from typing import Any


SKIP_RE = re.compile(
    r"^(меню|menu|meню|кухня|напитки|drinks|салаты|супы|горячее|десерты|"
    r"завтрак|обед|ужин|весовое|выход|вес|ккал|калори|"
    r"руб\.?|₽|\$|цена|итого|всего|раздел|страница)$",
    re.IGNORECASE,
)
PRICE_ONLY_RE = re.compile(r"^[\d\s.,]+(?:₽|руб\.?|р\.?|€|\$)?$", re.IGNORECASE)
WEIGHT_RE = re.compile(r"(\d{2,4})\s*(?:г|гр|g)\b", re.IGNORECASE)
KCAL_RE = re.compile(r"(\d{2,4})\s*(?:ккал|kcal)\b", re.IGNORECASE)
TRAILING_PRICE_RE = re.compile(r"[\s|/·•\-–—]+(?:\d[\d\s.,]{0,8})\s*(?:₽|руб\.?|р\.?)?$", re.IGNORECASE)
LATIN_FOOD_RE = re.compile(
    r"\b(pizza|pasta|burger|steak|salad|soup|risotto|ramen|taco|wrap|latte|cappuccino)\b",
    re.IGNORECASE,
)
CYR_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")


def _clean_name(text: str) -> str:
    text = TRAILING_PRICE_RE.sub("", text).strip(" -–—|/·•.,;")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _cyr_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for ch in letters if CYR_RE.match(ch))
    return cyr / len(letters)


def _looks_like_gibberish(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3:
        return True
    # Random Latin soup on RU menus (Megehh / HUPONEHOE).
    if _cyr_ratio(text) < 0.45 and not LATIN_FOOD_RE.search(text):
        return True
    # Too many uppercase Latin islands mixed into Cyrillic garbage.
    latin_upper = sum(1 for ch in text if "A" <= ch <= "Z")
    if latin_upper >= 4 and _cyr_ratio(text) < 0.7:
        return True
    # Almost no vowels in Cyrillic → OCR junk.
    vowels = sum(1 for ch in text.casefold() if ch in "аеёиоуыэюяaeiouyіїє")
    if vowels < max(1, len(letters) // 5):
        return True
    return False


def _looks_like_dish(text: str) -> bool:
    if len(text) < 3 or len(text) > 80:
        return False
    if SKIP_RE.match(text):
        return False
    if PRICE_ONLY_RE.match(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 3:
        return False
    if _looks_like_gibberish(text):
        return False
    return True


def _merge_word_fragments(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paddle often returns one word per box — glue short alpha fragments into phrases."""
    merged: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(str(part.get("text") or "").strip() for part in buf).strip()
        confs = [float(part.get("confidence") or 0) for part in buf]
        avg = sum(confs) / max(len(confs), 1)
        merged.append({"text": text, "confidence": avg})
        buf = []

    for line in lines:
        raw = str(line.get("text") or "").strip()
        if not raw:
            continue
        has_meta = bool(
            WEIGHT_RE.search(raw) or KCAL_RE.search(raw) or PRICE_ONLY_RE.match(raw) or re.search(r"\d", raw)
        )
        short_word = len(raw) <= 22 and sum(ch.isalpha() for ch in raw) >= 2 and not has_meta
        if short_word:
            buf.append(line)
            if len(buf) >= 5:
                flush()
            continue
        flush()
        merged.append(line)
    flush()
    return merged


def _estimate_macros(name: str, calories: int) -> tuple[float, float, float]:
    lowered = name.casefold()
    if any(word in lowered for word in ("салат", "овощ", "зелень")):
        return 8.0, 10.0, 12.0
    if any(word in lowered for word in ("суп", "бульон", "щи", "борщ", "солянка")):
        return 10.0, 8.0, 18.0
    if any(word in lowered for word in ("рыба", "лосось", "тунец", "треска", "форель", "сёмга", "семга")):
        return 28.0, 12.0, 2.0
    if any(word in lowered for word in ("курица", "куриный", "индейка", "грудк", "цыпл")):
        return 28.0, 10.0, 4.0
    if any(word in lowered for word in ("стейк", "говяд", "мясо", "свинин", "шашлык")):
        return 26.0, 18.0, 2.0
    if any(word in lowered for word in ("паста", "пицца", "рис", "лапш", "плов", "бургер", "картоф")):
        return 12.0, 14.0, 45.0
    if any(word in lowered for word in ("торт", "десерт", "мороженое", "чизкейк", "пирог", "блин")):
        return 6.0, 16.0, 40.0
    # Scale rough BJU to calories if we only know kcal.
    if calories <= 0:
        return 12.0, 10.0, 25.0
    protein = round(calories * 0.18 / 4, 1)
    fat = round(calories * 0.30 / 9, 1)
    carbs = round(calories * 0.52 / 4, 1)
    return protein, fat, carbs


def lines_to_dishes(lines: list[dict[str, Any]], restaurant_id: str | None = None) -> list[dict[str, Any]]:
    """Convert OCR lines ({text, confidence}) into dish dicts."""
    dishes: list[dict[str, Any]] = []
    seen: set[str] = set()

    for line in _merge_word_fragments(lines):
        raw = str(line.get("text") or "").strip()
        if not raw:
            continue
        conf = float(line.get("confidence") or 0)
        if conf and conf < 0.55:
            continue

        kcal_match = KCAL_RE.search(raw)
        weight_match = WEIGHT_RE.search(raw)
        name = _clean_name(KCAL_RE.sub("", WEIGHT_RE.sub("", raw)))
        if not _looks_like_dish(name):
            continue

        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)

        calories = int(kcal_match.group(1)) if kcal_match else _estimate_calories(name)
        weight = f"{weight_match.group(1)} г" if weight_match else "порция"
        portion_grams = int(weight_match.group(1)) if weight_match else 100
        calories_per_100 = round(calories * 100 / max(portion_grams, 1)) if weight_match else calories
        protein, fat, carbs = _estimate_macros(name, calories)

        dishes.append(
            {
                "id": f"ocr-{uuid.uuid4().hex[:10]}",
                "name": name,
                "calories": calories,
                "caloriesPer100g": calories_per_100,
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
                "weight": weight,
                "crop": _guess_crop(name),
                "confidence": round(conf, 3) if conf else None,
                "restaurantId": restaurant_id,
            }
        )

    return dishes[:24]


def _estimate_calories(name: str) -> int:
    lowered = name.casefold()
    if any(word in lowered for word in ("салат", "овощ", "зелень")):
        return 180
    if any(word in lowered for word in ("суп", "бульон", "щи", "борщ")):
        return 220
    if any(word in lowered for word in ("рыба", "лосось", "тунец")):
        return 380
    if any(word in lowered for word in ("курица", "индейка", "грудк", "куриный")):
        return 320
    if any(word in lowered for word in ("паста", "пицца", "рис", "лапш", "плов")):
        return 450
    if any(word in lowered for word in ("торт", "десерт", "мороженое", "чизкейк")):
        return 360
    return 280


def _guess_crop(name: str) -> str:
    lowered = name.casefold()
    if any(word in lowered for word in ("салат", "овощ", "зелень")):
        return "salad"
    if any(word in lowered for word in ("курица", "мясо", "стейк", "индейка")):
        return "chicken"
    if any(word in lowered for word in ("творог", "йогурт", "сырник", "десерт")):
        return "cottage"
    return "oats"
