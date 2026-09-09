"""PaddleOCR menu recognition service for Ритм Stage 1."""

from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .parse_menu import lines_to_dishes

app = FastAPI(title="Ritm PaddleOCR", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("OCR_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_ocr():
    # Import lazily so /health stays fast before models download.
    from paddleocr import PaddleOCR

    # lang=ru covers Cyrillic menus; English latin chars still work reasonably.
    return PaddleOCR(
        use_angle_cls=True,
        lang="ru",
        show_log=False,
        use_gpu=False,
    )


def _extract_lines(result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if not result:
        return lines

    # PaddleOCR classic API: list[page] -> list[[box, (text, score)]]
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        for item in page:
            try:
                if isinstance(item, dict):
                    text = item.get("transcription") or item.get("text") or ""
                    score = float(item.get("score") or item.get("confidence") or 0)
                else:
                    text = item[1][0]
                    score = float(item[1][1])
                text = str(text).strip()
                if text:
                    lines.append({"text": text, "confidence": score})
            except (IndexError, TypeError, ValueError, KeyError):
                continue
    return lines


@app.get("/health")
def health():
    return {"ok": True, "engine": "paddleocr", "lang": "ru"}


@app.get("/warmup")
def warmup():
    """Load PaddleOCR models into memory (cold start / keep-alive)."""
    try:
        get_ocr()
        return {"ok": True, "ready": True, "engine": "paddleocr", "lang": "ru"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"OCR warmup failed: {exc}") from exc


@app.on_event("startup")
def _preload_ocr_in_background() -> None:
    import threading

    def _run() -> None:
        try:
            get_ocr()
        except Exception:
            # First real request will retry; avoid crashing the process on boot.
            pass

    threading.Thread(target=_run, name="ocr-preload", daemon=True).start()


@app.post("/ocr/menu")
async def ocr_menu(
    file: UploadFile = File(...),
    restaurant_id: str | None = Form(default=None),
):
    content_type = (file.content_type or "").lower()
    # Browsers sometimes omit content-type or send octet-stream for camera captures.
    if content_type and not (
        content_type.startswith("image/") or content_type in ("application/octet-stream",)
    ):
        raise HTTPException(status_code=400, detail="Ожидается изображение (image/*)")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл больше 12 МБ")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось открыть изображение: {exc}") from exc

    # Downscale huge phone photos for speed/memory.
    max_side = 1600
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)))

    import numpy as np

    try:
        ocr = get_ocr()
        result = ocr.ocr(np.array(image), cls=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"PaddleOCR error: {exc}") from exc

    lines = _extract_lines(result)
    items = lines_to_dishes(lines, restaurant_id)

    return {
        "engine": "paddleocr",
        "restaurantId": restaurant_id,
        "photoName": file.filename or "menu.jpg",
        "lineCount": len(lines),
        "lines": lines[:80],
        "items": items,
        "recognizedAt": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
