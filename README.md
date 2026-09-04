# PaddleOCR menu service

FastAPI-обёртка над **настоящим** PaddleOCR (язык `ru`) для экрана «Фото меню».

## Локальный запуск (Windows)

```powershell
cd ocr-service
py -3.11 -m venv .venv   # или путь к Python 3.11
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

В корне PWA в `.env`:

```env
VITE_OCR_URL=http://127.0.0.1:8080
```

Перезапустите Vite. Кнопка «Демо без фото» остаётся mock-фолбэком; фото уходит на `/ocr/menu`.

## Docker

```bash
docker build -t ritm-ocr .
docker run --rm -p 8080:8080 ritm-ocr
```

## API

- `GET /health` — живость сервиса
- `POST /ocr/menu` — `multipart/form-data`: `file` (image), опционально `restaurant_id`
