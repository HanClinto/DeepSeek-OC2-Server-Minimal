# DeepSeek-OC2-Server-Minimal

A minimalistic web server for running [Unsloth's DeepSeek-OCR-2](https://huggingface.co/unsloth/DeepSeek-OCR-2) via web requests.

## Requirements

- Linux with a CUDA-capable NVIDIA GPU
- [uv](https://astral.sh/uv) (installed automatically by `install.sh` if missing)
- Internet access (to download the model from Hugging Face on first run)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/HanClinto/DeepSeek-OC2-Server-Minimal.git
cd DeepSeek-OC2-Server-Minimal

# 2. Install dependencies and download the model (~10 GB)
chmod +x install.sh start.sh
./install.sh

# 3. Start the server
./start.sh
```

The server starts at **http://localhost:8000**.

## API Endpoints

| Method | Path      | Description                                      |
|--------|-----------|--------------------------------------------------|
| GET    | `/`       | Self-documenting webpage with a drag-and-drop UI |
| GET    | `/health` | Health check – returns model status as JSON      |
| POST   | `/ocr`    | Run OCR on an uploaded image                     |
| GET    | `/docs`   | Interactive Swagger UI (auto-generated)          |

### POST /ocr

Accepts `multipart/form-data`:

| Field    | Type   | Required | Default                      | Description            |
|----------|--------|----------|------------------------------|------------------------|
| `file`   | file   | ✅       | –                            | Image (PNG, JPG, PDF…) |
| `prompt` | string | No       | `<image>\nFree OCR.`         | OCR instruction        |

Returns JSON:
```json
{"result": "extracted text..."}
```

### Example

```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@document.png" \
  -F "prompt=<image>\nFree OCR."
```

## Prompt Examples

| Prompt | Use case |
|--------|----------|
| `<image>\nFree OCR.` | Plain text extraction |
| `<image>\n<\|grounding\|>Convert the document to markdown.` | Document → Markdown |
| `<image>\nOCR this image.` | General OCR |
| `<image>\nParse the figure.` | Figure / chart parsing |
| `<image>\nDescribe this image in detail.` | Image description |

## Configuration

| Environment variable | Default       | Description             |
|----------------------|---------------|-------------------------|
| `MODEL_DIR`          | `./deepseek_ocr` | Path to model weights |
| `HOST`               | `0.0.0.0`     | Bind address            |
| `PORT`               | `8000`        | Port                    |

## Package Management

All Python dependencies are managed with [uv](https://astral.sh/uv):

```bash
# Install / sync dependencies
uv sync

# Add a new dependency
uv add <package>

# Run the server manually
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```
