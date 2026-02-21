"""DeepSeek-OCR2 web server powered by Unsloth."""

import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from unsloth import FastVisionModel
from transformers import AutoModel

MODEL_DIR = os.environ.get("MODEL_DIR", "./deepseek_ocr")

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_DIR,
        load_in_4bit=False,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth",
    )
    # Avoid "attention_mask not set" warning when pad == eos
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token = tokenizer.unk_token or "[PAD]"
    yield


app = FastAPI(title="DeepSeek-OCR2 Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(..., description="Image or PDF file to process"),
    prompt: str = Form(
        default="<image>\nFree OCR.",
        description="OCR prompt (see / for examples)",
    ),
):
    suffix = Path(file.filename).suffix.lower() if file.filename else ".png"
    raw = await file.read()

    if suffix == ".pdf":
        return await _ocr_pdf(raw, prompt)
    else:
        return await _ocr_image(raw, suffix, prompt)


async def _ocr_image(data: bytes, suffix: str, prompt: str) -> JSONResponse:
    """Run OCR on a single image and return the result."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = _infer(tmp_path, prompt)
        return JSONResponse({"result": result})
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _ocr_pdf(data: bytes, prompt: str) -> JSONResponse:
    """Convert each PDF page to an image, OCR it, then merge results."""
    doc = fitz.open(stream=data, filetype="pdf")
    page_texts: list[str] = []

    for page in doc:
        # Render at 2× for good OCR quality (144 DPI)
        pix = page.get_pixmap(dpi=144)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(pix.tobytes("png"))
            tmp_path = tmp.name
        try:
            text = _infer(tmp_path, prompt)
            page_texts.append(text)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    doc.close()
    merged = _merge_page_texts(page_texts)
    return JSONResponse({
        "result": merged,
        "pages": len(page_texts),
        "page_results": page_texts,
    })


def _infer(image_path: str, prompt: str) -> str:
    """Run the model on a single image file."""
    with tempfile.TemporaryDirectory() as output_dir:
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False,
        )
    return result


def _merge_page_texts(pages: list[str]) -> str:
    """Merge per-page OCR results, stitching text split across page breaks.

    Heuristics:
    - If a page ends mid-sentence (no sentence-ending punctuation) and the
      next page starts with a lowercase letter or continuation punctuation,
      join them with a single space instead of a double newline.
    - Otherwise, separate pages with a blank line.
    """
    if not pages:
        return ""
    if len(pages) == 1:
        return pages[0]

    _SENT_END = re.compile(r'[.!?:;»"\'"\)\]]\s*$')
    _CONT_START = re.compile(r'^\s*[a-z,;]')

    merged = pages[0]
    for prev_idx in range(len(pages) - 1):
        prev = pages[prev_idx].rstrip()
        nxt = pages[prev_idx + 1].lstrip()

        # Detect a mid-sentence page break
        if prev and nxt and not _SENT_END.search(prev) and _CONT_START.match(nxt):
            # Word was likely split by the page break → join with space
            if prev.endswith("-"):
                merged = merged.rstrip().removesuffix("-") + nxt
            else:
                merged = merged.rstrip() + " " + nxt
        else:
            merged = merged.rstrip() + "\n\n" + nxt

    return merged


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepSeek-OCR2 Server</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1 { color: #1a1a2e; }
  h2 { color: #16213e; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 32px; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; font-size: 0.88em; }
  pre { background: #f4f4f4; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 0.9em; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }
  th { background: #f4f4f4; font-size: 0.9em; }
  #drop-zone {
    border: 2px dashed #aaa; border-radius: 8px; padding: 40px 20px;
    text-align: center; cursor: pointer; transition: background 0.2s, border-color 0.2s;
    user-select: none;
  }
  #drop-zone.drag-over { background: #e8f4ff; border-color: #007bff; }
  #drop-zone p { margin: 0; color: #555; }
  #result { background: #f4f4f4; padding: 16px; border-radius: 8px; margin-top: 16px; white-space: pre-wrap; display: none; }
  select, button { padding: 8px 12px; border-radius: 6px; border: 1px solid #ccc; font-size: 0.95em; }
  select { width: 100%; margin-bottom: 12px; }
  button { background: #007bff; color: #fff; border-color: #007bff; cursor: pointer; margin-top: 8px; }
  button:hover { background: #0056b3; }
  #status { margin-top: 10px; color: #555; min-height: 1.4em; }
</style>
</head>
<body>
<h1>🔍 DeepSeek-OCR2 Server</h1>
<p>A minimalistic API server for <a href="https://huggingface.co/unsloth/DeepSeek-OCR-2" target="_blank">Unsloth's DeepSeek-OCR-2</a> model.</p>

<h2>Try It</h2>
<label for="prompt-select"><strong>Prompt:</strong></label>
<select id="prompt-select">
  <option value="&lt;image&gt;\nFree OCR.">Free OCR</option>
  <option value="&lt;image&gt;\n&lt;|grounding|&gt;Convert the document to markdown.">Document → Markdown</option>
  <option value="&lt;image&gt;\nOCR this image.">General OCR</option>
  <option value="&lt;image&gt;\nParse the figure.">Parse Figure</option>
  <option value="&lt;image&gt;\nDescribe this image in detail.">Describe Image</option>
</select>
<div id="drop-zone">
  <p>📁 Drag &amp; drop an image or PDF here, or <strong>click to select</strong></p>
</div>
<input type="file" id="file-input" style="display:none" accept="image/*,.pdf,application/pdf">
<div id="status"></div>
<pre id="result"></pre>

<script>
const zone = document.getElementById('drop-zone');
const input = document.getElementById('file-input');
const result = document.getElementById('result');
const status = document.getElementById('status');
const promptSelect = document.getElementById('prompt-select');

zone.addEventListener('click', () => input.click());
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); handleFile(e.dataTransfer.files[0]); });
input.addEventListener('change', () => { handleFile(input.files[0]); input.value = ''; });

async function handleFile(file) {
  if (!file) return;
  status.textContent = '⏳ Processing ' + file.name + '…';
  result.style.display = 'none';

  const form = new FormData();
  form.append('file', file);
  form.append('prompt', promptSelect.value);

  try {
    const res = await fetch('/ocr', { method: 'POST', body: form });
    const data = await res.json();
    result.textContent = JSON.stringify(data, null, 2);
    result.style.display = 'block';
    status.textContent = '✅ Done.';
  } catch (err) {
    status.textContent = '❌ Error: ' + err.message;
  }
}
</script>

<h2>Endpoints</h2>
<table>
  <tr><th>Method</th><th>Path</th><th>Description</th></tr>
  <tr><td><code>GET</code></td><td><code>/</code></td><td>This page</td></tr>
  <tr><td><code>GET</code></td><td><code>/health</code></td><td>Health check – returns model status as JSON</td></tr>
  <tr><td><code>POST</code></td><td><code>/ocr</code></td><td>Run OCR on an uploaded image or PDF; returns <code>{"result": "..."}</code></td></tr>
  <tr><td><code>GET</code></td><td><code>/docs</code></td><td>Interactive Swagger UI (auto-generated)</td></tr>
</table>

<h2>POST /ocr – Form Fields</h2>
<table>
  <tr><th>Field</th><th>Type</th><th>Required</th><th>Default</th><th>Description</th></tr>
  <tr><td><code>file</code></td><td>file</td><td>✅</td><td>–</td><td>Image or PDF to process (PNG, JPG, TIFF, PDF, …). Multi-page PDFs are split per-page and merged.</td></tr>
  <tr><td><code>prompt</code></td><td>string</td><td>No</td><td><code>&lt;image&gt;\\nFree OCR.</code></td><td>OCR instruction (see examples below)</td></tr>
</table>

<h2>Example curl</h2>
<pre>curl -X POST http://localhost:8000/ocr \\
  -F "file=@image.png" \\
  -F "prompt=&lt;image&gt;\\nFree OCR."</pre>

<h2>Prompt Examples</h2>
<table>
  <tr><th>Prompt</th><th>Use case</th></tr>
  <tr><td><code>&lt;image&gt;\\nFree OCR.</code></td><td>Plain text extraction</td></tr>
  <tr><td><code>&lt;image&gt;\\n&lt;|grounding|&gt;Convert the document to markdown.</code></td><td>Document → Markdown</td></tr>
  <tr><td><code>&lt;image&gt;\\nOCR this image.</code></td><td>General OCR</td></tr>
  <tr><td><code>&lt;image&gt;\\nParse the figure.</code></td><td>Figure / chart parsing</td></tr>
  <tr><td><code>&lt;image&gt;\\nDescribe this image in detail.</code></td><td>Image description</td></tr>
</table>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML
