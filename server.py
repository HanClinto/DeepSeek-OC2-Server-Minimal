"""DeepSeek-OCR2 web server powered by Unsloth."""

import asyncio
import json
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
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


@app.post("/ocr/stream")
async def ocr_stream(
    file: UploadFile = File(..., description="Image or PDF file to process"),
    prompt: str = Form(
        default="<image>\nFree OCR.",
        description="OCR prompt (see / for examples)",
    ),
):
    """SSE streaming endpoint – sends progress events while processing."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ".png"
    filename = file.filename or "file"
    raw = await file.read()

    async def event_generator():
        t0 = time.monotonic()

        if suffix == ".pdf":
            doc = fitz.open(stream=data_holder[0], filetype="pdf")
            total = len(doc)
            yield _sse("progress", {"step": "start", "filename": filename,
                                     "total_pages": total, "message": f"Processing {total}-page PDF…"})

            page_texts: list[str] = []
            for i, page in enumerate(doc):
                yield _sse("progress", {"step": "page_start", "page": i + 1,
                                         "total_pages": total,
                                         "message": f"OCR page {i + 1}/{total}…"})
                pix = page.get_pixmap(dpi=144)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(pix.tobytes("png"))
                    tmp_path = tmp.name
                try:
                    text = await asyncio.to_thread(_infer, tmp_path, prompt_holder[0])
                    page_texts.append(text)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                yield _sse("progress", {"step": "page_done", "page": i + 1,
                                         "total_pages": total,
                                         "message": f"Page {i + 1}/{total} complete."})
            doc.close()

            merged = _merge_page_texts(page_texts)
            elapsed = round(time.monotonic() - t0, 1)
            yield _sse("result", {"result": merged, "pages": total,
                                   "page_results": page_texts,
                                   "elapsed_seconds": elapsed})
        else:
            yield _sse("progress", {"step": "start", "filename": filename,
                                     "total_pages": 1, "message": "Processing image…"})
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_holder[0]) as tmp:
                tmp.write(data_holder[0])
                tmp_path = tmp.name
            try:
                text = await asyncio.to_thread(_infer, tmp_path, prompt_holder[0])
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            elapsed = round(time.monotonic() - t0, 1)
            yield _sse("result", {"result": text, "elapsed_seconds": elapsed})

    # Closures can't capture mutable locals across async generators easily,
    # so stash values in lists.
    data_holder = [raw]
    prompt_holder = [prompt]
    suffix_holder = [suffix]

    return EventSourceResponse(event_generator())


def _sse(event: str, data: dict) -> dict:
    """Format a dict as an SSE event for sse-starlette."""
    return {"event": event, "data": json.dumps(data)}


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
  #progress-wrap {
    margin-top: 14px; display: none;
  }
  #progress-bar-outer {
    background: #e0e0e0; border-radius: 8px; height: 22px; overflow: hidden; position: relative;
  }
  #progress-bar-inner {
    background: linear-gradient(90deg, #007bff 0%, #00b4d8 100%);
    height: 100%; width: 0%; border-radius: 8px;
    transition: width 0.4s ease;
  }
  #progress-bar-inner.indeterminate {
    width: 100%;
    background: linear-gradient(90deg, #007bff 0%, #00b4d8 50%, #007bff 100%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
  }
  @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
  #progress-text {
    font-size: 0.85em; color: #555; margin-top: 4px;
  }
  #elapsed { font-size: 0.8em; color: #888; margin-left: 8px; }
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
<div id="progress-wrap">
  <div id="progress-bar-outer"><div id="progress-bar-inner"></div></div>
  <div id="progress-text"></div>
</div>
<pre id="result"></pre>

<script>
const zone = document.getElementById('drop-zone');
const input = document.getElementById('file-input');
const result = document.getElementById('result');
const status = document.getElementById('status');
const promptSelect = document.getElementById('prompt-select');
const progressWrap = document.getElementById('progress-wrap');
const progressBar = document.getElementById('progress-bar-inner');
const progressText = document.getElementById('progress-text');

zone.addEventListener('click', () => input.click());
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); handleFile(e.dataTransfer.files[0]); });
input.addEventListener('change', () => { handleFile(input.files[0]); input.value = ''; });

function resetProgress() {
  progressWrap.style.display = 'none';
  progressBar.style.width = '0%';
  progressBar.classList.remove('indeterminate');
  progressText.textContent = '';
}

let startTime = 0;
let elapsedInterval = null;

function startTimer() {
  startTime = Date.now();
  if (elapsedInterval) clearInterval(elapsedInterval);
  elapsedInterval = setInterval(() => {
    const s = ((Date.now() - startTime) / 1000).toFixed(0);
    const el = document.getElementById('elapsed');
    if (el) el.textContent = s + 's elapsed';
  }, 500);
}

function stopTimer() {
  if (elapsedInterval) { clearInterval(elapsedInterval); elapsedInterval = null; }
}

async function handleFile(file) {
  if (!file) return;
  resetProgress();
  result.style.display = 'none';
  status.innerHTML = '⏳ Uploading <strong>' + file.name + '</strong>… <span id="elapsed"></span>';
  progressWrap.style.display = 'block';
  progressBar.classList.add('indeterminate');
  progressText.textContent = 'Uploading…';
  startTimer();

  const form = new FormData();
  form.append('file', file);
  form.append('prompt', promptSelect.value);

  try {
    const res = await fetch('/ocr/stream', { method: 'POST', body: form });
    if (!res.ok) throw new Error('Server returned ' + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop();  // keep incomplete line in buffer

      let currentEvent = null;
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ') && currentEvent) {
          const data = JSON.parse(line.slice(6));
          handleSSE(currentEvent, data);
          currentEvent = null;
        }
      }
    }
  } catch (err) {
    stopTimer();
    status.textContent = '❌ Error: ' + err.message;
    resetProgress();
  }
}

function handleSSE(event, data) {
  if (event === 'progress') {
    progressBar.classList.remove('indeterminate');
    status.innerHTML = '⏳ <strong>' + data.message + '</strong> <span id="elapsed"></span>';
    progressText.textContent = data.message;

    if (data.total_pages && data.total_pages > 1) {
      if (data.step === 'page_done') {
        const pct = Math.round((data.page / data.total_pages) * 100);
        progressBar.style.width = pct + '%';
      } else if (data.step === 'page_start') {
        const pct = Math.round(((data.page - 1) / data.total_pages) * 100);
        progressBar.style.width = pct + '%';
      } else if (data.step === 'start') {
        progressBar.style.width = '0%';
      }
    } else {
      progressBar.classList.add('indeterminate');
    }
  } else if (event === 'result') {
    stopTimer();
    const elapsed = data.elapsed_seconds ? ' (' + data.elapsed_seconds + 's)' : '';
    progressBar.classList.remove('indeterminate');
    progressBar.style.width = '100%';
    status.textContent = '✅ Done' + elapsed;
    progressText.textContent = 'Complete' + elapsed;
    result.textContent = JSON.stringify(data, null, 2);
    result.style.display = 'block';
  }
}
</script>

<h2>Endpoints</h2>
<table>
  <tr><th>Method</th><th>Path</th><th>Description</th></tr>
  <tr><td><code>GET</code></td><td><code>/</code></td><td>This page</td></tr>
  <tr><td><code>GET</code></td><td><code>/health</code></td><td>Health check – returns model status as JSON</td></tr>
  <tr><td><code>POST</code></td><td><code>/ocr</code></td><td>Run OCR on an uploaded image or PDF; returns <code>{"result": "..."}</code></td></tr>
  <tr><td><code>POST</code></td><td><code>/ocr/stream</code></td><td>Same as /ocr but streams progress via Server-Sent Events</td></tr>
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
