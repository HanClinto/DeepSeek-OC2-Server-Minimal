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
                                         "page_text": text or "",
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
            yield _sse("result", {"result": text or "", "elapsed_seconds": elapsed})

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
            eval_mode=True,
        )
    return result or ""


def _merge_page_texts(pages: list[str]) -> str:
    """Merge per-page OCR results, stitching text split across page breaks.

    Heuristics:
    - If a page ends mid-sentence (no sentence-ending punctuation) and the
      next page starts with a lowercase letter or continuation punctuation,
      join them with a single space instead of a double newline.
    - Otherwise, separate pages with a blank line.
    """
    # Coerce any None entries to empty strings
    pages = [p if p is not None else "" for p in pages]

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
  :root {
    --bg: #ffffff; --fg: #222; --fg-muted: #555; --fg-faint: #888;
    --surface: #f4f4f4; --surface2: #f9f9f9; --surface3: #f0f0f0;
    --border: #ddd; --border2: #eee; --border3: #e0e0e0; --border-table: #ccc;
    --accent: #007bff; --accent-hover: #0056b3;
    --h1: #1a1a2e; --h2: #16213e;
    --drop-hover-bg: #e8f4ff;
    --link: #007bff;
  }
  [data-theme="dark"] {
    --bg: #1a1a2e; --fg: #e0e0e0; --fg-muted: #aaa; --fg-faint: #777;
    --surface: #2a2a3e; --surface2: #242438; --surface3: #2e2e44;
    --border: #444; --border2: #3a3a50; --border3: #444; --border-table: #555;
    --accent: #4dabf7; --accent-hover: #74c0fc;
    --h1: #c5cae9; --h2: #9fa8da;
    --drop-hover-bg: #2a3a5e;
    --link: #74c0fc;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #1a1a2e; --fg: #e0e0e0; --fg-muted: #aaa; --fg-faint: #777;
      --surface: #2a2a3e; --surface2: #242438; --surface3: #2e2e44;
      --border: #444; --border2: #3a3a50; --border3: #444; --border-table: #555;
      --accent: #4dabf7; --accent-hover: #74c0fc;
      --h1: #c5cae9; --h2: #9fa8da;
      --drop-hover-bg: #2a3a5e;
      --link: #74c0fc;
    }
  }
  body { font-family: system-ui, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 20px; color: var(--fg); background: var(--bg); transition: background 0.3s, color 0.3s; }
  a { color: var(--link); }
  h1 { color: var(--h1); }
  h2 { color: var(--h2); border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-top: 32px; }
  code { background: var(--surface); padding: 2px 6px; border-radius: 4px; font-size: 0.88em; }
  pre { background: var(--surface); padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 0.9em; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border2); }
  th { background: var(--surface); font-size: 0.9em; }
  .header-row { display: flex; align-items: center; justify-content: space-between; }
  #theme-toggle {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 6px 12px; cursor: pointer; font-size: 1.1em; line-height: 1;
    color: var(--fg); transition: background 0.2s;
  }
  #theme-toggle:hover { background: var(--surface3); }
  #drop-zone {
    border: 2px dashed var(--border); border-radius: 8px; padding: 40px 20px;
    text-align: center; cursor: pointer; transition: background 0.2s, border-color 0.2s;
    user-select: none;
  }
  #drop-zone.drag-over { background: var(--drop-hover-bg); border-color: var(--accent); }
  #drop-zone p { margin: 0; color: var(--fg-muted); }
  #result { background: var(--surface); padding: 16px; border-radius: 8px; margin-top: 16px; white-space: pre-wrap; display: none; }
  #output-panel { display: none; margin-top: 16px; }
  .tab-bar { display: flex; gap: 0; border-bottom: 2px solid var(--border); }
  .tab-btn {
    padding: 8px 18px; cursor: pointer; border: none; background: transparent;
    font-size: 0.95em; color: var(--fg-muted); border-bottom: 2px solid transparent;
    margin-bottom: -2px; transition: all 0.2s;
  }
  .tab-btn:hover { color: var(--accent); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
  .tab-content { display: none; padding: 16px; background: var(--surface2); border: 1px solid var(--border3); border-top: none; border-radius: 0 0 8px 8px; min-height: 120px; max-height: 600px; overflow: auto; }
  .tab-content.active { display: block; }
  #raw-tab { white-space: pre-wrap; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.88em; }
  #rendered-tab { font-size: 0.95em; line-height: 1.6; }
  #rendered-tab h1, #rendered-tab h2, #rendered-tab h3 { margin-top: 0.8em; }
  #rendered-tab table { border-collapse: collapse; margin: 0.5em 0; }
  #rendered-tab th, #rendered-tab td { border: 1px solid var(--border-table); padding: 4px 8px; }
  #rendered-tab pre { background: var(--surface3); padding: 10px; border-radius: 4px; overflow-x: auto; }
  #rendered-tab code { background: var(--surface3); padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
  #rendered-tab img { max-width: 100%; }
  #json-tab { white-space: pre-wrap; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.85em; color: var(--fg-muted); }
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
    font-size: 0.85em; color: var(--fg-muted); margin-top: 4px;
  }
  #elapsed { font-size: 0.8em; color: var(--fg-faint); margin-left: 8px; }
  select, button { padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); font-size: 0.95em; background: var(--surface); color: var(--fg); }
  select { width: 100%; margin-bottom: 12px; }
  button:not(#theme-toggle) { background: var(--accent); color: #fff; border-color: var(--accent); cursor: pointer; margin-top: 8px; }
  button:not(#theme-toggle):hover { background: var(--accent-hover); }
  #status { margin-top: 10px; color: var(--fg-muted); min-height: 1.4em; }
</style>
</head>
<body>
<div class="header-row">
  <h1>🔍 DeepSeek-OCR2 Server</h1>
  <button id="theme-toggle" title="Toggle dark/light mode">🌙</button>
</div>
<p>A minimalistic API server for <a href="https://huggingface.co/unsloth/DeepSeek-OCR-2" target="_blank">Unsloth's DeepSeek-OCR-2</a> model.</p>
<script>
(function() {
  const toggle = document.getElementById('theme-toggle');
  const root = document.documentElement;
  const stored = localStorage.getItem('theme');
  if (stored) {
    root.setAttribute('data-theme', stored);
  }
  function updateIcon() {
    const isDark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    toggle.textContent = isDark ? '☀️' : '🌙';
  }
  updateIcon();
  toggle.addEventListener('click', () => {
    const isDark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    const next = isDark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateIcon();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateIcon);
})();
</script>

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
<pre id="result" style="display:none;"></pre>
<div id="output-panel">
  <div class="tab-bar">
    <button class="tab-btn active" data-tab="rendered-tab">Rendered</button>
    <button class="tab-btn" data-tab="raw-tab">Raw Markdown</button>
    <button class="tab-btn" data-tab="json-tab">JSON</button>
  </div>
  <div id="rendered-tab" class="tab-content active"></div>
  <div id="raw-tab" class="tab-content"></div>
  <div id="json-tab" class="tab-content"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<script>
const zone = document.getElementById('drop-zone');
const input = document.getElementById('file-input');
const status = document.getElementById('status');
const promptSelect = document.getElementById('prompt-select');
const progressWrap = document.getElementById('progress-wrap');
const progressBar = document.getElementById('progress-bar-inner');
const progressText = document.getElementById('progress-text');
const outputPanel = document.getElementById('output-panel');
const rawTab = document.getElementById('raw-tab');
const renderedTab = document.getElementById('rendered-tab');
const jsonTab = document.getElementById('json-tab');

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

let accumulatedText = '';

function updateOutputPanels(text) {
  accumulatedText = text;
  rawTab.textContent = text;
  try { renderedTab.innerHTML = marked.parse(text); } catch(e) { renderedTab.textContent = text; }
  outputPanel.style.display = 'block';
}

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
  accumulatedText = '';
  rawTab.textContent = '';
  renderedTab.innerHTML = '';
  jsonTab.textContent = '';
  outputPanel.style.display = 'none';
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
      buffer = lines.pop();

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

    // Stream page text as each page completes
    if (data.step === 'page_done' && data.page_text) {
      if (accumulatedText) accumulatedText += '\\n\\n';
      accumulatedText += data.page_text;
      updateOutputPanels(accumulatedText);
    }

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

    // Final update with merged/complete text
    updateOutputPanels(data.result || accumulatedText);
    jsonTab.textContent = JSON.stringify(data, null, 2);
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
