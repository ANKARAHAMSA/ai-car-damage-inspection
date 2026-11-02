import io
import uuid
import base64
from pathlib import Path
from typing import List
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from src.models.pipeline import InspectionPipeline
from src.api.pdf_report import generate_pdf_report
from src.data.auto_annotate import save_annotated_sample, trigger_incremental_finetune

app = FastAPI(title="AI Car Damage Inspection & Auto-Annotation API", version="2.0.0")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

pipeline = None

def get_pipeline() -> InspectionPipeline:
    global pipeline
    if pipeline is None:
        pipeline = InspectionPipeline()
    return pipeline

class BoxItem(BaseModel):
    type: str
    box: List[float]
    conf: float = 1.0

class SaveAnnotationRequest(BaseModel):
    image_name: str
    image_b64: str
    detections: List[BoxItem]

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-car-damage-inspection"}

@app.post("/inspect")
async def inspect_car(file: UploadFile = File(...)):
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG or PNG image.")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 10 MB.")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="Corrupt or invalid image file.")

    pipe = get_pipeline()
    result, ann_img = pipe.predict(pil_img)

    report_id = str(uuid.uuid4())[:12]
    generate_pdf_report(report_id, pil_img, ann_img, result)

    buf_ann = io.BytesIO()
    ann_img.save(buf_ann, format="JPEG", quality=85)
    b64_ann = base64.b64encode(buf_ann.getvalue()).decode("utf-8")

    buf_orig = io.BytesIO()
    pil_img.save(buf_orig, format="JPEG", quality=90)
    b64_orig = base64.b64encode(buf_orig.getvalue()).decode("utf-8")

    result["report_id"] = report_id
    result["report_url"] = f"/report/{report_id}"
    result["annotated_image_b64"] = f"data:image/jpeg;base64,{b64_ann}"
    result["original_image_b64"] = f"data:image/jpeg;base64,{b64_orig}"
    result["image_width"] = pil_img.width
    result["image_height"] = pil_img.height
    return result

@app.post("/auto-annotate/save")
def save_annotation_endpoint(req: SaveAnnotationRequest):
    header, b64_data = req.image_b64.split(",", 1) if "," in req.image_b64 else ("", req.image_b64)
    img_bytes = base64.b64decode(b64_data)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    dets = [{"type": d.type, "box": d.box, "conf": d.conf} for d in req.detections]
    info = save_annotated_sample(pil_img, req.image_name, dets)
    return info

@app.post("/auto-annotate/finetune")
def finetune_endpoint():
    global pipeline
    res = trigger_incremental_finetune(epochs=3)
    pipeline = InspectionPipeline()  # Reload updated weights
    return res

@app.get("/report/{report_id}")
def download_report(report_id: str):
    safe_id = "".join(c for c in report_id if c.isalnum() or c == "-")
    pdf_path = Path("reports/generated") / f"{safe_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Inspection report not found.")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=f"car_inspection_{safe_id}.pdf")

@app.get("/", response_class=HTMLResponse)
def web_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Car Damage Inspection & Auto-Annotation Studio</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 28px; }
    .container { max-width: 1100px; margin: 0 auto; }
    h1 { font-size: 26px; margin-bottom: 4px; }
    .sub { color: #94a3b8; margin-bottom: 20px; font-size: 14px; }
    .tabs { display: flex; gap: 12px; margin-bottom: 20px; }
    .tab { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .tab.active { background: #3b82f6; color: white; border-color: #3b82f6; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .btn { background: #3b82f6; color: white; border: none; padding: 9px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block; font-size: 13px; }
    .btn:hover { background: #2563eb; }
    .btn-green { background: #10b981; }
    .btn-green:hover { background: #059669; }
    .btn-purple { background: #8b5cf6; }
    .btn-purple:hover { background: #7c3aed; }
    .badge { padding: 6px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; text-transform: uppercase; display: inline-block; }
    .sev-none { background: #065f46; color: #6ee7b7; }
    .sev-minor { background: #78350f; color: #fcd34d; }
    .sev-moderate { background: #9a3412; color: #fdba74; }
    .sev-severe { background: #7f1d1d; color: #fca5a5; }
    .grid { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 20px; margin-top: 16px; }
    img { max-width: 100%; border-radius: 8px; border: 1px solid #475569; }
    pre { background: #0f172a; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 12px; color: #cbd5e1; max-height: 420px; }
    canvas { border: 2px solid #3b82f6; border-radius: 8px; cursor: crosshair; max-width: 100%; }
    select, input[type="text"] { background: #0f172a; color: white; border: 1px solid #475569; padding: 7px 10px; border-radius: 6px; }
    .box-item { display: flex; justify-content: space-between; align-items: center; background: #0f172a; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 13px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🚗 AI Car Damage Inspection & Auto-Annotation System</h1>
    <div class="sub">Stage 1: Calibrated ConvNeXt-Tiny &rarr; Stage 2: Multi-Threshold YOLO11 &rarr; Stage 3: Severity Scorer + Active Learning Studio</div>

    <div class="tabs">
      <div id="tabInspect" class="tab active" onclick="switchTab('inspect')">🔍 1. Vehicle Damage Inspection & PDF Report</div>
      <div id="tabStudio" class="tab" onclick="switchTab('studio')">✍️ 2. AI Auto-Annotation & Active Learning Studio</div>
    </div>

    <div class="card">
      <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp" />
      <button class="btn" onclick="runInspection()">Run AI Inspection & Auto-Label</button>
      <span id="status" style="margin-left:12px; color:#94a3b8;"></span>
    </div>

    <!-- TAB 1: INSPECTION VIEW -->
    <div id="inspectPane" style="display:block;">
      <div id="resultCard" class="card" style="display:none;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <span id="verdictBadge" class="badge"></span>
            <span id="sevBadge" class="badge" style="margin-left:8px;"></span>
            <span id="reviewFlag" style="margin-left:12px; color:#fbbf24; font-weight:600;"></span>
          </div>
          <a id="pdfLink" class="btn" href="#" target="_blank">📄 Download PDF Report</a>
        </div>
        <div class="grid">
          <div>
            <h4>Annotated Vehicle Inspection</h4>
            <img id="annImg" src="" />
          </div>
          <div>
            <h4>Structured Pipeline JSON Output</h4>
            <pre id="jsonOut"></pre>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: AI AUTO-ANNOTATION STUDIO -->
    <div id="studioPane" style="display:none;">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <div>
            <strong>Draw new box with mouse:</strong> Select class
            <select id="newBoxClass">
              <option value="dent">dent</option>
              <option value="scratch">scratch</option>
              <option value="crack_gap">crack_gap</option>
              <option value="severe_replace">severe_replace</option>
            </select>
            <span style="color:#94a3b8; font-size:13px; margin-left:8px;">(Click & drag on the photo below to add any missed dent/scratch!)</span>
          </div>
          <div>
            <button class="btn btn-green" onclick="saveAnnotations()">💾 Save to Dataset (LabelMe JSON + YOLO)</button>
            <button class="btn btn-purple" onclick="runFinetune()" style="margin-left:8px;">🚀 Fine-Tune AI on New Data</button>
          </div>
        </div>
        <div id="studioStatus" style="color:#6ee7b7; font-size:13px; margin-bottom:10px;"></div>
        <div class="grid">
          <div>
            <canvas id="annotCanvas" width="640" height="480"></canvas>
          </div>
          <div>
            <h4>AI Pre-Labeled Boxes (Editable)</h4>
            <div id="boxList">Upload an image above to start AI auto-annotation!</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentOrigB64 = "";
    let currentFileName = "unseen_car.jpg";
    let currentBoxes = [];
    let imgObj = new Image();
    let isDrawing = false;
    let startX = 0, startY = 0;

    function switchTab(t) {
      document.getElementById('tabInspect').className = 'tab' + (t === 'inspect' ? ' active' : '');
      document.getElementById('tabStudio').className = 'tab' + (t === 'studio' ? ' active' : '');
      document.getElementById('inspectPane').style.display = t === 'inspect' ? 'block' : 'none';
      document.getElementById('studioPane').style.display = t === 'studio' ? 'block' : 'none';
      if (t === 'studio' && currentOrigB64) renderCanvas();
    }

    async function runInspection() {
      const input = document.getElementById('fileInput');
      if (!input.files.length) return alert('Please select a car photo first!');
      currentFileName = input.files[0].name;
      document.getElementById('status').innerText = 'Running 3-stage AI inspection & auto-annotation...';
      const formData = new FormData();
      formData.append('file', input.files[0]);
      const res = await fetch('/inspect', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        document.getElementById('status').innerText = 'Error: ' + (data.detail || 'Failed');
        return;
      }
      document.getElementById('status').innerText = 'Done in ' + data.latency_ms.total + ' ms! (' + data.detections.length + ' damage regions auto-labeled)';
      document.getElementById('resultCard').style.display = 'block';
      document.getElementById('annImg').src = data.annotated_image_b64;
      document.getElementById('pdfLink').href = data.report_url;

      const vBadge = document.getElementById('verdictBadge');
      vBadge.innerText = 'Verdict: ' + data.verdict + ' (' + (data.confidence * 100).toFixed(1) + '%)';
      vBadge.className = 'badge ' + (data.verdict === 'damaged' ? 'sev-severe' : 'sev-none');
      const sBadge = document.getElementById('sevBadge');
      sBadge.innerText = 'Severity: ' + data.severity;
      sBadge.className = 'badge sev-' + data.severity;
      document.getElementById('reviewFlag').innerText = data.needs_review ? '⚠️ Needs Human Review' : '✅ High-Confidence Pass';

      currentOrigB64 = data.original_image_b64;
      currentBoxes = JSON.parse(JSON.stringify(data.detections));

      const cleanJson = Object.assign({}, data);
      delete cleanJson.annotated_image_b64;
      delete cleanJson.original_image_b64;
      document.getElementById('jsonOut').innerText = JSON.stringify(cleanJson, null, 2);

      imgObj.onload = () => { renderCanvas(); renderBoxList(); };
      imgObj.src = currentOrigB64;
    }

    const canvas = document.getElementById('annotCanvas');
    const ctx = canvas.getContext('2d');

    function renderCanvas() {
      if (!imgObj.src) return;
      canvas.width = imgObj.width;
      canvas.height = imgObj.height;
      ctx.drawImage(imgObj, 0, 0);
      const colors = { scratch: '#ffd700', dent: '#ff7800', crack_gap: '#ff3232', severe_replace: '#d228ff' };
      currentBoxes.forEach((b, idx) => {
        const [x1, y1, x2, y2] = b.box;
        ctx.strokeStyle = colors[b.type] || '#3b82f6';
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = colors[b.type] || '#3b82f6';
        ctx.fillRect(x1, Math.max(0, y1 - 24), 170, 24);
        ctx.fillStyle = '#000';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('#' + (idx + 1) + ' ' + b.type + ' (' + Math.round(b.conf * 100) + '%)', x1 + 4, Math.max(16, y1 - 6));
      });
    }

    function renderBoxList() {
      const container = document.getElementById('boxList');
      if (!currentBoxes.length) {
        container.innerHTML = '<div style="color:#94a3b8;">No boxes yet. Drag on the image to draw one!</div>';
        return;
      }
      container.innerHTML = currentBoxes.map((b, i) => `
        <div class="box-item">
          <div>
            <strong>#${i+1}</strong>
            <select onchange="changeBoxType(${i}, this.value)">
              <option value="scratch" ${b.type==='scratch'?'selected':''}>scratch</option>
              <option value="dent" ${b.type==='dent'?'selected':''}>dent</option>
              <option value="crack_gap" ${b.type==='crack_gap'?'selected':''}>crack_gap</option>
              <option value="severe_replace" ${b.type==='severe_replace'?'selected':''}>severe_replace</option>
            </select>
            <span style="color:#94a3b8; margin-left:6px;">[${b.box.map(x=>Math.round(x)).join(', ')}]</span>
          </div>
          <button class="btn" style="background:#ef4444; padding:4px 8px;" onclick="deleteBox(${i})">✕</button>
        </div>
      `).join('');
    }

    function changeBoxType(i, val) { currentBoxes[i].type = val; renderCanvas(); }
    function deleteBox(i) { currentBoxes.splice(i, 1); renderCanvas(); renderBoxList(); }

    canvas.addEventListener('mousedown', (e) => {
      if (!imgObj.src) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      startX = (e.clientX - rect.left) * scaleX;
      startY = (e.clientY - rect.top) * scaleY;
      isDrawing = true;
    });

    canvas.addEventListener('mouseup', (e) => {
      if (!isDrawing) return;
      isDrawing = false;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const endX = (e.clientX - rect.left) * scaleX;
      const endY = (e.clientY - rect.top) * scaleY;
      if (Math.abs(endX - startX) > 10 && Math.abs(endY - startY) > 10) {
        const cls = document.getElementById('newBoxClass').value;
        currentBoxes.push({
          type: cls,
          box: [Math.min(startX, endX), Math.min(startY, endY), Math.max(startX, endX), Math.max(startY, endY)],
          conf: 1.0
        });
        renderCanvas();
        renderBoxList();
      }
    });

    async function saveAnnotations() {
      if (!currentOrigB64) return alert('Upload an image first!');
      const res = await fetch('/auto-annotate/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_name: currentFileName, image_b64: currentOrigB64, detections: currentBoxes })
      });
      const data = await res.json();
      document.getElementById('studioStatus').innerText = '✅ Saved LabelMe JSON (' + data.saved_labelme_json + ') & YOLO TXT! Total new samples: ' + data.total_new_samples_collected;
    }

    async function runFinetune() {
      document.getElementById('studioStatus').innerText = '⏳ Fine-tuning YOLO11 detector on your newly annotated vehicles (takes ~30s)...';
      const res = await fetch('/auto-annotate/finetune', { method: 'POST' });
      const data = await res.json();
      document.getElementById('studioStatus').innerText = '🎉 Active Learning Complete! Model weights updated (' + data.updated_model + ')!';
    }
  </script>
</body>
</html>
"""
