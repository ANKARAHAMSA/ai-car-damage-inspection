import io
import uuid
import base64
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from src.models.pipeline import InspectionPipeline
from src.api.pdf_report import generate_pdf_report

app = FastAPI(title="AI Car Damage Inspection API", version="1.0.0")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

pipeline = None

def get_pipeline() -> InspectionPipeline:
    global pipeline
    if pipeline is None:
        pipeline = InspectionPipeline()
    return pipeline

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

    buf = io.BytesIO()
    ann_img.save(buf, format="JPEG", quality=85)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    result["report_id"] = report_id
    result["report_url"] = f"/report/{report_id}"
    result["annotated_image_b64"] = f"data:image/jpeg;base64,{b64_img}"
    return result

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
  <title>AI Car Damage Inspection System</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 32px; }
    .container { max-width: 960px; margin: 0 auto; }
    h1 { font-size: 28px; margin-bottom: 6px; }
    .sub { color: #94a3b8; margin-bottom: 24px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    .btn { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block; }
    .btn:hover { background: #2563eb; }
    .badge { padding: 6px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; text-transform: uppercase; display: inline-block; }
    .sev-none { background: #065f46; color: #6ee7b7; }
    .sev-minor { background: #78350f; color: #fcd34d; }
    .sev-moderate { background: #9a3412; color: #fdba74; }
    .sev-severe { background: #7f1d1d; color: #fca5a5; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }
    img { max-width: 100%; border-radius: 8px; border: 1px solid #475569; }
    pre { background: #0f172a; padding: 14px; border-radius: 8px; overflow-x: auto; font-size: 13px; color: #cbd5e1; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🚗 AI Car Damage Inspection System</h1>
    <div class="sub">Quality Gate &rarr; Stage 1: Calibrated ConvNeXt-Tiny Classifier &rarr; Stage 2: YOLO11 Detector &rarr; Stage 3: Severity Scorer + PDF Report</div>
    <div class="card">
      <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp" />
      <button class="btn" onclick="runInspection()">Run AI Inspection</button>
      <span id="status" style="margin-left:12px; color:#94a3b8;"></span>
    </div>
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
  <script>
    async function runInspection() {
      const input = document.getElementById('fileInput');
      if (!input.files.length) return alert('Please select a car photo first!');
      document.getElementById('status').innerText = 'Inspecting image across 3-stage pipeline...';
      const formData = new FormData();
      formData.append('file', input.files[0]);
      const res = await fetch('/inspect', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        document.getElementById('status').innerText = 'Error: ' + (data.detail || 'Failed');
        return;
      }
      document.getElementById('status').innerText = 'Done in ' + data.latency_ms.total + ' ms!';
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
      const cleanJson = Object.assign({}, data);
      delete cleanJson.annotated_image_b64;
      document.getElementById('jsonOut').innerText = JSON.stringify(cleanJson, null, 2);
    }
  </script>
</body>
</html>
"""
