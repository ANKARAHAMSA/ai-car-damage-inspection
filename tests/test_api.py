from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import app
from src.utils import load_config

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_valid_image_returns_200_and_pdf_report():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    sample_img = raw_dir / Path("data/splits/test.txt").read_text().splitlines()[0]

    with open(sample_img, "rb") as f:
        res = client.post("/inspect", files={"file": ("car.jpg", f, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    assert "verdict" in data
    assert "report_id" in data

    pdf_res = client.get(f"/report/{data['report_id']}")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF")

def test_corrupt_file_returns_400():
    res = client.post("/inspect", files={"file": ("corrupt.jpg", b"not-an-image-bytes", "image/jpeg")})
    assert res.status_code == 400

def test_non_image_type_rejected_415():
    res = client.post("/inspect", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert res.status_code == 415

def test_huge_file_rejected_413():
    huge_payload = b"0" * (10 * 1024 * 1024 + 100)
    res = client.post("/inspect", files={"file": ("huge.jpg", huge_payload, "image/jpeg")})
    assert res.status_code == 413
