import time
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from ultralytics import YOLO
from src.utils import load_config
from src.models.train_classifier import DamageClassifier
from src.models.severity_scorer import extract_features_from_detections, rule_based_severity
from src.models.baselines import get_device

class InspectionPipeline:
    def __init__(self, config_path="configs/default.yaml", device=None):
        self.cfg = load_config(config_path)
        self.device = torch.device(device) if device else get_device()
        self.q_cfg = self.cfg["quality_gate"]

        ckpt = torch.load("models/stage1_classifier.pt", map_location=self.device, weights_only=False)
        self.img_size = ckpt["img_size"]
        self.calib = ckpt["calibration"]
        self.scale = self.calib["calib_scale"]
        self.bias = self.calib["calib_bias"]
        self.op_thresh = self.calib["operating_threshold"]
        self.uncertain_band = self.calib["uncertain_band"]

        self.classifier = DamageClassifier(model_name=ckpt["model_name"], pretrained=False)
        self.classifier.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.classifier.to(self.device)
        self.classifier.eval()

        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.detector = YOLO("models/stage2_detector.pt")
        self.yolo_device = "mps" if self.device.type == "mps" else ("0" if self.device.type == "cuda" else "cpu")

    def check_quality(self, pil_img: Image.Image) -> dict:
        w, h = pil_img.size
        if w < self.q_cfg["min_width"] or h < self.q_cfg["min_height"]:
            return {
                "passed": False,
                "reason": f"Image resolution too small ({w}x{h}px). Minimum required is {self.q_cfg['min_width']}x{self.q_cfg['min_height']}px."
            }

        gray = np.array(pil_img.convert("L"))
        brightness = float(gray.mean())
        if brightness < self.q_cfg["min_brightness"]:
            return {
                "passed": False,
                "reason": f"Image is too dark (brightness={brightness:.1f} < {self.q_cfg['min_brightness']}). Please retake photo in better lighting."
            }
        if brightness > self.q_cfg["max_brightness"]:
            return {
                "passed": False,
                "reason": f"Image is overexposed/glare (brightness={brightness:.1f} > {self.q_cfg['max_brightness']})."
            }

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.q_cfg["min_blur_laplacian"]:
            return {
                "passed": False,
                "reason": f"Image is too blurry (Laplacian sharpness={laplacian_var:.1f} < {self.q_cfg['min_blur_laplacian']}). Please hold camera steady."
            }

        return {"passed": True, "reason": "OK", "brightness": round(brightness, 1), "sharpness": round(laplacian_var, 1)}

    def predict(self, image_input, conf_thresh: float = 0.15):
        t0 = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        w, h = pil_img.size

        t_q0 = time.perf_counter()
        q_res = self.check_quality(pil_img)
        t_q_ms = (time.perf_counter() - t_q0) * 1000.0

        if not q_res["passed"]:
            return {
                "verdict": "rejected_quality",
                "quality_check": q_res,
                "confidence": 0.0,
                "severity": "none",
                "damage_ratio": 0.0,
                "detections": [],
                "needs_review": True,
                "message": q_res["reason"],
                "latency_ms": {"quality_gate": round(t_q_ms, 2), "total": round(t_q_ms, 2)}
            }, pil_img

        t_c0 = time.perf_counter()
        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = float(self.classifier(x).cpu().item())
        prob_damaged = float(1.0 / (1.0 + np.exp(-(logit * self.scale + self.bias))))
        t_c_ms = (time.perf_counter() - t_c0) * 1000.0

        low_b, high_b = self.uncertain_band
        needs_review = bool(low_b <= prob_damaged <= high_b)

        if prob_damaged < self.op_thresh and not needs_review:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "verdict": "undamaged",
                "quality_check": q_res,
                "confidence": round(1.0 - prob_damaged, 4),
                "damage_probability": round(prob_damaged, 4),
                "severity": "none",
                "damage_ratio": 0.0,
                "detections": [],
                "needs_review": False,
                "latency_ms": {
                    "quality_gate": round(t_q_ms, 2),
                    "stage1_classifier": round(t_c_ms, 2),
                    "stage2_detector": 0.0,
                    "stage3_severity": 0.0,
                    "total": round(total_ms, 2)
                }
            }, pil_img

        t_d0 = time.perf_counter()
        yolo_res = self.detector.predict(pil_img, imgsz=640, conf=conf_thresh, device=self.yolo_device, verbose=False)[0]
        t_d_ms = (time.perf_counter() - t_d0) * 1000.0

        detections = []
        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)

        colors = {
            "scratch": (255, 200, 0),
            "dent": (255, 120, 0),
            "crack_gap": (255, 50, 50),
            "severe_replace": (220, 0, 255)
        }

        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            confs = yolo_res.boxes.conf.cpu().numpy()
            cls_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)

            for box, cf, cid in zip(boxes_xyxy, confs, cls_ids):
                cname = self.detector.names[cid]
                b_list = [round(float(c), 1) for c in box]
                detections.append({
                    "type": cname,
                    "box": b_list,
                    "conf": round(float(cf), 4)
                })
                c_rgb = colors.get(cname, (255, 50, 50))
                draw.rectangle(b_list, outline=c_rgb, width=4)
                draw.rectangle([b_list[0], max(0, b_list[1] - 22), b_list[0] + 150, b_list[1]], fill=c_rgb)
                draw.text((b_list[0] + 4, max(0, b_list[1] - 19)), f"{cname} ({cf:.2f})", fill=(0, 0, 0))

        t_s0 = time.perf_counter()
        feat = extract_features_from_detections(detections, w, h)
        if len(detections) == 0 and prob_damaged >= self.op_thresh:
            severity = "minor"
            needs_review = True
        else:
            severity = rule_based_severity(feat)
        t_s_ms = (time.perf_counter() - t_s0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        verdict = "damaged" if prob_damaged >= self.op_thresh else "uncertain"
        return {
            "verdict": verdict,
            "quality_check": q_res,
            "confidence": round(prob_damaged if verdict == "damaged" else (1.0 - prob_damaged), 4),
            "damage_probability": round(prob_damaged, 4),
            "severity": severity,
            "damage_ratio": feat["damage_area_ratio"],
            "detections": detections,
            "needs_review": needs_review,
            "latency_ms": {
                "quality_gate": round(t_q_ms, 2),
                "stage1_classifier": round(t_c_ms, 2),
                "stage2_detector": round(t_d_ms, 2),
                "stage3_severity": round(t_s_ms, 2),
                "total": round(total_ms, 2)
            }
        }, annotated

def run_latency_benchmark():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    test_imgs = [raw_dir / line for line in Path("data/splits/test.txt").read_text().splitlines()[:15]]

    bench = {}
    for dev_name in ["cpu", "mps"]:
        if dev_name == "mps" and not torch.backends.mps.is_available():
            continue
        pipe = InspectionPipeline(device=dev_name)
        pipe.predict(test_imgs[0])

        latencies = []
        for p in test_imgs:
            res, _ = pipe.predict(p)
            latencies.append(res["latency_ms"])

        bench[dev_name] = {
            stage: round(float(np.mean([x.get(stage, 0.0) for x in latencies])), 2)
            for stage in ["quality_gate", "stage1_classifier", "stage2_detector", "stage3_severity", "total"]
        }

    with open("reports/phase9_latency_benchmark.json", "w") as f:
        json.dump(bench, f, indent=2)

    print("\n⚡ Phase 9 Pipeline Latency Benchmark (ms per image):")
    print(json.dumps(bench, indent=2))

if __name__ == "__main__":
    run_latency_benchmark()
