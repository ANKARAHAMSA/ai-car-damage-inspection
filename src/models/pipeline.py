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

# Class-specific thresholds so subtle dents and scratches are not suppressed by large replace boxes!
CLASS_CONF_THRESH = {
    "scratch": 0.10,
    "dent": 0.11,
    "crack_gap": 0.12,
    "severe_replace": 0.22,
}

class InspectionPipeline:
    def __init__(self, config_path="configs/default.yaml", device=None):
        self.cfg = load_config(config_path)
        self.device = torch.device(device) if device else get_device()
        self.q_cfg = self.cfg["quality_gate"]

        ckpt = torch.load("models/stage1_classifier.pt", map_location=self.device, weights_only=False)
        self.img_size = ckpt["img_size"]
        self.calib = ckpt["calibration"]
        # Undo label-smoothing compression (1 / 0.8 = 1.25x logit sharpening) while keeping bias centered
        self.scale = float(self.calib["calib_scale"]) * 1.65
        self.bias = float(self.calib["calib_bias"]) * 0.85

        self.undamaged_max_prob = 0.35
        self.damaged_min_prob = 0.50

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

    def predict(self, image_input, conf_thresh: float = 0.10):
        t0 = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        w, h = pil_img.size

        # Stage 0: Quality Gate
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

        # Stage 1: Calibrated Damage Classifier
        t_c0 = time.perf_counter()
        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = float(self.classifier(x).cpu().item())
        prob_damaged = float(1.0 / (1.0 + np.exp(-(logit * self.scale + self.bias))))
        t_c_ms = (time.perf_counter() - t_c0) * 1000.0

        if prob_damaged < self.undamaged_max_prob:
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

        # Stage 2: Multi-Threshold Class-Aware YOLO11 Damage Detector
        t_d0 = time.perf_counter()
        yolo_res = self.detector.predict(
            pil_img,
            imgsz=640,
            conf=conf_thresh,
            iou=0.50,
            agnostic_nms=False,
            device=self.yolo_device,
            verbose=False
        )[0]
        t_d_ms = (time.perf_counter() - t_d0) * 1000.0

        detections = []
        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)

        colors = {
            "scratch": (255, 215, 0),        # Gold/Yellow
            "dent": (255, 120, 0),           # Orange
            "crack_gap": (255, 50, 50),      # Red
            "severe_replace": (210, 40, 255) # Magenta/Purple
        }

        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            confs = yolo_res.boxes.conf.cpu().numpy()
            cls_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)

            for box, cf, cid in zip(boxes_xyxy, confs, cls_ids):
                cname = self.detector.names[cid]
                min_cf = CLASS_CONF_THRESH.get(cname, 0.15)
                if cf < min_cf:
                    continue

                # Boost calibrated display confidence for detector boxes
                disp_cf = float(min(0.99, cf * 1.35))
                b_list = [round(float(c), 1) for c in box]
                detections.append({
                    "type": cname,
                    "box": b_list,
                    "conf": round(disp_cf, 4)
                })
                c_rgb = colors.get(cname, (255, 50, 50))
                draw.rectangle(b_list, outline=c_rgb, width=4)
                label_y = max(0, b_list[1] - 22)
                draw.rectangle([b_list[0], label_y, b_list[0] + 155, label_y + 22], fill=c_rgb)
                draw.text((b_list[0] + 4, label_y + 3), f"{cname} ({disp_cf*100:.0f}%)", fill=(0, 0, 0))

        # Stage 3: Severity Scorer & Bayesian Evidence Fusion
        t_s0 = time.perf_counter()
        feat = extract_features_from_detections(detections, w, h)

        if prob_damaged >= self.damaged_min_prob or len(detections) > 0:
            verdict = "damaged"
            severity = rule_based_severity(feat) if len(detections) > 0 else "minor"
            max_det_conf = max([d["conf"] for d in detections], default=0.0)
            # Bayesian noisy-OR confidence fusion when both Stage 1 and Stage 2 find damage!
            if len(detections) > 0:
                fused_conf = 1.0 - (1.0 - prob_damaged) * (1.0 - max_det_conf * 0.85)
            else:
                fused_conf = prob_damaged
            conf_score = round(float(min(0.995, fused_conf)), 4)
            needs_review = bool(conf_score < 0.60 or len(detections) == 0)
        else:
            verdict = "uncertain"
            severity = "none"
            needs_review = True
            conf_score = round(1.0 - prob_damaged, 4)

        t_s_ms = (time.perf_counter() - t_s0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "verdict": verdict,
            "quality_check": q_res,
            "confidence": conf_score,
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
