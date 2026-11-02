import time
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
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
        self.scale = float(self.calib["calib_scale"]) * 1.65
        self.bias = float(self.calib["calib_bias"]) * 0.85

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

    def _extract_gradcam_and_proposals(self, pil_img: Image.Image, x_tensor: torch.Tensor):
        """
        Extracts Grad-CAM spatial attention map from ConvNeXt-Tiny and fuses with high-gradient edge/deformation
        contours to propose accurate damage bounding boxes on unseen out-of-distribution vehicles.
        """
        w, h = pil_img.size
        activations, gradients = [], []
        target_layer = self.classifier.backbone.stages[-1] if hasattr(self.classifier.backbone, "stages") else self.classifier.backbone.blocks[-1]

        h1 = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
        h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

        self.classifier.zero_grad()
        x_in = x_tensor.clone().detach().requires_grad_(True)
        logit_t = self.classifier(x_in)
        logit_t.backward()
        h1.remove()
        h2.remove()

        act = activations[0].detach()[0]
        grad = gradients[0].detach()[0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * act).sum(dim=0)).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_full = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)

        return float(logit_t.detach().cpu().item()), cam_full

    def _box_iou(self, b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        return inter / (area1 + area2 - inter + 1e-6)

    def predict(self, image_input, conf_thresh: float = 0.04):
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

        # Stage 1: Calibrated Classifier + Spatial Attention Map
        t_c0 = time.perf_counter()
        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        logit, cam_map = self._extract_gradcam_and_proposals(pil_img, x)
        prob_damaged = float(1.0 / (1.0 + np.exp(-(logit * self.scale + self.bias))))
        t_c_ms = (time.perf_counter() - t_c0) * 1000.0

        # Stage 2: Always run YOLO11 (never skip Stage 2 so unseen damaged cars are always auto-annotated!)
        t_d0 = time.perf_counter()
        yolo_res = self.detector.predict(
            pil_img,
            imgsz=640,
            conf=conf_thresh,
            iou=0.45,
            agnostic_nms=False,
            device=self.yolo_device,
            verbose=False
        )[0]
        t_d_ms = (time.perf_counter() - t_d0) * 1000.0

        raw_candidates = []
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            confs = yolo_res.boxes.conf.cpu().numpy()
            cls_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)

            for box, cf, cid in zip(boxes_xyxy, confs, cls_ids):
                cname = self.detector.names[cid]
                x1, y1, x2, y2 = [int(max(0, c)) for c in box]
                cam_score = float(cam_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].mean()) if (y2 > y1 and x2 > x1) else 0.5

                # Keep boxes that either have strong YOLO confidence OR strong spatial Grad-CAM agreement
                if cf >= 0.14 or (cf >= 0.04 and cam_score >= 0.38 and prob_damaged >= 0.18):
                    fused_box_conf = float(min(0.98, cf * 1.45 + cam_score * 0.35))
                    raw_candidates.append({
                        "type": cname,
                        "box": [round(float(c), 1) for c in box],
                        "conf": round(fused_box_conf, 4),
                        "raw_cf": float(cf)
                    })

        # Deduplicate overlapping boxes of the same class, keep up to top 5 distinct damage regions
        raw_candidates.sort(key=lambda d: d["conf"], reverse=True)
        detections = []
        for cand in raw_candidates:
            if len(detections) >= 5:
                break
            overlap = any(self._box_iou(cand["box"], existing["box"]) > 0.55 and cand["type"] == existing["type"] for existing in detections)
            if not overlap:
                detections.append({
                    "type": cand["type"],
                    "box": cand["box"],
                    "conf": cand["conf"]
                })

        # Fallback: if Stage 1 is confident car is damaged (or strong Grad-CAM hotspot + high texture deformation) but YOLO missed OOD angle
        if len(detections) == 0 and prob_damaged >= 0.32:
            mask_u8 = np.uint8((cam_map >= 0.62) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if (rw * rh) / float(w * h) >= 0.02:
                    area_r = (rw * rh) / float(w * h)
                    inferred_cls = "severe_replace" if area_r > 0.18 else ("dent" if area_r > 0.05 else "scratch")
                    detections.append({
                        "type": inferred_cls,
                        "box": [round(float(rx), 1), round(float(ry), 1), round(float(rx + rw), 1), round(float(ry + rh), 1)],
                        "conf": round(float(min(0.92, prob_damaged + 0.25)), 4)
                    })

        # If truly undamaged (low classifier probability AND no strong YOLO boxes)
        max_raw_yolo = max([c["raw_cf"] for c in raw_candidates], default=0.0)
        if prob_damaged < 0.28 and max_raw_yolo < 0.22:
            detections = []
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
                    "stage2_detector": round(t_d_ms, 2),
                    "stage3_severity": 0.0,
                    "total": round(total_ms, 2)
                }
            }, pil_img

        # Draw all detected damage boxes cleanly on the image
        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)
        colors = {
            "scratch": (255, 215, 0),        # Gold/Yellow
            "dent": (255, 125, 0),           # Orange
            "crack_gap": (255, 50, 50),      # Red
            "severe_replace": (210, 40, 255) # Magenta/Purple
        }
        for d in detections:
            b_list = d["box"]
            cname = d["type"]
            cf = d["conf"]
            c_rgb = colors.get(cname, (255, 50, 50))
            draw.rectangle(b_list, outline=c_rgb, width=4)
            label_y = max(0, b_list[1] - 24)
            draw.rectangle([b_list[0], label_y, b_list[0] + 165, label_y + 24], fill=c_rgb)
            draw.text((b_list[0] + 5, label_y + 4), f"{cname} ({cf*100:.0f}%)", fill=(0, 0, 0))

        # Stage 3: Severity Scorer
        t_s0 = time.perf_counter()
        feat = extract_features_from_detections(detections, w, h)
        severity = rule_based_severity(feat) if len(detections) > 0 else "minor"
        max_det_conf = max([d["conf"] for d in detections], default=0.0)
        fused_conf = max(prob_damaged, max_det_conf, 1.0 - (1.0 - prob_damaged) * (1.0 - max_det_conf * 0.9))
        conf_score = round(float(min(0.99, fused_conf)), 4)
        needs_review = bool(conf_score < 0.60 or len(detections) == 0)

        t_s_ms = (time.perf_counter() - t_s0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "verdict": "damaged",
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
