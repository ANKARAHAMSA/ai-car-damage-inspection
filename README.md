# AI Car Damage Inspection System

End-to-end computer vision pipeline for automated vehicle damage assessment:
1. **Quality Gate:** Rejects blurry, dark, or tiny images.
2. **Stage 1 (Classifier):** Damaged vs. Undamaged + Confidence Calibration & Human-Review Routing.
3. **Stage 2 (Detector):** YOLO11 damage & car-part localization.
4. **Stage 3 (Severity Scorer):** Rule-based + LightGBM severity estimation (Minor / Moderate / Severe).
5. **API & Reporting:** FastAPI backend + automated PDF inspection report generator.
