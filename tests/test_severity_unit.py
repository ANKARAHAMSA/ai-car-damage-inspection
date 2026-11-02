from src.models.severity_scorer import extract_features_from_detections, rule_based_severity

def test_severity_minor_scratch():
    dets = [{"type": "scratch", "box": [10, 10, 40, 40], "conf": 0.9}]
    feat = extract_features_from_detections(dets, img_w=1000, img_h=1000)
    assert rule_based_severity(feat) == "minor"

def test_severity_moderate_dent_or_crack():
    dets = [{"type": "crack_gap", "box": [100, 100, 200, 200], "conf": 0.85}]
    feat = extract_features_from_detections(dets, img_w=1000, img_h=1000)
    assert rule_based_severity(feat) == "moderate"

def test_severity_severe_large_area_or_replace():
    dets = [{"type": "dent", "box": [0, 0, 600, 600], "conf": 0.95}]
    feat = extract_features_from_detections(dets, img_w=1000, img_h=1000)
    assert rule_based_severity(feat) == "severe"
