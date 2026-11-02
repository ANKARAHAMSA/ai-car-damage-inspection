import os
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(report_id: str, orig_img: Image.Image, ann_img: Image.Image, result: dict, out_dir: str = "reports/generated") -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = str(Path(out_dir) / f"{report_id}.pdf")
    orig_tmp = str(Path(out_dir) / f"{report_id}_orig.jpg")
    ann_tmp = str(Path(out_dir) / f"{report_id}_ann.jpg")

    orig_img.convert("RGB").save(orig_tmp, quality=85)
    ann_img.convert("RGB").save(ann_tmp, quality=85)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#1e293b"), spaceAfter=6)
    sub_style = ParagraphStyle("SubStyle", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#64748b"), spaceAfter=14)

    story.append(Paragraph("AI Car Damage Inspection Report", title_style))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"<b>Report ID:</b> {report_id} &nbsp;|&nbsp; <b>Timestamp:</b> {ts}", sub_style))

    verdict = result.get("verdict", "unknown").upper()
    sev = result.get("severity", "none").upper()
    conf = f"{result.get('confidence', 0.0) * 100:.1f}%"
    area_pct = f"{result.get('damage_ratio', 0.0) * 100:.2f}%"
    review_str = "YES (Human Review)" if result.get("needs_review") else "NO (Auto Pass)"

    summary_data = [
        ["Inspection Verdict", verdict, "Confidence Score", conf],
        ["Estimated Severity", sev, "Damaged Area Ratio", area_pct],
        ["Needs Human Review", review_str, "Pipeline Latency", f"{result.get('latency_ms', {}).get('total', 0.0):.1f} ms"],
    ]
    t_sum = Table(summary_data, colWidths=[125, 135, 125, 135])
    t_sum.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Visual Inspection (Original vs. AI Annotated Detections)</b>", styles["Heading3"]))
    story.append(Spacer(1, 6))
    img_table = Table([
        [RLImage(orig_tmp, width=245, height=185), RLImage(ann_tmp, width=245, height=185)],
        ["Original Vehicle Photo", "Annotated Damage Localization (YOLO11)"]
    ], colWidths=[260, 260])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Oblique"),
        ("FONTSIZE", (0, 1), (-1, 1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Localized Damage Regions</b>", styles["Heading3"]))
    dets = result.get("detections", [])
    if dets:
        det_rows = [["#", "Damage Class", "Confidence", "Bounding Box [x1, y1, x2, y2]"]]
        for i, d in enumerate(dets, 1):
            det_rows.append([str(i), d["type"], f"{d['conf']*100:.1f}%", str(d["box"])])
        t_det = Table(det_rows, colWidths=[40, 140, 100, 240])
        t_det.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_det)
    else:
        story.append(Paragraph("No localized bounding-box detections above threshold.", styles["Normal"]))

    story.append(Spacer(1, 24))
    disc_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#64748b"))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report was generated automatically by the AI Car Damage Inspection System "
        "(ConvNeXt-Tiny Classifier + YOLO11 Detector + Calibrated Severity Heuristic). Severity scores are calibrated "
        "estimates for insurance claims triage and rental return checks, not a certified structural repair quote.",
        disc_style
    ))

    doc.build(story)
    for tmp in (orig_tmp, ann_tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
    return pdf_path
