"""
Blood Cell Detection & Counting - v2.2
Pre-trained model auto-download (no training needed!)
Run: python app.py
"""
 
from flask import Flask, render_template, request, jsonify, send_file
import cv2, numpy as np, os, io, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
 
try:
    from PIL import Image as PILImage
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
 
app = Flask(__name__)
 
# ─────────────────────────────────────────────
#  Auto-download pre-trained model
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join("weights", "best.pt")
os.makedirs("weights", exist_ok=True)
 
def download_model():
    """Download pre-trained YOLOv8 blood cell model from HuggingFace"""
    import urllib.request
    url = "https://huggingface.co/keremberke/yolov8n-blood-cell-detection/resolve/main/best.pt"
    print("📥 Downloading pre-trained blood cell model (~6 MB)...")
    print("   (One-time download, please wait...)")
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("✅ Model downloaded!\n")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False
 
def load_model():
    global model
    try:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            success = download_model()
            if not success:
                print("⚠️  Could not download model. Check internet connection.")
                return
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded and ready!\n")
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        print(f"❌ Model error: {e}")
 
model = None
load_model()
 
# ─────────────────────────────────────────────
#  Class names & colors
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "Platelets", 1: "RBC", 2: "WBC"}
COLORS = {
    "Platelets": (255, 215, 0),
    "RBC":       (220, 53,  69),
    "WBC":       (30,  144, 255),
}
 
OUTPUT_DIR      = os.path.join("static", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROCESSED_IMAGE = os.path.join(OUTPUT_DIR, "processed.jpg")
CHART_IMAGE     = os.path.join(OUTPUT_DIR, "chart.png")
PDF_REPORT      = os.path.join(OUTPUT_DIR, "blood_report.pdf")
 
latest_counts: dict = {}
latest_boxes:  list = []
 
# ─────────────────────────────────────────────
#  Image decoder (JPG, PNG, WebP, BMP ...)
# ─────────────────────────────────────────────
def decode_image(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    if PILLOW_AVAILABLE:
        pil = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    raise ValueError("Cannot decode image. Install Pillow: pip install Pillow")
 
# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")
 
 
@app.route("/predict", methods=["POST"])
def predict():
    global latest_counts, latest_boxes
 
    if model is None:
        return jsonify({"error": "Model not ready. Check terminal for details."}), 503
 
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image provided"}), 400
 
    try:
        img = decode_image(request.files["image"].read())
        results    = model(img)[0]
        counts     = {name: 0 for name in CLASS_NAMES.values()}
        boxes_data = []
 
        for box in results.boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = CLASS_NAMES.get(cls_id, "Unknown")
            counts[class_name] = counts.get(class_name, 0) + 1
 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(class_name, (200, 200, 200))
 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(img, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
 
            boxes_data.append({
                "x": x1, "y": y1, "w": x2-x1, "h": y2-y1,
                "label": class_name,
                "confidence": round(conf*100, 1),
                "color": f"rgb{color}"
            })
 
        cv2.imwrite(PROCESSED_IMAGE, img)
        latest_counts = counts
        latest_boxes  = boxes_data
 
        return jsonify({
            "counts": counts,
            "boxes":  boxes_data,
            "image":  PROCESSED_IMAGE,
            "total":  sum(counts.values())
        })
 
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
 
 
# ─────────────────────────────────────────────
#  Chart & PDF
# ─────────────────────────────────────────────
def generate_chart(counts):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels  = list(counts.keys())
    values  = list(counts.values())
    bars    = ax.bar(labels, values, color=["#f4b400","#e63946","#1e9aff"],
                     width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(val), ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Blood Cell Distribution", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values)*1.3+1 if values else 5)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART_IMAGE, dpi=120)
    plt.close(fig)
 
 
def generate_pdf():
    doc    = SimpleDocTemplate(PDF_REPORT, pagesize=A4,
                               leftMargin=40, rightMargin=40,
                               topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
 
    # Custom styles
    centered = ParagraphStyle(
        "centered", parent=styles["Normal"],
        alignment=TA_CENTER, fontSize=10
    )
    contact_style = ParagraphStyle(
        "contact", parent=styles["Normal"],
        alignment=TA_CENTER, fontSize=9,
        textColor=colors.HexColor("#555555")
    )
    link_style = ParagraphStyle(
        "link", parent=styles["Normal"],
        alignment=TA_CENTER, fontSize=9,
        textColor=colors.HexColor("#1e9aff")
    )
 
    story = []
 
    # ── Title (no emoji — ReportLab built-in fonts don't support them)
    story.append(Paragraph("AI Hematology Analysis Report", styles["Title"]))
    story.append(Spacer(1, 6))
 
    # ── Developer info block
    story.append(Paragraph("<b>Developed by: Adarsh Kumar</b>", centered))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Email: galaxyadarsh53@gmail.com", contact_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        'LinkedIn: <a href="https://www.linkedin.com/in/adarsh-kumar-685a11281/">'
        'linkedin.com/in/adarsh-kumar-685a11281</a>',
        link_style
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#1e9aff")))
    story.append(Spacer(1, 10))
 
    # ── Generated date
    story.append(Paragraph(
        f"Generated: <b>{datetime.now().strftime('%d %B %Y | %H:%M')}</b>",
        styles["Normal"]))
    story.append(Spacer(1, 14))
 
    # ── Processed image (smaller so chart fits on same page)
    if os.path.exists(PROCESSED_IMAGE):
        story.append(Paragraph("Processed Blood Smear", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(RLImage(PROCESSED_IMAGE, width=400, height=250))
        story.append(Spacer(1, 10))
 
    # ── Detection summary table
    story.append(Paragraph("Detection Summary", styles["Heading2"]))
    story.append(Spacer(1, 6))
    data = [["Cell Type", "Count"]] + [[k, str(v)] for k, v in latest_counts.items()]
    data.append(["Total", str(sum(latest_counts.values()))])
    tbl = Table(data, colWidths=[200, 100])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  colors.HexColor("#1e9aff")),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 11),
        ("ALIGN",         (0, 0),  (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -2), [colors.whitesmoke, colors.white]),
        ("BACKGROUND",    (0, -1), (-1, -1), colors.HexColor("#f0f0f0")),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID",          (0, 0),  (-1, -1), 0.5, colors.grey),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))
 
    # ── Chart (smaller so it fits on page 1)
    if os.path.exists(CHART_IMAGE):
        story.append(Paragraph("Cell Distribution Chart", styles["Heading2"]))
        story.append(Spacer(1, 4))
        story.append(RLImage(CHART_IMAGE, width=280, height=190))
 
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<i>AI-generated report for research/educational use only.</i>",
        contact_style))
 
    doc.build(story)
 
 
@app.route("/download-report")
def download_report():
    if not latest_counts:
        return "Pehle image analyze karo.", 400
    generate_chart(latest_counts)
    generate_pdf()
    return send_file(PDF_REPORT, as_attachment=True,
                     download_name="blood_cell_report.pdf")
 
 
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🩸 Blood Cell Detection Server")
    print("   Browser mein kholo: http://127.0.0.1:5000\n")
    app.run(debug=True, use_reloader=False)