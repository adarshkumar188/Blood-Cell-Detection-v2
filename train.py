"""
Blood Cell Detection - Training Script v2.1
Dataset manually download karo, phir run karo.
"""

import os, sys

# ── Auto-install ultralytics if missing ─────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system(f"{sys.executable} -m pip install ultralytics -q")
    from ultralytics import YOLO

# ── Check dataset ────────────────────────────────────────
TRAIN_IMGS = os.path.join("data", "train", "images")

def count_images(path):
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])

n = count_images(TRAIN_IMGS)

if n == 0:
    print("\n" + "="*58)
    print("❌  Dataset not found in data/train/images/")
    print("="*58)
    print()
    print("  Download karo (free, 2 min):")
    print("  👉 https://universe.roboflow.com/roboflow-100/bccd-yfzgt")
    print()
    print("  Steps:")
    print("  1. Site pe jaao → Download Dataset")
    print("  2. Format: YOLOv8 → Download zip to computer")
    print("  3. ZIP extract karo")
    print("  4. train/, valid/, test/ folders ko")
    print("     apne  data/  folder mein paste karo")
    print("  5. python train.py  dobara run karo")
    print()
    print("="*58 + "\n")
    sys.exit(1)

print(f"✅ Dataset ready — {n} training images found\n")

# ── Write correct data.yaml ──────────────────────────────
with open("data.yaml", "w") as f:
    f.write("""path: data
train: train/images
val:   valid/images
test:  test/images

nc: 3
names:
  0: Platelets
  1: RBC
  2: WBC
""")

# ── Train ────────────────────────────────────────────────
print("🏋  Training YOLOv8n...")
print("    CPU pe ~15-30 min lagenge. Chai pi lo ☕\n")

model = YOLO("yolov8n.pt")

model.train(
    data     = "data.yaml",
    epochs   = 50,
    imgsz    = 640,
    batch    = 8,        # RAM kam ho toh 4 karo
    device   = "cpu",    # GPU ho toh "0" karo
    patience = 10,
    project  = "runs/detect",
    name     = "train",
    exist_ok = True,
)

# ── Done ─────────────────────────────────────────────────
W = os.path.join("runs", "detect", "train", "weights", "best.pt")
if os.path.exists(W):
    print("\n" + "="*50)
    print("✅  Training complete!")
    print(f"   Weights: {W}")
    print("\n▶️   Run:  python app.py")
    print("    Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
else:
    print(f"\n⚠️  Training done but best.pt not found at: {W}")
