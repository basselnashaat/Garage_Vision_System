# 🚗 Garage Vision System — AI-Powered Visitor Profiling

A production-grade computer vision system that processes real-time garage camera feeds to estimate visitor purchasing power through automated license plate recognition and vehicle classification.

---

## 🧠 How It Works

The system runs a **5-stage deep learning pipeline** on every vehicle entering the garage:

```
Video Stream → Plate Detection → Arabic OCR → Vehicle Classification → Scoring Engine
```

1. **Plate Detection** — YOLOv11n detects license plates in real-time camera frames
2. **Enhancement** — EDSR 4x Super-Resolution upscales low-resolution crops; HSV histogram equalization improves contrast for night/garage lighting
3. **Arabic OCR** — Fine-tuned YOLOv11n reads Arabic-Indic numerals and characters from the plate
4. **Vehicle Classification** — EfficientNetV2-S classifies the vehicle's make, model, and generation across 1,211 classes
5. **Scoring Engine** — Combines vehicle generation and market price data to estimate visitor purchasing power, logged to a PostgreSQL database

---

## 📊 Key Metrics

| Model | Metric | Score |
|---|---|---|
| Plate Detection (YOLOv11n) | mAP50 | **0.995** |
| Plate Detection (YOLOv11n) | mAP50-95 | **0.898** |
| Arabic OCR (YOLOv11n) | mAP50 | **0.989** |
| Arabic OCR (YOLOv11n) | mAP50-95 | **0.737** |
| Vehicle Classification (EfficientNetV2-S) | Top-5 Accuracy | **92.02%** |
| Dataset Size | Images | **257,957** across **1,211** classes |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| ML / Computer Vision | PyTorch, YOLOv11, EfficientNetV2-S, EDSR Super-Resolution, OpenCV |
| Data Collection | Selenium (Cloudflare bypass), CVAT (labeling) |
| Backend | FastAPI (async), Pydantic, Psycopg2 connection pooling |
| Database | Supabase / PostgreSQL — full ACID transactions |
| Frontend Dashboard | React, TypeScript, Vite, TanStack Query, Recharts, Tailwind CSS, Framer Motion |

---

## 🗂️ Project Structure

```
├── app/
│   ├── pipeline/
│   │   ├── detector.py        # YOLOv11 plate detection
│   │   ├── enhancer.py        # EDSR super-resolution + HSV equalization
│   │   ├── ocr.py             # Arabic character recognition
│   │   ├── classifier.py      # EfficientNetV2-S vehicle classification
│   │   ├── scorer.py          # Purchasing power scoring engine
│   │   └── coordinator.py     # Pipeline orchestration
│   ├── database/
│   │   ├── connection.py      # PostgreSQL connection pooling
│   │   └── logger.py          # Visitor profile logging
│   └── main.py                # FastAPI entry point
├── src/
│   └── capture/               # Camera feed capture module
├── vehicle-intelligence-dashboard/  # React frontend
├── requirements.txt
└── thisishowtorun.md          # Setup & run instructions
```

---

## ⚙️ Setup & Run

See [`thisishowtorun.md`](./thisishowtorun.md) for full setup instructions.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.backend.txt

# Set up environment variables
cp .env.example .env
# Fill in your Supabase credentials and model paths

# Run the backend
python app/main.py

# Run the dashboard (in a separate terminal)
cd vehicle-intelligence-dashboard
npm install
npm run dev
```

> **Note:** Model weights (`.pt` files) are not included in this repo due to size. Download links are provided in `thisishowtorun.md`.

---

## 🔑 Key Engineering Challenges Solved

- **Cloudflare bypass** — Built a Selenium-based scraper to collect 257,957 images from a protected automotive portal
- **Class imbalance** — Handled 1,211 long-tail classes using class weights in CrossEntropyLoss and label smoothing (0.1)
- **Arabic numeral bug** — Fixed Python's native `.isdigit()` returning `False` for Arabic-Indic numerals (٠–٩) with a custom containment check
- **Low-res night feeds** — EDSR 4x upscaling + HSV V-channel histogram equalization before OCR stage
- **2025 Egyptian plate update** — Handled the new "Kaf" (ك) series using targeted CVAT labeling and balanced augmentation without catastrophic forgetting

---

## 📈 Live Dashboard

The React frontend provides:
- Real-time KPI indicators per vehicle entry
- Visitor ledger with full detection history
- 30-day interactive trend visualizers (Recharts)

---

## 👤 Author

**Bassel Nashaat**
- LinkedIn: [linkedin.com/in/bassel-nashaat](https://linkedin.com/in/bassel-nashaat)
- GitHub: [github.com/basselnashaat](https://github.com/basselnashaat)
