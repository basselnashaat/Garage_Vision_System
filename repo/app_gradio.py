import logging
import difflib
from datetime import datetime
from typing import Optional
import cv2 as cv
import numpy as np
import gradio as gr

from app.pipeline.coordinator import LPRPipeline
from app.database.connection import Database
from app.database.logger import VisitLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_pipeline: Optional[LPRPipeline] = None
_visit_logger: Optional[VisitLogger] = None

def get_pipeline() -> LPRPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing real LPRPipeline and loading model weights...")
        _pipeline = LPRPipeline(use_hugging_face=True)
    return _pipeline

def get_logger() -> Optional[VisitLogger]:
    """Initializes the database connection for Gradio exactly like FastAPI does."""
    global _visit_logger
    if _visit_logger is None:
        try:
            logger.info("Connecting to Supabase for Gradio logging...")
            db = Database()
            _visit_logger = VisitLogger(db)
        except Exception as e:
            logger.warning(f"Database logging disabled in Gradio (check config): {e}")
    return _visit_logger

def _log_detections_to_db(detections: list[dict]):
    """Helper to silently log to the database using a valid camera ID."""
    v_logger = get_logger()
    if not v_logger:
        return

    # 1. Fetch a real, valid camera ID from Supabase
    valid_camera_id = None
    try:
        existing_cameras = v_logger.db.query("cameras", select="id")
        if existing_cameras and len(existing_cameras) > 0:
            # Use the first available camera
            valid_camera_id = existing_cameras[0]["id"]
        else:
            # AUTOMATIC FIX: Create a default camera if the table is empty!
            logger.info("No cameras found in DB. Creating a default 'Gradio' camera...")
            new_cam = v_logger.db.insert("cameras", {
                "location_name": "Gradio Manual Uploads",
                "location_type": "dashboard"
            })
            valid_camera_id = new_cam[0]["id"]
            
    except Exception as e:
        logger.error(f"❌ Failed to fetch or create camera in DB: {e}")
        return

    # 2. Log the detections using the guaranteed real camera ID
    now = datetime.now()
    for det in detections:
        try:
            visit_id = v_logger.log(
                plate_digits=det.get("digits", "0000"),
                plate_letters=det.get("letters", "UNKNOWN"),
                plate_score=det.get("plate_score", 0.0),
                segment=det.get("segment", "unknown"),
                segment_score=det.get("segment_score", 0.0),
                purchasing_power=det.get("purchasing_power", 0.0),
                car_brand=det.get("car_brand", "unknown"),
                car_model=det.get("car_model", "unknown"),
                timestamp=now,
                camera_id=valid_camera_id, 
            )
            det["visit_id"] = visit_id
            logger.info(f"✅ Successfully logged Gradio detection to DB (Visit: {visit_id})")
        except Exception as e:
            logger.error(f"❌ Failed to log to Supabase from Gradio: {e}")
# ── Hotfix Overrides ─────────────────────────────────────────────────────────
VEHICLE_DB_HOTFIX = {
    "G63": {"segment": "Luxury", "price": 18000000.0, "power": 0.98},
}

def _patch_detection(det: dict) -> dict:
    if not det: return det
    raw_segment = str(det.get("segment", ""))
    if "_" in raw_segment:  
        det["segment"] = "Luxury" if "G63" in raw_segment else "Premium" 
    
    model = str(det.get("car_model", ""))
    if model in VEHICLE_DB_HOTFIX:
        fix = VEHICLE_DB_HOTFIX[model]
        det["segment"] = fix["segment"]
        if not det.get("estimated_price") or det.get("estimated_price") <= 0:
            det["estimated_price"] = fix["price"]
        if det.get("purchasing_power", 0.0) < fix["power"]:
            det["purchasing_power"] = fix["power"]
            
    return det

# ── Schema Formatting Helpers ────────────────────────────────────────────────

def _fmt_price(price: Optional[float]) -> str:
    if price is None or price <= 0:
        return "Unknown (Not in DB)"
    return f"EGP {price:,.0f}"

def _fmt_power(power: float) -> str:
    try:
        p = float(power)
        if p >= 0.90: return f"🟣 Elite  ({p:.2f})"
        elif p >= 0.70: return f"🔴 High  ({p:.2f})"
        elif p >= 0.40: return f"🟡 Mid  ({p:.2f})"
        else: return f"🟢 Standard  ({p:.2f})"
    except (ValueError, TypeError):
        return "🟢 Standard  (0.00)"

def _fmt_special(level: str) -> str:
    lvl = str(level).title()
    if lvl.lower() in ["normal", "none", "false", ""]:
        return "Normal"
    return f"✨ {lvl}"

def _best_detection(detections: list) -> Optional[dict]:
    if not detections: return None
    return max(detections, key=lambda d: d.get("purchasing_power", 0.0))

def _draw_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    out = frame.copy()
    for idx, det in enumerate(detections):
        bbox = det.get("bbox")
        if bbox is None: continue
        x1, y1, x2, y2 = map(int, bbox)
        cv.rectangle(out, (x1, y1), (x2, y2), (225, 112, 85), 3)
        cv.putText(out, f"Plate #{idx + 1}", (x1, y1 - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (225, 112, 85), 2)
    return out

# ── Core Processing ──────────────────────────────────────────────────────────

def process_image(image: np.ndarray):
    empty_returns = (None, "", "", "", "", "", "", "", "", "", "", "⚠️ No image provided.")
    if image is None: return empty_returns
    
    try:
        pipeline = get_pipeline()
        frame_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        result = pipeline.process(frame_bgr)

        if not result.get("success"):
            return (image, "", "", "", "", "", "", "", "", "", "", f"❌ Error: {result.get('error')}")

        detections = result.get("detections", [])
        if not detections: 
            return (image, "", "", "", "", "", "", "", "", "", "", "ℹ️ No license plates detected.")

        # Patch detections before logging
        patched_detections = [_patch_detection(d) for d in detections]
        
        # LOG TO DATABASE HERE
        _log_detections_to_db(patched_detections)

        annotated_rgb = cv.cvtColor(_draw_boxes(frame_bgr, patched_detections), cv.COLOR_BGR2RGB)
        det = _best_detection(patched_detections)

        return (
            annotated_rgb,
            str(det.get("plate_string", "—")),
            str(det.get("digits", "—")),
            str(det.get("letters", "—")),
            _fmt_special(det.get("special_plate_level", "Normal")),
            f"{det.get('special_plate_score', 0.0):.1f} / 20.0",
            str(det.get("car_brand", "—")).title(),
            str(det.get("car_model", "—")).replace("_", " ").title(),
            str(det.get("segment", "Unknown")).title(),
            _fmt_price(det.get("estimated_price")),
            _fmt_power(det.get("purchasing_power", 0.0)),
            f"✅ Processed {len(patched_detections)} plate(s) and logged to DB.",
        )
    except Exception as e:
        logger.exception("Image processing failed")
        return (image, "", "", "", "", "", "", "", "", "", "", f"❌ Failed: {e}")

def process_video(video_path: str):
    if not video_path: return "", [], "⚠️ No video provided."
    try:
        pipeline = get_pipeline()
        result = pipeline.process_video(video_path)

        if not result.get("success"):
            return "", [], f"❌ Error: {result.get('error')}"

        detections = result.get("detections", [])
        
        # Patch detections before logging
        patched_detections = [_patch_detection(d) for d in detections]
        
        # LOG TO DATABASE HERE
        _log_detections_to_db(patched_detections)

        unique_rows = []
        for det in patched_detections:
            plate_str = str(det.get("plate_string", "—")).strip()
            digits = str(det.get("digits", "—"))
            letters = str(det.get("letters", "—"))
            sp_level = _fmt_special(det.get("special_plate_level", "Normal"))
            brand = str(det.get("car_brand", "—")).title()
            model = str(det.get("car_model", "—")).replace("_", " ").title()
            segment = str(det.get("segment", "Unknown")).title()
            price = _fmt_price(det.get("estimated_price"))
            power = _fmt_power(det.get("purchasing_power", 0.0))
            
            is_dup = False
            for r in unique_rows:
                if plate_str != "—" and difflib.SequenceMatcher(None, plate_str, r[0]).ratio() >= 0.75:
                    is_dup = True
                    break
            
            if not is_dup:
                unique_rows.append([plate_str, digits, letters, sp_level, brand, model, segment, price, power])

        summary = f"**Total Frames Tracked:** {result.get('num_vehicles_processed', 0)}  |  **Unique Output Logs:** {len(unique_rows)}"
        return summary, unique_rows, f"✅ Processing complete and logged to DB."
    except Exception as e:
        logger.exception("Video processing failed")
        return "", [], f"❌ Failed: {e}"

# ── Dashboard Layout ─────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Vehicle Intelligence Dashboard") as demo:
        gr.Markdown("# 🚗 Vehicle Intelligence Dashboard")
        with gr.Tabs():
            
            with gr.Tab("📷 Image Processing"):
                with gr.Row():
                    img_input = gr.Image(label="Input View", type="numpy", height=340)
                    img_output = gr.Image(label="Tracking Overlays", type="numpy", height=340, interactive=False)
                
                img_btn = gr.Button("Run Real-Time Pipeline", variant="primary")
                img_status = gr.Markdown("")
                
                gr.Markdown("### 📋 Deep Analysis (Schema Data)")
                with gr.Row():
                    
                    with gr.Group():
                        gr.Markdown("#### 🪪 Plate OCR & Analysis")
                        out_plate = gr.Textbox(label="Full Plate String", interactive=False)
                        with gr.Row():
                            out_digits = gr.Textbox(label="🔢 Digits", interactive=False)
                            out_letters = gr.Textbox(label="🔠 Letters", interactive=False)
                        with gr.Row():
                            out_sp_level = gr.Textbox(label="✨ Special Level", interactive=False)
                            out_sp_score = gr.Textbox(label="⭐ Special Score", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("#### 🚘 Vehicle Classification")
                        with gr.Row():
                            out_brand = gr.Textbox(label="🏷️ Brand", interactive=False)
                            out_model = gr.Textbox(label="🚗 Model", interactive=False)
                        out_segment = gr.Textbox(label="🚙 Segment", interactive=False)
                        with gr.Row():
                            out_price = gr.Textbox(label="💰 Est. Valuation", interactive=False)
                            out_power = gr.Textbox(label="📊 Purchasing Power", interactive=False)

                img_btn.click(
                    fn=process_image, 
                    inputs=[img_input], 
                    outputs=[
                        img_output, out_plate, out_digits, out_letters, out_sp_level, out_sp_score, 
                        out_brand, out_model, out_segment, out_price, out_power, img_status
                    ]
                )

            with gr.Tab("🎬 Video Processing"):
                vid_input = gr.Video(label="Upload surveillance stream", height=300)
                vid_btn = gr.Button("Run Sequential Batch Processing", variant="primary")
                vid_summary = gr.Markdown("")
                
                vid_table = gr.Dataframe(
                    headers=["Plate", "Digits", "Letters", "Special Level", "Brand", "Model", "Segment", "Est. Value", "Power"], 
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str"], 
                    label="Surveillance Logs", 
                    interactive=False, 
                    wrap=True
                )
                vid_status = gr.Markdown("")
                vid_btn.click(fn=process_video, inputs=[vid_input], outputs=[vid_summary, vid_table, vid_status])
                
    return demo

if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7862, theme=gr.themes.Soft(), share=False)