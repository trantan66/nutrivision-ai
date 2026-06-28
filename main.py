"""
Model Comparison API
====================
FastAPI server to load and compare predictions from 6 Keras models:

Frontend: http://localhost:8000/app  (served from ../Fe/)
  - best_model_vgg_finetune.keras
  - best_model_vgg_transfer.keras
  - best_model_resnet_finetune.keras
  - best_model_resnet_transfer.keras
  - best_model_1.keras
  - best_model_2.keras

All models: input=(224,224,3), output=30 food classes (softmax)

Usage:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  GET  /                    - API info
  GET  /health              - Health check + model status
  GET  /classes             - List all 30 class labels
  POST /predict/upload      - Predict from uploaded image file
  POST /predict/url         - Predict from image URL
  GET  /compare             - Compare all 6 models (alias for predict endpoints)
"""

import os
import io
import time
import logging
from pathlib import Path
from typing import Optional, List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "model"

# 30 food classes (Vietnamese food dataset — adjust if your classes differ)
CLASS_NAMES = [
    "Banh beo",
    "Banh bot loc",
    "Banh can",
    "Banh canh",
    "Banh chung",
    "Banh cuon",
    "Banh duc",
    "Banh gio",
    "Banh khot",
    "Banh mi",
    "Banh pia",
    "Banh tet",
    "Banh trang nuong",
    "Banh xeo",
    "Bun bo Hue",
    "Bun dau mam tom",
    "Bun mam",
    "Bun rieu",
    "Bun thit nuong",
    "Ca kho to",
    "Canh chua",
    "Cao lau",
    "Chao long",
    "Com tam",
    "Goi cuon",
    "Hu tieu",
    "Mi Quang",
    "Nem chua",
    "Pho",
    "Xoi xeo",
]

MODEL_CONFIGS = [
    {
        "id": "vgg_finetune",
        "label": "VGG16 Fine-tune",
        "filename": "best_model_vgg_finetune.keras",
        "description": "VGG16 với toàn bộ các lớp được fine-tune trên tập dữ liệu món ăn Việt Nam",
        "color": "#6366f1",
    },
    {
        "id": "vgg_transfer",
        "label": "VGG16 Transfer",
        "filename": "best_model_vgg_transfer.keras",
        "description": "VGG16 Transfer Learning — chỉ huấn luyện các lớp fully-connected cuối",
        "color": "#0ea5e9",
    },
    {
        "id": "model_1",
        "label": "Custom CNN v1",
        "filename": "best_model_1.keras",
        "description": "Mạng CNN tùy chỉnh phiên bản 1 với kiến trúc sâu hơn (22M params)",
        "color": "#10b981",
    },
    {
        "id": "model_2",
        "label": "Custom CNN v2",
        "filename": "best_model_2.keras",
        "description": "Mạng CNN tùy chỉnh phiên bản 2 nhỏ gọn hơn (658K params)",
        "color": "#f59e0b",
    },
    {
        "id": "resnet_finetune",
        "label": "ResNet50 Fine-tune",
        "filename": "best_model_resnet_finetune.keras",
        "description": "ResNet50 với các lớp được fine-tune trên tập dữ liệu món ăn",
        "color": "#ec4899",
    },
    {
        "id": "resnet_transfer",
        "label": "ResNet50 Transfer",
        "filename": "best_model_resnet_transfer.keras",
        "description": "ResNet50 Transfer Learning — chỉ huấn luyện các lớp fully-connected cuối",
        "color": "#8b5cf6",
    },
]

IMG_SIZE = (224, 224)
TOP_K = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Food Model Comparison API",
    description="So sánh kết quả dự đoán từ 6 model nhận dạng món ăn",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the React frontend at /app
FE_DIR = Path(__file__).parent.parent / "Fe"
if FE_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FE_DIR), html=True), name="frontend")

# ---------------------------------------------------------------------------
# Model registry — lazy loading to avoid startup crash
# ---------------------------------------------------------------------------
_models: dict = {}
_load_errors: dict = {}


def _load_model(model_id: str, filename: str):
    """Load a single Keras model and cache it."""
    if model_id in _models:
        return _models[model_id]

    try:
        import tensorflow as tf
        path = MODEL_DIR / filename
        logger.info(f"Loading model: {filename}")
        t0 = time.perf_counter()
        model = tf.keras.models.load_model(str(path))
        elapsed = time.perf_counter() - t0
        logger.info(f"  ✓ {filename} loaded in {elapsed:.2f}s")
        _models[model_id] = model
        return model
    except Exception as e:
        logger.error(f"  ✗ Failed to load {filename}: {e}")
        _load_errors[model_id] = str(e)
        return None


def _get_all_models():
    """Ensure all models are loaded; return dict of id -> model (None if failed)."""
    for cfg in MODEL_CONFIGS:
        if cfg["id"] not in _models and cfg["id"] not in _load_errors:
            _load_model(cfg["id"], cfg["filename"])
    return _models


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def _preprocess_pil(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to normalized numpy array for model input."""
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


async def _image_from_url(url: str) -> Image.Image:
    """Fetch image from URL using httpx async client."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type and not url.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"URL không trỏ đến file ảnh hợp lệ. Content-Type: {content_type}",
                )
            return Image.open(io.BytesIO(response.content))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Không thể tải ảnh từ URL: {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Lỗi kết nối khi tải ảnh: {str(e)}")


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------
def _predict_single(model_id: str, model, arr: np.ndarray) -> dict:
    """Run inference on one model; return structured result."""
    t0 = time.perf_counter()
    preds = model.predict(arr, verbose=0)[0]  # shape (30,)
    elapsed = time.perf_counter() - t0

    top_indices = np.argsort(preds)[::-1][:TOP_K]
    top_classes = [
        {
            "rank": int(i + 1),
            "class_id": int(idx),
            "class_name": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}",
            "confidence": float(round(preds[idx] * 100, 2)),
        }
        for i, idx in enumerate(top_indices)
    ]

    return {
        "model_id": model_id,
        "top_prediction": top_classes[0]["class_name"],
        "top_confidence": top_classes[0]["confidence"],
        "top_k": top_classes,
        "inference_time_ms": float(round(elapsed * 1000, 2)),
    }


def _run_comparison(arr: np.ndarray) -> dict:
    """Run all models and build comparison response."""
    _get_all_models()
    results = []

    for cfg in MODEL_CONFIGS:
        mid = cfg["id"]
        if mid in _load_errors:
            results.append(
                {
                    "model_id": mid,
                    "label": cfg["label"],
                    "color": cfg["color"],
                    "description": cfg["description"],
                    "error": _load_errors[mid],
                }
            )
        elif mid in _models:
            prediction = _predict_single(mid, _models[mid], arr)
            results.append(
                {
                    **prediction,
                    "label": cfg["label"],
                    "color": cfg["color"],
                    "description": cfg["description"],
                }
            )
        else:
            results.append(
                {
                    "model_id": mid,
                    "label": cfg["label"],
                    "color": cfg["color"],
                    "description": cfg["description"],
                    "error": "Model chưa được tải",
                }
            )

    # Determine consensus
    top_preds = [r.get("top_prediction") for r in results if "top_prediction" in r]
    consensus = max(set(top_preds), key=top_preds.count) if top_preds else None
    agreement_count = top_preds.count(consensus) if consensus else 0

    return {
        "status": "success",
        "num_models": len(MODEL_CONFIGS),
        "consensus": {
            "prediction": consensus,
            "agreement": f"{agreement_count}/{len(MODEL_CONFIGS)} models đồng ý",
            "agreement_count": agreement_count,
        },
        "models": results,
        "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
    }


# ---------------------------------------------------------------------------
# Startup — eagerly load all models
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting model pre-load...")
    _get_all_models()
    loaded = len(_models)
    failed = len(_load_errors)
    logger.info(f"✓ Loaded {loaded} models, {failed} failed")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", summary="API Info")
async def root():
    return {
        "name": "Food Model Comparison API",
        "version": "1.0.0",
        "description": "So sánh kết quả dự đoán từ 6 model nhận dạng món ăn Việt Nam",
        "endpoints": {
            "GET /health": "Trạng thái API và các model",
            "GET /classes": "Danh sách 30 nhãn lớp",
            "POST /predict/upload": "Dự đoán từ file ảnh upload",
            "POST /predict/url": "Dự đoán từ URL ảnh",
        },
    }


@app.get("/health", summary="Health check")
async def health():
    _get_all_models()
    model_status = {}
    for cfg in MODEL_CONFIGS:
        mid = cfg["id"]
        if mid in _models:
            model_status[mid] = {"status": "loaded", "label": cfg["label"]}
        elif mid in _load_errors:
            model_status[mid] = {"status": "error", "label": cfg["label"], "error": _load_errors[mid]}
        else:
            model_status[mid] = {"status": "not_loaded", "label": cfg["label"]}

    all_ok = all(v["status"] == "loaded" for v in model_status.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "models_loaded": len(_models),
        "models_failed": len(_load_errors),
        "models": model_status,
    }


@app.get("/classes", summary="List all class labels")
async def get_classes():
    return {
        "total": len(CLASS_NAMES),
        "classes": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }


# ---------------------------------------------------------------------------
# YOLOv11 & Depth Anything v2 - Portion & Weight Estimation
# ---------------------------------------------------------------------------
YOLO_CLASSES = [
    "banh_bao", "banh_beo", "banh_bot_loc", "banh_mi", "banh_tet",
    "banh_trung_thu", "bun_bo_hue", "bun_dau_mam_tom", "com_lam",
    "cua_hap", "oc_buoi_hap", "pho", "rau_muong_xao", "thit_kho_tau",
    "xoi_xeo"
]

YOLO_CLASS_INFO = {
    # density_factor = typical_weight / (typical_area * typical_height)
    # Calibrated per-class so that at typical image framing the formula returns (min_w + max_w) / 2
    "banh_bao":       {"vi": "Bánh bao",        "en": "Banh bao",       "calories_per_100g": 230, "density_factor": 2750,  "min_w": 80,  "max_w": 250},
    "banh_beo":       {"vi": "Bánh bèo",        "en": "Banh beo",       "calories_per_100g": 120, "density_factor": 7400,  "min_w": 100, "max_w": 300},
    "banh_bot_loc":   {"vi": "Bánh bột lọc",    "en": "Banh bot loc",   "calories_per_100g": 180, "density_factor": 9300,  "min_w": 100, "max_w": 400},
    "banh_mi":        {"vi": "Bánh mì",         "en": "Banh mi",        "calories_per_100g": 270, "density_factor": 3200,  "min_w": 100, "max_w": 250},
    "banh_tet":       {"vi": "Bánh tét",        "en": "Banh tet",       "calories_per_100g": 200, "density_factor": 4900,  "min_w": 150, "max_w": 600},
    "banh_trung_thu": {"vi": "Bánh trung thu",  "en": "Banh trung thu", "calories_per_100g": 380, "density_factor": 4000,  "min_w": 100, "max_w": 300},
    "bun_bo_hue":     {"vi": "Bún bò Huế",     "en": "Bun bo Hue",     "calories_per_100g": 110, "density_factor": 15700, "min_w": 300, "max_w": 800},
    "bun_dau_mam_tom":{"vi": "Bún đậu mắm tôm","en": "Bun dau mam tom","calories_per_100g": 150, "density_factor": 12400, "min_w": 250, "max_w": 700},
    "com_lam":        {"vi": "Cơm lam",         "en": "Com lam",        "calories_per_100g": 180, "density_factor": 5500,  "min_w": 150, "max_w": 450},
    "cua_hap":        {"vi": "Cua hấp",         "en": "Cua hap",        "calories_per_100g": 100, "density_factor": 8000,  "min_w": 200, "max_w": 600},
    "oc_buoi_hap":    {"vi": "Ốc bươu hấp",    "en": "Oc buou hap",    "calories_per_100g": 80,  "density_factor": 8200,  "min_w": 150, "max_w": 500},
    "pho":            {"vi": "Phở",             "en": "Pho",            "calories_per_100g": 100, "density_factor": 15700, "min_w": 300, "max_w": 800},
    "rau_muong_xao":  {"vi": "Rau muống xào",   "en": "Rau muong xao",  "calories_per_100g": 70,  "density_factor": 5100,  "min_w": 100, "max_w": 350},
    "thit_kho_tau":   {"vi": "Thịt kho tàu",   "en": "Thit kho tau",   "calories_per_100g": 220, "density_factor": 6700,  "min_w": 150, "max_w": 500},
    "xoi_xeo":        {"vi": "Xôi xéo",        "en": "Xoi xeo",        "calories_per_100g": 240, "density_factor": 4900,  "min_w": 150, "max_w": 500},
}

class PredictionItem(BaseModel):
    class_id: int
    dish_name_en: str
    dish_name_vi: str
    confidence: float
    serving_size_g: Optional[float] = None
    box: Optional[List[float]] = None

class PredictResponse(BaseModel):
    class_id: int
    dish_name_en: str
    dish_name_vi: str
    confidence: float
    top5: List[PredictionItem]
    detections: Optional[List[PredictionItem]] = None
    serving_size_g: Optional[float] = None

_yolo_model = None
_depth_estimator = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        logger.info(f"Loading YOLOv11 model from {MODEL_DIR / 'yolobv11' / 'best.pt'}...")
        _yolo_model = YOLO(str(MODEL_DIR / "yolobv11" / "best.pt"))
        logger.info("YOLOv11 loaded.")
    return _yolo_model

def get_depth_estimator():
    global _depth_estimator
    if _depth_estimator is None:
        import torch
        from transformers import pipeline
        logger.info("Loading Depth Anything v2 model...")
        device = 0 if torch.cuda.is_available() else -1
        _depth_estimator = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device
        )
        logger.info("Depth Anything v2 loaded.")
    return _depth_estimator

def estimate_food_weight(
    class_name: str,
    box: list,
    depth_map_array: np.ndarray,
    d_min: float,
    d_max: float,
    d_range: float
) -> float:
    # Lookup class metadata first so typical_weight is available for the degenerate-box fallback
    info = YOLO_CLASS_INFO.get(class_name, {
        "calories_per_100g": 150,
        "density_factor": 6000,
        "min_w": 100,
        "max_w": 600
    })
    min_w = info.get("min_w", 100)
    max_w = info.get("max_w", 600)
    density_factor = info.get("density_factor", 6000)
    typical_weight = (min_w + max_w) / 2.0

    h_img, w_img = depth_map_array.shape
    xmin, ymin, xmax, ymax = box

    x1, y1 = max(0, int(xmin)), max(0, int(ymin))
    x2, y2 = min(w_img, int(xmax)), min(h_img, int(ymax))

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return float(typical_weight)

    # Normalize depth patch; clip to handle values outside the 2nd–98th percentile range
    box_depth = depth_map_array[y1:y2, x1:x2]
    normalized_box_depth = np.clip((box_depth - d_min) / d_range, 0.0, 1.0)

    # Background depth: median of the 1-pixel border ring is more robust than global min
    # (global min may belong to the food itself when it fills the bounding box)
    border_pixels = np.concatenate([
        normalized_box_depth[0, :],
        normalized_box_depth[-1, :],
        normalized_box_depth[1:-1, 0],
        normalized_box_depth[1:-1, -1]
    ])
    bg_depth = (
        float(np.median(border_pixels))
        if border_pixels.size >= 4
        else float(normalized_box_depth.min())
    )
    height_map = np.maximum(0.0, normalized_box_depth - bg_depth)
    avg_height = float(np.mean(height_map))

    w_frac = (x2 - x1) / w_img
    h_frac = (y2 - y1) / h_img
    box_area = w_frac * h_frac

    # Unified formula: weight = area × height × density_factor
    # density_factor encodes per-class typical framing so at typical (area, height) the
    # result equals typical_weight = (min_w + max_w) / 2
    raw_weight = box_area * avg_height * density_factor
    estimated_weight = max(min_w, min(max_w, raw_weight))

    logger.info(
        f"[Weight Calc] '{class_name}': "
        f"box_area={box_area:.4f}, "
        f"avg_height={avg_height:.4f}, "
        f"density_factor={density_factor}, "
        f"raw_weight={raw_weight:.2f}g -> "
        f"final_weight={estimated_weight:.2f}g (clamped to {min_w}g-{max_w}g)"
    )
    return float(round(estimated_weight, 2))

@app.post("/predict", response_model=PredictResponse, summary="Classify and estimate food calories and weight using YOLOv11 and Depth Anything v2")
async def predict(file: UploadFile = File(..., description="Food image (JPEG/PNG/WEBP)")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize image if too large to speed up YOLO and Depth Anything v2
        max_dim = 800
        w_img, h_img = img.size
        img_resized = img
        if max(w_img, h_img) > max_dim:
            scale = max_dim / max(w_img, h_img)
            new_w = int(w_img * scale)
            new_h = int(h_img * scale)
            img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
        w_res, h_res = img_resized.size
        
        yolo_model = get_yolo_model()
        yolo_results = yolo_model(img_resized, conf=0.1)
        
        depth_estimator = get_depth_estimator()
        depth_result = depth_estimator(img_resized)
        depth_map_array = np.array(depth_result["depth"], dtype=np.float32)
        
        # Precompute depth map statistics once
        d_min = float(np.percentile(depth_map_array, 2))
        d_max = float(np.percentile(depth_map_array, 98))
        d_range = d_max - d_min if (d_max - d_min) > 1e-6 else 1.0
        
        detections = []
        boxes = yolo_results[0].boxes
        
        CONFIDENCE_THRESHOLD = 0.65
        logger.info(f"[Detect] Found {len(boxes)} raw YOLO detections on image ({w_res}x{h_res})")
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name_raw = yolo_model.names.get(cls_id, f"class_{cls_id}")
            
            # Skip weight estimation and do not add to detections if confidence is too low (<0.65)
            if conf < CONFIDENCE_THRESHOLD:
                logger.info(f"  [-] Skipped '{class_name_raw}' (confidence={conf:.3f} < {CONFIDENCE_THRESHOLD})")
                continue
                
            logger.info(f"  [+] Accepted '{class_name_raw}' (confidence={conf:.3f} >= {CONFIDENCE_THRESHOLD})")
            
            info = YOLO_CLASS_INFO.get(class_name_raw, {
                "vi": class_name_raw.replace("_", " ").title(),
                "en": class_name_raw.replace("_", " ").title(),
                "calories_per_100g": 150
            })
            
            weight_g = estimate_food_weight(class_name_raw, xyxy, depth_map_array, d_min, d_max, d_range)
            box_norm = [
                round(xyxy[0] / w_res, 4),
                round(xyxy[1] / h_res, 4),
                round(xyxy[2] / w_res, 4),
                round(xyxy[3] / h_res, 4)
            ]
            
            detections.append(PredictionItem(
                class_id=cls_id,
                dish_name_en=info["en"],
                dish_name_vi=info["vi"],
                confidence=conf,
                serving_size_g=weight_g,
                box=box_norm
            ))
            
        if not detections:
            fallback_class = "pho"
            info = YOLO_CLASS_INFO[fallback_class]
            detections.append(PredictionItem(
                class_id=11,
                dish_name_en=info["en"],
                dish_name_vi=info["vi"],
                confidence=0.0,
                serving_size_g=400.0,
                box=None
            ))
            
        detections.sort(key=lambda x: x.confidence, reverse=True)
        best = detections[0]
        
        top5 = []
        for det in detections[:5]:
            top5.append(PredictionItem(
                class_id=det.class_id,
                dish_name_en=det.dish_name_en,
                dish_name_vi=det.dish_name_vi,
                confidence=det.confidence,
                serving_size_g=det.serving_size_g,
                box=det.box
            ))
            
        if len(top5) < 5:
            existing_ids = {item.class_id for item in top5}
            for cls_name in YOLO_CLASSES:
                if len(top5) >= 5:
                    break
                cls_id_found = None
                for k, v in yolo_model.names.items():
                    if v == cls_name:
                        cls_id_found = k
                        break
                if cls_id_found is not None and cls_id_found not in existing_ids:
                    info = YOLO_CLASS_INFO[cls_name]
                    top5.append(PredictionItem(
                        class_id=cls_id_found,
                        dish_name_en=info["en"],
                        dish_name_vi=info["vi"],
                        confidence=0.01,
                        serving_size_g=200.0,
                        box=None
                    ))
                    existing_ids.add(cls_id_found)
                    
        real_detections = [d for d in detections if d.confidence >= 0.25]
        if not real_detections:
            real_detections = [best]
            
        serving_size_g = sum(d.serving_size_g for d in real_detections if d.serving_size_g is not None)
        
        response = PredictResponse(
            class_id=best.class_id,
            dish_name_en=best.dish_name_en,
            dish_name_vi=best.dish_name_vi,
            confidence=best.confidence,
            top5=top5,
            detections=detections,
            serving_size_g=round(serving_size_g, 2)
        )
        return response
    except Exception as e:
        logger.error(f"YOLO + Depth prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/upload", summary="Predict from uploaded image")
async def predict_upload(file: UploadFile = File(...)):
    """Upload một file ảnh và nhận kết quả so sánh từ tất cả 6 model."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File phải là ảnh. Nhận được: {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file ảnh: {str(e)}")

    arr = _preprocess_pil(image)
    result = _run_comparison(arr)
    result["source"] = "upload"
    result["filename"] = file.filename
    return JSONResponse(content=result)


class PredictURLRequest(BaseModel):
    url: str


@app.post("/predict/url", summary="Predict from image URL")
async def predict_url(body: PredictURLRequest):
    """Cung cấp URL ảnh và nhận kết quả so sánh từ tất cả 6 model."""
    if not body.url.strip():
        raise HTTPException(status_code=400, detail="URL không được để trống")

    image = await _image_from_url(body.url.strip())
    arr = _preprocess_pil(image)
    result = _run_comparison(arr)
    result["source"] = "url"
    result["image_url"] = body.url.strip()
    return JSONResponse(content=result)
