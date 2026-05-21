"""
FastAPI Server for Real-time YOLO Object Detection
Receives image frames from mobile app and returns bounding box detections
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import yaml
import os
import uvicorn
from typing import List, Dict, Optional
import logging

# ==============================
# Configuration
# ==============================
MODEL_PATH = "runs/detect/runs/detect/train/weights/best.pt"
DATA_YAML_PATH = "dataset/data.yaml"
CONFIDENCE_THRESHOLD = 0.5
PORT = 8000
HOST = "0.0.0.0"  # Accessible from any IP

# ==============================
# Logging Setup
# ==============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Global Variables
# ==============================
model = None
CLASS_NAMES = {
    0: "Drill",
    1: "Hammer",
    2: "Pliers",
    3: "Screwdriver",
    4: "Wrench"
}

# ==============================
# Load Class Names from data.yaml
# ==============================
def load_class_names_from_yaml(yaml_path):
    """Load class names from data.yaml file"""
    global CLASS_NAMES
    
    if not os.path.exists(yaml_path):
        logger.warning(f"data.yaml not found at {yaml_path}, using defaults")
        return
    
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        if 'names' in data:
            CLASS_NAMES = data['names']
            logger.info(f"Loaded {len(CLASS_NAMES)} class names from data.yaml")
        else:
            logger.warning("'names' field not found in data.yaml")
    except Exception as e:
        logger.error(f"Error loading data.yaml: {e}")

# ==============================
# Load YOLO Model
# ==============================
def load_model_on_startup():
    """Load YOLO model on server startup"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        return False
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        logger.info("✓ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# ==============================
# FastAPI App Setup
# ==============================
app = FastAPI(
    title="YOLO Detection API",
    description="Real-time object detection API using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific URLs in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Helper Functions
# ==============================
def process_image(image_bytes):
    """Convert image bytes to numpy array"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image_array
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def format_detection_results(results, original_width, original_height):
    """Format YOLO results into JSON-friendly format"""
    detections = []
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return detections
    
    boxes = results[0].boxes
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        
        # Skip low confidence detections
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        # Get class name
        class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
        
        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bounding_box": {
                "x": round(x1, 2),
                "y": round(y1, 2),
                "width": round(width, 2),
                "height": round(height, 2),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        }
        
        detections.append(detection)
    
    return detections

# ==============================
# API Endpoints
# ==============================

@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup"""
    logger.info("Starting YOLO Detection API Server...")
    load_class_names_from_yaml(DATA_YAML_PATH)
    
    if not load_model_on_startup():
        logger.error("Failed to load model!")
    else:
        logger.info(f"Server ready to accept detection requests on port {PORT}")

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "api": "YOLO Detection API",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    return {"status": "healthy", "message": "API is running"}

@app.get("/info", tags=["Info"])
async def get_info():
    """Get API information and loaded classes"""
    return {
        "api_name": "YOLO Detection API",
        "version": "1.0.0",
        "model_path": MODEL_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "model_loaded": model is not None
    }

@app.post("/detect", tags=["Detection"])
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with detection results including bounding boxes and class names
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Process image
        image_array = process_image(contents)
        if image_array is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )
        
        # Get original dimensions
        height, width = image_array.shape[:2]
        
        # Run YOLO detection
        logger.info(f"Processing image: {file.filename} ({width}x{height})")
        results = model(image_array, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Format results
        detections = format_detection_results(results, width, height)
        
        response = {
            "status": "success",
            "file_name": file.filename,
            "image_dimensions": {
                "width": width,
                "height": height
            },
            "total_detections": len(detections),
            "detections": detections
        }
        
        logger.info(f"Detection complete: {len(detections)} objects found")
        return response
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

@app.post("/detect-batch", tags=["Detection"])
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Detect objects in multiple uploaded images
    
    Args:
        files: List of image files
    
    Returns:
        JSON with detection results for each image
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    batch_results = []
    
    for file in files:
        try:
            contents = await file.read()
            image_array = process_image(contents)
            
            if image_array is None:
                batch_results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "message": "Invalid image format"
                })
                continue
            
            height, width = image_array.shape[:2]
            results = model(image_array, conf=CONFIDENCE_THRESHOLD, verbose=False)
            detections = format_detection_results(results, width, height)
            
            batch_results.append({
                "file_name": file.filename,
                "status": "success",
                "image_dimensions": {"width": width, "height": height},
                "total_detections": len(detections),
                "detections": detections
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            batch_results.append({
                "file_name": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {
        "status": "success",
        "total_files": len(files),
        "results": batch_results
    }

@app.post("/detect-base64", tags=["Detection"])
async def detect_base64(data: dict):
    """
    Detect objects in base64 encoded image
    
    Args:
        data: JSON with 'image' field containing base64 encoded image
    
    Returns:
        JSON with detection results
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        import base64
        
        if "image" not in data:
            raise HTTPException(
                status_code=400,
                detail="'image' field is required in request body"
            )
        
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image_array = process_image(image_data)
        
        if image_array is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )
        
        height, width = image_array.shape[:2]
        results = model(image_array, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = format_detection_results(results, width, height)
        
        return {
            "status": "success",
            "image_dimensions": {"width": width, "height": height},
            "total_detections": len(detections),
            "detections": detections
        }
        
    except Exception as e:
        logger.error(f"Error during base64 detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("YOLO Detection FastAPI Server")
    logger.info("="*60)
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Documentation: http://localhost:{PORT}/docs")
    logger.info(f"Alternative docs: http://localhost:{PORT}/redoc")
    logger.info("="*60)
    
    uvicorn.run(
        "fastapi_detection_server:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
