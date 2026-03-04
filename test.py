from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# ==============================
# Load trained model
# ==============================
# Try to find the model in common locations
model_paths = [
    "runs/detect/runs/detect/train/weights/best.pt",  # Standard location after training
    "runs/detect/train/weights/best.pt",              # Alternative location
    "best.pt",                                        # Current directory
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print(f"❌ Model not found!")
    print("Checked locations:")
    for path in model_paths:
        print(f"  - {path}")
    print("\nPlease train the model first using: python train.py")
    exit(1)

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# ==============================
# Class names
# ==============================
class_names = {
    0: "Drill",
    1: "Hammer",
    2: "Pliers",
    3: "Screwdriver",
    4: "Wrench"
}

# ==============================
# Test on single image
# ==============================
def test_single_image(image_path, conf_threshold=0.5):
    """Test on a single image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\nTesting on image: {image_path}")
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        verbose=False
    )
    
    # Process results
    for result in results:
        print(f"Image size: {result.orig_shape}")
        
        if len(result.boxes) == 0:
            print("❌ No objects detected")
        else:
            print(f"✅ Detected {len(result.boxes)} object(s):")
            
            for idx, box in enumerate(result.boxes):
                # Extract information
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                print(f"\n  Object {idx + 1}:")
                print(f"    Class: {class_names.get(class_id, 'Unknown')} (ID: {class_id})")
                print(f"    Confidence: {confidence:.2%}")
                print(f"    Bounding Box: ({x1}, {y1}, {x2}, {y2})")

# ==============================
# Test on dataset
# ==============================
def test_dataset(dataset_path="dataset/test/images", conf_threshold=0.5):
    """Test on all images in test dataset"""
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    image_files = list(Path(dataset_path).glob("*.[jJ][pP][gG]")) + \
                  list(Path(dataset_path).glob("*.[jJ][pP][eE][gG]")) + \
                  list(Path(dataset_path).glob("*.[pP][nN][gG]"))
    
    if not image_files:
        print(f"❌ No images found in {dataset_path}")
        return
    
    print(f"\n📁 Testing on {len(image_files)} images from {dataset_path}")
    
    results = model.predict(
        source=dataset_path,
        conf=conf_threshold,
        verbose=False
    )
    
    total_detections = 0
    for idx, result in enumerate(results):
        num_detections = len(result.boxes)
        total_detections += num_detections
        print(f"  Image {idx + 1}: {num_detections} detection(s)")
    
    print(f"✅ Total detections: {total_detections}")

# ==============================
# Run tests
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 Object Detection Test")
    print("=" * 60)
    
    # Test on single image (if available)
    test_image = "test.jpg"
    if os.path.exists(test_image):
        test_single_image(test_image, conf_threshold=0.5)
    
    # Test on entire test dataset
    print("\n" + "=" * 60)
    test_dataset("dataset/test/images", conf_threshold=0.5)
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
