from ultralytics import YOLO
import torch
import os

# ==============================
# Check GPU availability
# ==============================
print("=" * 60)
print("CHECKING HARDWARE")
print("=" * 60)

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
print(f"PyTorch Version: {torch.__version__}")

if cuda_available:
    device = "0"  # Use first GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ Using GPU: {gpu_name}")
else:
    device = "cpu"  # Fallback to CPU
    print(f"⚠️  Using CPU (GPU not detected)")
    print("\nTo use GPU, reinstall PyTorch with CUDA support:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("=" * 60)

if __name__ == '__main__':
    # Use smaller model for faster training (yolov8n is fastest)
    model = YOLO("yolov8n.pt")

    # ==============================
    # Train the model (OPTIMIZED FOR SPEED)
    # ==============================
    results = model.train(
        data="dataset/data.yaml",           # Path to dataset YAML file
        epochs=5,                           # Reduced epochs for faster training
        imgsz=416,                          # Optimized image size
        device=device,                      # GPU device or 'cpu'
        batch=16,                           # Larger batch size for GPU (reduced for CPU)
        patience=2,                         # Early stopping patience
        save=True,                          # Save model checkpoints
        save_period=5,                      # Save checkpoint every 5 epochs
        close_mosaic=4,                     # Disable mosaic after 4 epochs
        
        # Minimal augmentation for speed
        mosaic=0.5,                         # Reduce mosaic ratio
        mixup=0.0,                          # Disable mixup
        copy_paste=0.0,                     # Disable copy-paste
        auto_augment=None,                  # Disable auto-augment for speed
        erasing=0.0,                        # Disable erasing
        
        # Minimal geometric augmentation
        degrees=0.0,                        # No rotation
        translate=0.05,                     # Minimal translation
        scale=0.3,                          # Minimal scale
        flipud=0.0,                         # No flip up-down
        fliplr=0.3,                         # Minimal flip left-right
        perspective=0.0,                    # No perspective
        shear=0.0,                          # No shear
        
        # Minimal HSV augmentation
        hsv_h=0.0,                          # No HSV hue
        hsv_s=0.0,                          # No HSV saturation
        hsv_v=0.0,                          # No HSV value
        
        # Optimizer settings
        optimizer="SGD",                    # SGD is faster than Adam
        lr0=0.01,                           # Initial learning rate
        lrf=0.01,                           # Final learning rate ratio
        momentum=0.937,                     # Optimizer momentum
        weight_decay=0.0005,                # Weight decay
        warmup_epochs=1,                    # Reduced warmup
        warmup_bias_lr=0.1,                 # Warmup bias learning rate
        
        # Loss weights (simplified)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Other optimized settings
        workers=4,                          # Number of data loader workers
        project="runs/detect",              # Project directory
        name="train",                       # Experiment name
        exist_ok=True,                      # Allow overwriting
        pretrained=True,                    # Use pretrained weights
        verbose=True,                       # Print output
        seed=0,                             # Random seed
        amp=False,                          # Disable automatic mixed precision for stability
    )

    print("\n✅ Training completed!")
    print(f"Best model saved at: runs/detect/train/weights/best.pt")
    print(f"Last model saved at: runs/detect/train/weights/last.pt")

