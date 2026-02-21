from pathlib import Path
import shutil
import multiprocessing
from ultralytics import YOLO

# ===================== CONFIG PATHS =====================
CONFIG_DIR = Path("config")
DATA_YAML = CONFIG_DIR / "scut_head_clahe.yaml"
PRETRAINED_WEIGHTS = "yolo12x.pt"

# Training output settings
PROJECT_NAME = "SCUT_Head_Project"
EXPERIMENT_NAME = "yolo12x_CLAHE_v0"
CHECKPOINT_DIR = Path("ckpts")
FINAL_BEST_NAME = f"{EXPERIMENT_NAME}.pt"

# ===================== MAIN EXECUTION BLOCK =====================
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Bắt buộc trên Windows để hỗ trợ multiprocessing DataLoader

    # ===================== MODEL INITIALIZATION =====================
    print("🚀 Khởi tạo mô hình YOLO12x...")
    print(f"   - Pretrained weights: {PRETRAINED_WEIGHTS}")

    # Load pretrained model (không override P2 ở script này)
    model = YOLO(PRETRAINED_WEIGHTS)

    # In thông tin mô hình
    model.info(verbose=True)

    # ===================== TRAINING CONFIGURATION =====================
    print("\n🚀 Bắt đầu huấn luyện với dataset CLAHE preprocessing...")
    print(f"   - Dataset: {DATA_YAML}")
    print(f"   - Image size: 640")
    print(f"   - Batch size: 8")
    print(f"   - Epochs: 100 (early stopping patience=20)")
    print("   - Augmentation mạnh cho small/occluded heads")
    print("   - Freeze: 2 layers đầu")

    results = model.train(
        # Core settings
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,

        # Optimizer & scheduler
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,

        # Data augmentation (tuned for small/occluded heads)
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.2,
        copy_paste=0.4,
        fliplr=0.5,
        scale=0.6,
        rect=False,

        # Project & run naming
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,

        # Performance & logging
        verbose=True,
        amp=True,                # Automatic Mixed Precision
        single_cls=True,         # Chỉ detect 1 class (head)
        freeze=2,                # Freeze 2 layer đầu (nhẹ hơn so với yolo26x)
    )

    print("\n✅ Huấn luyện hoàn tất!")

    # ===================== SAVE BEST CHECKPOINT TO ckpts/ =====================
    if results is not None:
        weights_dir = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights"
        best_src = weights_dir / "best.pt"

        if best_src.exists():
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            best_dst = CHECKPOINT_DIR / FINAL_BEST_NAME

            shutil.copy2(best_src, best_dst)
            print(f"📦 Đã sao chép best checkpoint tới:")
            print(f"   → {best_dst}")
        else:
            print("⚠️ Không tìm thấy file best.pt trong thư mục weights.")
            print("   Kiểm tra thư mục kết quả:", weights_dir)
    else:
        print("⚠️ Không có kết quả huấn luyện hợp lệ.")