from pathlib import Path
import shutil
from ultralytics import YOLO
import multiprocessing  # ← thêm dòng này

# ===================== CONFIG PATHS =====================
# (giữ nguyên phần config, không cần thay đổi)

CONFIG_DIR = Path("config")
MODEL_YAML = CONFIG_DIR / "yolo26x-p2.yaml"
DATA_YAML = CONFIG_DIR / "scut_head.yaml"
PRETRAINED_WEIGHTS = "yolo26x.pt"

PROJECT_NAME = "SCUT_Head_Project"
EXPERIMENT_NAME = "yolo26x_p2_v0_800"
CHECKPOINT_DIR = Path("ckpts")
FINAL_BEST_NAME = f"{EXPERIMENT_NAME}.pt"

# ===================== CHỈ CHẠY KHI LÀ MAIN PROCESS =====================
if __name__ == '__main__':
    multiprocessing.freeze_support()  # ← Bắt buộc trên Windows để tránh lỗi spawn

    # ===================== MODEL INITIALIZATION =====================
    print("🚀 Khởi tạo mô hình YOLO26x với kiến trúc P2...")
    print(f"   - Pretrained weights: {PRETRAINED_WEIGHTS}")
    print(f"   - Custom architecture: {MODEL_YAML}")

    model = YOLO(MODEL_YAML).load(PRETRAINED_WEIGHTS)
    model.info(verbose=True)

    # ===================== TRAINING CONFIGURATION =====================
    print("\n🚀 Bắt đầu huấn luyện với P2 detection head...")
    print(f"   - Dataset: {DATA_YAML}")
    print(f"   - Image size: 800")
    print(f"   - Batch size: 4")
    print(f"   - Epochs: 100 (early stopping patience=20)")
    print("   - Augmentation mạnh cho small/occluded heads")

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=800,
        batch=4,
        patience=20,

        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,

        mosaic=1.0,
        close_mosaic=10,
        mixup=0.2,
        copy_paste=0.4,
        fliplr=0.5,
        scale=0.6,
        rect=False,

        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True,

        verbose=True,
        amp=True,
        single_cls=True,
        freeze=10,
    )

    print("\n✅ Huấn luyện hoàn tất!")

    # ===================== SAVE BEST CHECKPOINT =====================
    if results is not None:
        weights_dir = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights"
        best_src = weights_dir / "best.pt"

        if best_src.exists():
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            best_dst = CHECKPOINT_DIR / FINAL_BEST_NAME
            shutil.copy2(best_src, best_dst)
            print(f"📦 Đã sao chép best checkpoint tới: {best_dst}")
        else:
            print("⚠️ Không tìm thấy file best.pt trong thư mục weights.")
            print("   Kiểm tra:", weights_dir)
    else:
        print("⚠️ Không có kết quả huấn luyện hợp lệ.")