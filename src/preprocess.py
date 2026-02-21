import os
import shutil
import cv2
from tqdm import tqdm
import warnings

def apply_clahe_on_lab(img_bgr, clip_limit=2.0, tile_grid_size=(16, 16)):
    """
    Áp dụng CLAHE chỉ trên kênh L của không gian màu LAB.
    
    Args:
        img_bgr (np.ndarray): Ảnh đầu vào ở định dạng BGR (từ cv2.imread).
        clip_limit (float): Giới hạn clip cho CLAHE (thường 2.0 - 4.0).
        tile_grid_size (tuple): Kích thước ô lưới (ví dụ: (16, 16)).
    
    Returns:
        np.ndarray: Ảnh đã tăng cường tương phản (BGR).
    """
    if img_bgr is None:
        raise ValueError("Ảnh đầu vào không hợp lệ (None).")
    
    # Chuyển sang LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # Tách kênh
    l, a, b = cv2.split(lab)
    
    # Tạo và áp dụng CLAHE lên kênh L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # Ghép lại và chuyển về BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    result_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result_bgr


def preprocess_and_copy_dataset(
    source_dir: str,
    dest_dir: str,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (16, 16),
    overwrite: bool = False
):
    """
    Sao chép cấu trúc dataset, áp dụng CLAHE cho tất cả ảnh .jpg trong thư mục 'images',
    giữ nguyên file labels và các file khác.
    
    Args:
        source_dir (str): Thư mục gốc dataset (ví dụ: /kaggle/input/scut-head/PartA)
        dest_dir (str): Thư mục đích để lưu dataset đã tiền xử lý
        clip_limit (float): Tham số clip cho CLAHE
        tile_grid_size (tuple): Kích thước tile cho CLAHE
        overwrite (bool): Nếu True, xóa folder đích nếu đã tồn tại (cẩn thận!)
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Thư mục nguồn không tồn tại: {source_dir}")
    
    # Xử lý folder đích nếu đã tồn tại
    if os.path.exists(dest_dir):
        if overwrite:
            print(f"Xóa thư mục đích cũ: {dest_dir}")
            shutil.rmtree(dest_dir)
        else:
            warnings.warn(
                f"Thư mục đích đã tồn tại: {dest_dir}\n"
                "Nếu muốn ghi đè, đặt overwrite=True. Hiện tại sẽ bỏ qua."
            )
            return
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Các subfolder chính: train / valid / test
    subfolders = ['train', 'valid', 'test']
    
    for sub in subfolders:
        source_sub = os.path.join(source_dir, sub)
        dest_sub = os.path.join(dest_dir, sub)
        
        if not os.path.exists(source_sub):
            print(f"Không tìm thấy subfolder {sub}, bỏ qua.")
            continue
        
        os.makedirs(dest_sub, exist_ok=True)
        
        # Sao chép toàn bộ folder labels (nếu có)
        source_labels = os.path.join(source_sub, 'labels')
        dest_labels = os.path.join(dest_sub, 'labels')
        if os.path.exists(source_labels):
            shutil.copytree(source_labels, dest_labels, dirs_exist_ok=True)
            print(f"Đã sao chép labels: {sub}/labels")
        
        # Xử lý folder images
        source_images = os.path.join(source_sub, 'images')
        dest_images = os.path.join(dest_sub, 'images')
        if not os.path.exists(source_images):
            print(f"Không tìm thấy images trong {sub}, bỏ qua.")
            continue
        
        os.makedirs(dest_images, exist_ok=True)
        
        # Lấy danh sách file ảnh .jpg (không phân biệt hoa thường)
        image_files = [f for f in os.listdir(source_images) 
                       if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Không tìm thấy file .jpg trong {source_images}")
            continue
        
        print(f"Xử lý {len(image_files)} ảnh trong {sub}/images...")
        
        # Tiến trình tqdm để theo dõi
        for filename in tqdm(image_files, desc=f"CLAHE {sub}", unit="img"):
            src_path = os.path.join(source_images, filename)
            dst_path = os.path.join(dest_images, filename)
            
            try:
                img = cv2.imread(src_path)
                if img is None:
                    print(f"  Không đọc được: {filename}")
                    continue
                
                enhanced = apply_clahe_on_lab(
                    img, clip_limit=clip_limit, tile_grid_size=tile_grid_size
                )
                
                success = cv2.imwrite(dst_path, enhanced)
                if not success:
                    print(f"  Lỗi lưu file: {filename}")
            except Exception as e:
                print(f"  Lỗi xử lý {filename}: {e}")
        
        print(f"Hoàn thành subfolder: {sub}\n")


# ==================== SỬ DỤNG ====================
if __name__ == "__main__":
    SOURCE_DIR = "dataset\PartA"
    DEST_DIR   = "dataset\PartA_CLAHE"  

    # Nếu muốn ghi đè (xóa folder cũ nếu tồn tại), đặt overwrite=True
    preprocess_and_copy_dataset(
        source_dir=SOURCE_DIR,
        dest_dir=DEST_DIR,
        clip_limit=2.0,
        tile_grid_size=(16, 16),
        overwrite=False   # Đổi thành True nếu cần xóa folder cũ
    )
    
    print("Hoàn tất tiền xử lý CLAHE và sao chép dataset!")