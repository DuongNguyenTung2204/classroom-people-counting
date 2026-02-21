import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
from config.inference_config import InferenceConfig
from utils.utils import horizontal_flip, apply_clahe_on_lab, revert_horizontal_flip_boxes

def process_single_image(
    orig_img: np.ndarray,
    model1: YOLO,
    model2: YOLO,
    device: str | int,
    cfg: InferenceConfig
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Thực hiện inference ensemble và WBF.
    Trả về:
    - final_count: số lượng head detect được
    - final_boxes_pixel: các box đã scale về pixel (numpy array [N,4])
    - final_scores: confidence scores tương ứng (numpy array [N])
    """
    if orig_img is None:
        raise ValueError("Không đọc được ảnh đầu vào")

    h, w = orig_img.shape[:2]

    # Model 1 - Original
    results1 = model1.predict(
        source=orig_img,
        conf=cfg.conf_model1,
        imgsz=cfg.imgsz,
        save=False,
        show=False,
        device=device,
        verbose=False
    )
    result1 = results1[0]
    boxes1 = result1.boxes.xyxy.cpu().numpy() if len(result1.boxes) > 0 else np.empty((0,4))
    scores1 = result1.boxes.conf.cpu().numpy() if len(result1.boxes) > 0 else np.empty((0,))
    labels1 = result1.boxes.cls.cpu().numpy().astype(int) if len(result1.boxes) > 0 else np.empty((0,))

    # Model 1 - Flip
    flipped_img = horizontal_flip(orig_img)
    results_flip = model1.predict(
        source=flipped_img,
        conf=cfg.conf_model1_flip,
        imgsz=cfg.imgsz,
        save=False,
        show=False,
        device=device,
        verbose=False
    )
    result_flip = results_flip[0]
    boxes_flip = result_flip.boxes.xyxy.cpu().numpy() if len(result_flip.boxes) > 0 else np.empty((0,4))
    scores_flip = result_flip.boxes.conf.cpu().numpy() if len(result_flip.boxes) > 0 else np.empty((0,))
    labels_flip = result_flip.boxes.cls.cpu().numpy().astype(int) if len(result_flip.boxes) > 0 else np.empty((0,))

    if len(boxes_flip) > 0:
        boxes_flip = revert_horizontal_flip_boxes(boxes_flip, w)

    # Model 2: CLAHE
    clahe_img = apply_clahe_on_lab(
        orig_img,
        clip_limit=cfg.clahe_clip_limit,
        tile_size=cfg.clahe_tile_size
    )
    results2 = model2.predict(
        source=clahe_img,
        conf=cfg.conf_model2,
        imgsz=cfg.imgsz,
        save=False,
        show=False,
        device=device,
        verbose=False
    )
    result2 = results2[0]
    boxes2 = result2.boxes.xyxy.cpu().numpy() if len(result2.boxes) > 0 else np.empty((0,4))
    scores2 = result2.boxes.conf.cpu().numpy() if len(result2.boxes) > 0 else np.empty((0,))
    labels2 = result2.boxes.cls.cpu().numpy().astype(int) if len(result2.boxes) > 0 else np.empty((0,))

    # Normalize boxes về [0,1]
    boxes1_norm   = boxes1   / np.array([w, h, w, h]) if len(boxes1)   > 0 else boxes1
    boxes_flip_norm = boxes_flip / np.array([w, h, w, h]) if len(boxes_flip) > 0 else boxes_flip
    boxes2_norm   = boxes2   / np.array([w, h, w, h]) if len(boxes2)   > 0 else boxes2

    boxes_list  = [boxes1_norm, boxes_flip_norm, boxes2_norm]
    scores_list = [scores1,     scores_flip,     scores2]
    labels_list = [labels1,     labels_flip,     labels2]

    # Weighted Boxes Fusion
    try:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=cfg.wbf_weights,
            iou_thr=cfg.wbf_iou_thr,
            skip_box_thr=cfg.wbf_skip_box_thr,
            conf_type=cfg.wbf_conf_type,
            allows_overflow=False
        )
    except Exception as e:
        raise RuntimeError(f"WBF lỗi: {e}")

    # Lọc confidence cuối
    keep = fused_scores > cfg.final_score_thr
    final_boxes_norm   = fused_boxes[keep]
    final_scores       = fused_scores[keep]
    final_labels       = fused_labels[keep]  # giữ lại nhưng không dùng ở đây
    final_count        = len(final_boxes_norm)

    # Scale về pixel
    final_boxes_pixel = final_boxes_norm * np.array([w, h, w, h])

    return final_count, final_boxes_pixel, final_scores