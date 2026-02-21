import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import torch
from ultralytics import YOLO
from config.inference_config import InferenceConfig

def apply_clahe_on_lab(img_bgr, clip_limit=3.0, tile_size=8):
    """
    Áp dụng CLAHE trên kênh L của không gian màu LAB.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def horizontal_flip(img):
    """Lật ngang ảnh."""
    return cv2.flip(img, 1)

def revert_horizontal_flip_boxes(boxes, width):
    """Đưa boxes từ ảnh flip ngang về ảnh gốc."""
    boxes_reverted = boxes.copy()
    boxes_reverted[:, [0, 2]] = width - boxes[:, [2, 0]]
    return boxes_reverted

def draw_detections_on_image(img, boxes_pixel, scores):
    """
    Vẽ bounding box, tâm và confidence score lên ảnh.
    - boxes_pixel: numpy array [N, 4] (x1,y1,x2,y2) ở pixel scale
    - scores: numpy array [N] confidence scores
    Trả về ảnh đã vẽ (copy của ảnh gốc)
    """
    drawn_img = img.copy()
    if len(boxes_pixel) == 0:
        return drawn_img

    for i, box in enumerate(boxes_pixel):
        x1, y1, x2, y2 = map(int, box)
        conf = scores[i]

        # Vẽ rectangle (màu xanh lá)
        cv2.rectangle(drawn_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Vẽ tâm (màu đỏ)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(drawn_img, (cx, cy), radius=3, color=(0, 0, 255), thickness=-1)

        # Ghi confidence score (màu vàng, nhỏ, phía trên box nếu được)
        text = f"{conf:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(drawn_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return drawn_img

