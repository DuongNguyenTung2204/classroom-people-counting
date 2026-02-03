from dataclasses import dataclass, field
from typing import List

@dataclass
class InferenceConfig:
    model1_path: str = r"ckpts\yolo26x_p2_v0_800.pt"
    model2_path: str = r"ckpts\yolo12x_CLAHE_v0.pt"

    imgsz: int = 768

    conf_model1: float = 0.1
    conf_model1_flip: float = 0.15
    conf_model2: float = 0.075

    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 16

    wbf_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.9])
    wbf_iou_thr: float = 0.5
    wbf_skip_box_thr: float = 0.05
    wbf_conf_type: str = 'absent_model_aware_avg' # avg, max, box_and_model_avg, absent_model_aware_avg

    final_score_thr: float = 0.2
