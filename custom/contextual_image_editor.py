from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any
import copy
import json
import optuna
import cv2
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter
from rembg import remove
from torchvision import transforms as T
from transformers import (
    AutoModelForImageSegmentation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    DepthProForDepthEstimation,
    DepthProImageProcessorFast,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

try:
    from transformers import Sam2Model, Sam2Processor  # type: ignore
    _SAM_AVAILABLE = True
except ImportError:
    Sam2Model = None  # type: ignore
    Sam2Processor = None  # type: ignore
    _SAM_AVAILABLE = False

try:
    from diffusers import MarigoldNormalsPipeline, MarigoldIntrinsicsPipeline  # type: ignore
    _MARIGOLD_AVAILABLE = True
except ImportError:
    MarigoldNormalsPipeline = None  # type: ignore
    MarigoldIntrinsicsPipeline = None  # type: ignore
    _MARIGOLD_AVAILABLE = False


_DET_PROCESSOR = None
_DET_MODEL = None
_SAM_PROCESSOR = None
_SAM_MODEL = None

_BIREFNET_MODEL = None
_DEPTH_PROCESSOR = None
_DEPTH_MODEL = None
_ONEFORMER_PROCESSOR = None
_ONEFORMER_MODEL = None
_NORMALS_PIPE = None
_INTRINSICS_PIPE = None


DEFAULT_CONFIG = {
    "models": {
        "detector_id": "IDEA-Research/grounding-dino-base",
        "sam_id": "facebook/sam2.1-hiera-large",
        "birefnet_id": "ZhengPeng7/BiRefNet",
        "depth_id": "apple/DepthPro-hf",
        "oneformer_id": "shi-labs/oneformer_ade20k_swin_large",
        "normals_id": "prs-eth/marigold-normals-v1-1",
        "intrinsics_id": "prs-eth/marigold-iid-lighting-v1-1",
    },
    "detection": {
        "object_threshold": 0.24,
        "object_text_threshold": 0.18,
        "support_threshold": 0.22,
        "support_text_threshold": 0.18,
        "context_threshold": 0.22,
        "context_text_threshold": 0.18,
        "max_side": 1280,
    },
    "segmentation": {
        "use_birefnet": True,
        "birefnet_size": 1024,
        "birefnet_threshold": 0.35,
        "use_oneformer": True,
        "oneformer_task": "panoptic",
        "support_label_keywords": [
            "table", "desk", "counter", "countertop", "shelf", "cabinet", "island",
            "tray", "stand", "bar", "kitchen"
        ],
        "avoid_label_keywords": [
            "person", "hand", "arm", "bowl", "plate", "cup", "mug", "glass", "bottle", "jar",
            "vase", "plant", "basket", "box", "fruit", "food", "knife", "fork", "spoon",
            "sink", "stove", "toaster", "microwave", "kettle", "coffee", "appliance",
            "phone", "tablet", "laptop", "bag"
        ],
        "min_segment_area_ratio": 0.002,
    },
    "extraction": {
        "use_rembg_fallback": True,
        "rembg_alpha_threshold": 12,
        "sam_box_padding": 18,
        "matte_edge_blur": 0.65,
        "grabcut_refine": True,
        "grabcut_iters": 1,
        "combine_mode": "auto",
        "dark_bg_fallback": True,
        "erode_px": 0,
        "dilate_px": 1,
        "reference_box_pad_ratio": 0.18,
        "reference_box_min_pad_px": 24,
        "component_max_gap_px": 96,
        "component_min_area_ratio": 0.0012,
        "max_border_touch_ratio": 0.22,
        "reference_prompts": [
            "food item",
            "ingredient cluster",
            "fruit with garnish",
            "fruit with leaves",
            "produce",
            "ingredient",
            "object",
        ],
    },
    "geometry": {
        "use_normals": True,
        "use_intrinsics": True,
        "normals_steps": 2,
        "intrinsics_steps": 2,
        "plane_normal_min_up": 0.35,
    },
    "placement": {
        "max_supports_to_try": 8,
        "candidate_step_x_divisor": 7,
        "attempt_index": 0,
        "top_k_to_keep": 16,
        "edge_margin_ratio": 0.05,
        "min_scale_ratio": 0.72,
        "max_scale_ratio": 1.34,
        "avoid_overlap_weight": 6.0,
        "total_overlap_weight": 2.0,
        "depth_std_weight": 1.6,
        "center_offset_weight": 0.30,
        "support_band_weight": 1.3,
        "perspective_weight": 1.8,
        "favor_empty_space_weight": 1.0,
        "size_consistency_weight": 2.0,
        "support_mask_penalty_weight": 5.0,
        "support_depth_mismatch_weight": 2.0,
        "normal_penalty_weight": 2.5,
        "default_object_width_ratio": 0.11,
        "min_object_width_px": 24,
        "fallback_global_search": True,
        "fallback_support_margin_ratio": 0.08,
    },
    "support_geometry": {
        "thin_height_ratio": 0.03,
        "plane_min_height_ratio": 0.05,
        "edge_depth_slope_max": 0.03,
        "plane_depth_slope_min": 0.035,
        "plane_depth_variance_min": 0.012,
        "plane_back_start_ratio": 0.08,
        "plane_front_end_ratio": 0.72,
        "edge_contact_offset_px": 1,
        "plane_candidate_step_y_divisor": 7,
        "surface_mask_expand_px": 18,
        "surface_mask_blur_px": 7,
        "surface_valid_threshold": 0.28,
        "min_support_coverage": 0.10,
    },
    "support_preferences": {
        "preferred_labels": ["table", "desk", "island", "countertop", "counter", "kitchen counter"],
        "disfavored_labels": ["shelf"],
        "prefer_mode": "plane",
        "label_match_bonus": 2.5,
        "mode_match_bonus": 1.5,
        "disfavored_label_penalty": 2.0,
    },
    "collision": {
        "enabled": True,
        "max_iou": 0.01,
        "max_intersection_ratio_of_candidate": 0.02,
        "use_occupancy_map": True,
        "occupancy_blur_px": 9,
        "occupancy_threshold": 0.20,
        "occupancy_penalty_weight": 10.0,
        "hard_occupancy_reject": False,
    },
    "lighting": {
        "enabled": True,
        "max_sources": 3,
        "bright_quantile": 0.985,
        "min_component_area_ratio": 0.0005,
        "prefer_upper_half": 1.35,
        "prefer_border_regions": 1.2,
        "window_boost": 1.35,
        "specular_boost": 1.15,
        "ambient_floor": 0.18,
        "use_depth_for_elevation": True,
    },
    "shadow": {
        "enabled": True,
        "contact_opacity": 0.28,
        "contact_blur_px": 2.4,
        "cast_opacity": 0.14,
        "cast_blur_px": 8.0,
        "cast_length_scale": 0.66,
        "squash_ratio": 0.34,
        "shear_strength": 0.22,
        "shadow_color_mode": "surface_tinted",
        "ambient_occlusion_band_ratio": 0.08,
        "shadow_softness_influence": 0.35,
    },
    "relighting": {
        "enabled": True,
        "mean_match_strength": 0.50,
        "std_match_strength": 0.35,
        "color_match_strength": 0.25,
        "saturation_match_strength": 0.12,
        "directional_shading_strength": 0.18,
        "bottom_occlusion_strength": 0.10,
        "highlight_strength": 0.05,
        "use_intrinsics_shading": True,
    },
    "occlusion": {
        "enabled": True,
        "feather_px": 2.0,
        "depth_bias": 0.03,
        "foreground_hardness": 0.70,
        "object_depth_top_offset": 0.08,
        "prefer_segment_occluders": True,
    },
    "output": {
        "timestamp_outputs": True,
        "save_candidate_overlay": False,
        "save_support_mask": False,
    },
}


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    label: str = ""

    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def area(self) -> float:
        return self.width() * self.height()

    def centre(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def clamp(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x0=max(0.0, min(self.x0, width)),
            y0=max(0.0, min(self.y0, height)),
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            score=self.score,
            label=self.label,
        )

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1))


@dataclass
class ExtractedObject:
    rgba: Image.Image
    mask: Image.Image
    bbox: BoundingBox
    label: str


@dataclass
class LightSource:
    x: float
    y: float
    strength: float
    kind: str = "heuristic"


@dataclass
class Placement:
    x: int
    y: int
    width: int
    height: int
    support_box: Optional[BoundingBox]


@dataclass
class PlacementCandidate:
    placement: Placement
    score: float
    debug: str = ""


@dataclass
class SupportGeometry:
    box: BoundingBox
    mode: str
    plane_y_min: int
    plane_y_max: int
    depth_slope: float
    depth_variance: float
    score: float
    surface_mask: Optional[np.ndarray] = None
    normal_score: float = 0.0


SURFACE_LABELS = ["countertop", "kitchen counter", "table", "desk", "shelf", "island", "tray", "counter"]
CONTEXT_OBJECT_LABELS: List[str] = [
    "fruit bowl", "bowl", "plate", "cup", "mug", "glass", "bottle", "jar", "vase",
    "utensil", "fork", "knife", "spoon", "napkin", "tray", "cutting board", "board",
    "sink", "stove", "cabinet", "appliance", "toaster", "kettle", "coffee maker",
    "microwave", "container", "basket", "book", "box", "plant", "flower pot", "lamp",
    "phone", "tablet", "laptop", "bag", "food", "fruit", "bread", "banana", "apple",
    "orange", "lemon", "person", "hand",
]


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str]) -> dict:
    base = copy.deepcopy(DEFAULT_CONFIG)
    if not path:
        return base
    config_path = Path(path)
    if not config_path.exists():
        return base
    with config_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(base, user_cfg)

def model_dtype_for_device(device: torch.device):
    return torch.float16 if device.type == "cuda" else torch.float32


def get_detector(device: torch.device, cfg: dict):
    global _DET_PROCESSOR, _DET_MODEL
    if _DET_PROCESSOR is None or _DET_MODEL is None:
        model_id = cfg["models"]["detector_id"]
        _DET_PROCESSOR = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
        _DET_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, token=HF_TOKEN)
        _DET_MODEL.to(device)
        _DET_MODEL.eval()
    return _DET_PROCESSOR, _DET_MODEL


def get_sam(device: torch.device, cfg: dict):
    global _SAM_PROCESSOR, _SAM_MODEL
    if not _SAM_AVAILABLE:
        return None, None
    if _SAM_PROCESSOR is None or _SAM_MODEL is None:
        model_id = cfg["models"]["sam_id"]
        _SAM_PROCESSOR = Sam2Processor.from_pretrained(model_id, token=HF_TOKEN)
        _SAM_MODEL = Sam2Model.from_pretrained(model_id, token=HF_TOKEN)
        _SAM_MODEL.to(device)
        _SAM_MODEL.eval()
    return _SAM_PROCESSOR, _SAM_MODEL


def get_birefnet(device: torch.device, cfg: dict):
    global _BIREFNET_MODEL
    if _BIREFNET_MODEL is None:
        model_id = cfg["models"]["birefnet_id"]
        _BIREFNET_MODEL = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        _BIREFNET_MODEL.to(device)
        _BIREFNET_MODEL.eval()
    return _BIREFNET_MODEL


def get_depth_model(device: torch.device, cfg: dict):
    global _DEPTH_PROCESSOR, _DEPTH_MODEL
    if _DEPTH_PROCESSOR is None or _DEPTH_MODEL is None:
        model_id = cfg["models"]["depth_id"]
        _DEPTH_PROCESSOR = DepthProImageProcessorFast.from_pretrained(model_id, token=HF_TOKEN)
        _DEPTH_MODEL = DepthProForDepthEstimation.from_pretrained(
            model_id,
            torch_dtype=model_dtype_for_device(device),
            token=HF_TOKEN,
        )
        _DEPTH_MODEL.to(device)
        _DEPTH_MODEL.eval()
    return _DEPTH_PROCESSOR, _DEPTH_MODEL


def get_oneformer(device: torch.device, cfg: dict):
    global _ONEFORMER_PROCESSOR, _ONEFORMER_MODEL
    if _ONEFORMER_PROCESSOR is None or _ONEFORMER_MODEL is None:
        model_id = cfg["models"]["oneformer_id"]
        _ONEFORMER_PROCESSOR = OneFormerProcessor.from_pretrained(model_id, token=HF_TOKEN)
        _ONEFORMER_MODEL = OneFormerForUniversalSegmentation.from_pretrained(model_id, token=HF_TOKEN)
        _ONEFORMER_MODEL.to(device)
        _ONEFORMER_MODEL.eval()
    return _ONEFORMER_PROCESSOR, _ONEFORMER_MODEL


def get_normals_pipe(device: torch.device, cfg: dict):
    global _NORMALS_PIPE
    if not _MARIGOLD_AVAILABLE or not cfg["geometry"].get("use_normals", True):
        return None
    if _NORMALS_PIPE is None:
        model_id = cfg["models"]["normals_id"]
        kwargs = {}
        if device.type == "cuda":
            kwargs["variant"] = "fp16"
            kwargs["torch_dtype"] = torch.float16
        _NORMALS_PIPE = MarigoldNormalsPipeline.from_pretrained(model_id, **kwargs)
        _NORMALS_PIPE.to(device)
    return _NORMALS_PIPE


def get_intrinsics_pipe(device: torch.device, cfg: dict):
    global _INTRINSICS_PIPE
    if not _MARIGOLD_AVAILABLE or not cfg["geometry"].get("use_intrinsics", True):
        return None
    if _INTRINSICS_PIPE is None:
        model_id = cfg["models"]["intrinsics_id"]
        kwargs = {}
        if device.type == "cuda":
            kwargs["variant"] = "fp16"
            kwargs["torch_dtype"] = torch.float16
        _INTRINSICS_PIPE = MarigoldIntrinsicsPipeline.from_pretrained(model_id, **kwargs)
        _INTRINSICS_PIPE.to(device)
    return _INTRINSICS_PIPE


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def maybe_resize_for_detection(image: Image.Image, max_side: int = 1024) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale == 1.0:
        return image, 1.0
    new_w = max(32, int(round(w * scale / 8) * 8))
    new_h = max(32, int(round(h * scale / 8) * 8))
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def scale_box(box: BoundingBox, inv_scale: float) -> BoundingBox:
    return BoundingBox(box.x0 * inv_scale, box.y0 * inv_scale, box.x1 * inv_scale, box.y1 * inv_scale, score=box.score, label=box.label)


def detect_objects(
    image: Image.Image,
    labels: Iterable[str],
    device: torch.device,
    cfg: dict,
    threshold: float,
    text_threshold: float,
    max_side: Optional[int] = None,
) -> List[BoundingBox]:
    labels = [label.strip() for label in labels if label.strip()]
    if not labels:
        return []
    resized, scale = maybe_resize_for_detection(image, max_side=max_side or cfg["detection"]["max_side"])
    inv_scale = 1.0 / scale
    processor, model = get_detector(device, cfg)
    text_labels = [labels]
    inputs = processor(images=resized, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[resized.size[::-1]],
    )
    detections: List[BoundingBox] = []
    if not results:
        return detections
    result = results[0]
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels_out = result.get("labels", [])
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        score = float(scores[i].item()) if i < len(scores) else 1.0
        label = str(labels_out[i]) if i < len(labels_out) else ""
        detections.append(scale_box(BoundingBox(x0, y0, x1, y1, score=score, label=label), inv_scale).clamp(*image.size))
    return detections


def _connected_components_from_binary(mask: np.ndarray) -> List[tuple[int, int, int, int, int]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        comps.append((x, y, w, h, area))
    return comps


def detect_light_sources(scene_image: Image.Image, depth_map: Optional[np.ndarray], cfg: dict) -> List[LightSource]:
    arr = np.array(scene_image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    thresh = float(np.quantile(v, float(cfg["lighting"].get("bright_quantile", 0.985))))
    bright = (v >= thresh).astype(np.uint8) * 255
    specular_like = ((v > max(0.88, thresh - 0.04)) & (s < 0.35)).astype(np.uint8) * 255
    bright = np.maximum(bright, specular_like)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    min_area = max(8, int(round(h * w * float(cfg["lighting"].get("min_component_area_ratio", 0.0005)))))
    comps = _connected_components_from_binary(bright)
    sources: List[LightSource] = []
    for x, y, cw, ch, area in comps:
        if area < min_area:
            continue
        x0, y0, x1, y1 = x, y, x + cw, y + ch
        patch_v = v[y0:y1, x0:x1]
        patch_s = s[y0:y1, x0:x1]
        if patch_v.size == 0:
            continue
        cx = x0 + cw / 2.0
        cy = y0 + ch / 2.0
        strength = float(np.mean(patch_v)) * np.sqrt(area)
        if cy < h * 0.5:
            strength *= float(cfg["lighting"].get("prefer_upper_half", 1.35))
        border_margin_x = w * 0.12
        border_margin_y = h * 0.12
        if (cx < border_margin_x) or (cx > w - border_margin_x) or (cy < border_margin_y):
            strength *= float(cfg["lighting"].get("prefer_border_regions", 1.2))
        low_sat_ratio = float(np.mean(patch_s < 0.30))
        if low_sat_ratio > 0.55 and cw * ch > 0.01 * w * h:
            strength *= float(cfg["lighting"].get("window_boost", 1.35))
        if low_sat_ratio > 0.45 and area < 0.01 * w * h:
            strength *= float(cfg["lighting"].get("specular_boost", 1.15))
        if depth_map is not None and cfg["lighting"].get("use_depth_for_elevation", True):
            patch_d = depth_map[y0:y1, x0:x1]
            if patch_d.size > 0:
                d = float(np.median(patch_d))
                strength *= float(np.interp(d, [0.0, 1.0], [0.95, 1.15]))
        sources.append(LightSource(x=cx, y=cy, strength=strength))
    sources.sort(key=lambda s: s.strength, reverse=True)
    return sources[: int(cfg["lighting"].get("max_sources", 3))]


def estimate_light_vector_from_sources(placement: Placement, scene_size: Tuple[int, int], light_sources: List[LightSource], cfg: dict) -> Tuple[float, float, float]:
    scene_w, _ = scene_size
    obj_cx = placement.x + placement.width * 0.5
    obj_base_y = placement.y + placement.height * 0.92
    if not light_sources:
        return 0.35, 0.94, float(cfg["lighting"].get("ambient_floor", 0.18))
    vx_total = 0.0
    vy_total = 0.0
    strength_total = 0.0
    for src in light_sources:
        dx = obj_cx - src.x
        dy = obj_base_y - src.y
        norm = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        vx = dx / norm
        vy = max(0.05, dy / norm)
        weight = src.strength * (1.15 if src.y < obj_base_y else 0.8)
        vx_total += vx * weight
        vy_total += vy * weight
        strength_total += weight
    norm = max(1e-6, (vx_total * vx_total + vy_total * vy_total) ** 0.5)
    ambient = float(cfg["lighting"].get("ambient_floor", 0.18))
    direct_strength = min(1.0, strength_total / max(1.0, scene_w * 0.02))
    return vx_total / norm, vy_total / norm, max(ambient, direct_strength)


def choose_best_detection(boxes: List[BoundingBox]) -> Optional[BoundingBox]:
    return max(boxes, key=lambda b: (b.score, b.area())) if boxes else None


def crop_to_alpha_bbox(rgba: Image.Image) -> tuple[Image.Image, Image.Image, BoundingBox]:
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return rgba, alpha, BoundingBox(0, 0, rgba.width, rgba.height)
    x0, y0, x1, y1 = bbox
    pad = 2
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgba.width, x1 + pad)
    y1 = min(rgba.height, y1 + pad)
    cropped_rgba = rgba.crop((x0, y0, x1, y1))
    cropped_alpha = alpha.crop((x0, y0, x1, y1))
    return cropped_rgba, cropped_alpha, BoundingBox(float(x0), float(y0), float(x1), float(y1))


def apply_morph(mask: np.ndarray, erode_px: int = 0, dilate_px: int = 0) -> np.ndarray:
    out = mask.copy()
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        out = cv2.erode(out, k, iterations=1)
    if dilate_px > 0:
        k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        out = cv2.dilate(out, k, iterations=1)
    return out


def clean_extracted_object(rgba: Image.Image, cfg: dict) -> Image.Image:
    arr = np.array(rgba).copy()
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    alpha[alpha < 8] = 0
    alpha = apply_morph(alpha, int(cfg["extraction"].get("erode_px", 0)), int(cfg["extraction"].get("dilate_px", 0)))
    alpha_img = Image.fromarray(alpha, mode="L")
    blur = float(cfg["extraction"].get("matte_edge_blur", 0.0))
    if blur > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur))
    alpha = np.array(alpha_img, dtype=np.uint8)
    alpha = np.where(alpha >= 32, alpha, 0).astype(np.uint8)
    rgb[alpha < 6] = 0
    arr[:, :, :3] = rgb
    arr[:, :, 3] = alpha
    cleaned = Image.fromarray(arr, mode="RGBA")
    cropped_rgba, cropped_alpha, _ = crop_to_alpha_bbox(cleaned)
    cropped_rgba.putalpha(cropped_alpha)
    return cropped_rgba


def mask_from_rembg(image: Image.Image, alpha_threshold: int = 20) -> Optional[Image.Image]:
    try:
        buf = io.BytesIO()
        image.convert("RGBA").save(buf, format="PNG")
        buf.seek(0)
        out_bytes = remove(buf.getvalue())
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        alpha = np.array(out_img.getchannel("A"), dtype=np.uint8)
        alpha = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(alpha, mode="L")
    except Exception:
        return None


def mask_from_birefnet(image: Image.Image, device: torch.device, cfg: dict) -> Optional[Image.Image]:
    if not cfg["segmentation"].get("use_birefnet", True):
        return None
    try:
        model = get_birefnet(device, cfg)
        image_size = int(cfg["segmentation"].get("birefnet_size", 1024))
        tfm = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(x)
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]
        elif hasattr(preds, "logits"):
            pred = preds.logits
        else:
            pred = preds[-1]
        pred = pred.sigmoid().float().cpu()[0]
        if pred.ndim == 3:
            pred = pred.squeeze(0)
        pred = cv2.resize(pred.numpy(), image.size, interpolation=cv2.INTER_LINEAR)
        pred = np.clip(pred, 0.0, 1.0)
        alpha = (pred * 255.0).astype(np.uint8)
        thresh = float(cfg["segmentation"].get("birefnet_threshold", 0.35))
        alpha = np.where(alpha >= int(thresh * 255), 255, 0).astype(np.uint8)
        return Image.fromarray(alpha, mode="L")
    except Exception:
        return None


def refine_mask_with_sam(image: Image.Image, box: BoundingBox, device: torch.device, cfg: dict) -> Optional[Image.Image]:
    sam_processor, sam_model = get_sam(device, cfg)
    if sam_processor is None or sam_model is None:
        return None
    pad = int(cfg["extraction"].get("sam_box_padding", 18))
    try:
        input_boxes = [[[max(0.0, box.x0 - pad), max(0.0, box.y0 - pad), min(image.width, box.x1 + pad), min(image.height, box.y1 + pad)]]]
        sam_inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = sam_model(**sam_inputs)
        masks = sam_processor.post_process_masks(outputs.pred_masks.cpu(), sam_inputs["original_sizes"], sam_inputs["reshaped_input_sizes"])
        if not masks or len(masks[0]) == 0:
            return None
        best_mask = None
        best_score = -1.0
        for m in masks[0][0]:
            mask_np = (m.numpy() > 0).astype(np.uint8) * 255
            ys, xs = np.where(mask_np > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            mask_area = float((mask_np > 0).sum())
            bbox_area = float(max(1, (x1 - x0 + 1) * (y1 - y0 + 1)))
            fill_ratio = mask_area / bbox_area
            border_touch = float(((mask_np[0, :] > 0).sum() + (mask_np[-1, :] > 0).sum() + (mask_np[:, 0] > 0).sum() + (mask_np[:, -1] > 0).sum())) / max(1.0, (2 * mask_np.shape[0] + 2 * mask_np.shape[1]))
            score = mask_area * fill_ratio * (1.0 - 0.6 * border_touch)
            if 0.02 <= fill_ratio <= 0.995 and score > best_score:
                best_score = score
                best_mask = mask_np
        if best_mask is None:
            best_mask = (masks[0][0][0].numpy() > 0).astype(np.uint8) * 255
        return Image.fromarray(best_mask, mode="L")
    except Exception:
        return None


def extract_dark_bg_foreground(image: Image.Image, box: BoundingBox) -> Image.Image:
    x0, y0, x1, y1 = box.to_int_tuple()
    crop = image.crop((x0, y0, x1, y1)).convert("RGBA")
    arr = np.array(crop).copy()
    rgb = arr[:, :, :3].astype(np.int16)
    bg = (rgb[:, :, 0] < 40) & (rgb[:, :, 1] < 40) & (rgb[:, :, 2] < 40)
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    bg |= ((channel_max < 55) & ((channel_max - channel_min) < 18))
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    alpha = np.array(alpha_img, dtype=np.uint8)
    arr[:, :, 3] = alpha
    arr[alpha == 0, 0:3] = 0
    out = Image.fromarray(arr, mode="RGBA")
    out, _, _ = crop_to_alpha_bbox(out)
    return out


def intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def candidate_intersection_ratio(candidate: BoundingBox, other: BoundingBox) -> float:
    return intersection_area(candidate, other) / max(1.0, candidate.area())


def build_occupancy_map(image_size: Tuple[int, int], occupied_boxes: List[BoundingBox], blur_px: int = 9) -> np.ndarray:
    w, h = image_size
    occ = np.zeros((h, w), dtype=np.float32)
    for box in occupied_boxes:
        x0 = max(0, min(w, int(round(box.x0))))
        y0 = max(0, min(h, int(round(box.y0))))
        x1 = max(x0 + 1, min(w, int(round(box.x1))))
        y1 = max(y0 + 1, min(h, int(round(box.y1))))
        occ[y0:y1, x0:x1] = 1.0
    if blur_px > 0:
        occ = cv2.GaussianBlur(occ, (0, 0), sigmaX=float(blur_px), sigmaY=float(blur_px))
    return np.clip(occ, 0.0, 1.0)


def occupancy_score_for_box(occupancy_map: np.ndarray, box: BoundingBox) -> float:
    h, w = occupancy_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0))))
    y0 = max(0, min(h - 1, int(round(box.y0))))
    x1 = max(x0 + 1, min(w, int(round(box.x1))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))
    patch = occupancy_map[y0:y1, x0:x1]
    return 0.0 if patch.size == 0 else float(np.mean(patch))


def union_boxes(boxes: List[BoundingBox], image_size: Tuple[int, int]) -> Optional[BoundingBox]:
    if not boxes:
        return None
    x0 = min(b.x0 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    score = max(b.score for b in boxes)
    label = max(boxes, key=lambda b: b.score).label
    return BoundingBox(x0, y0, x1, y1, score=score, label=label).clamp(*image_size)


def expand_box(box: BoundingBox, image_size: Tuple[int, int], pad_ratio: float, min_pad_px: int) -> BoundingBox:
    pad = max(min_pad_px, int(round(max(box.width(), box.height()) * pad_ratio)))
    return BoundingBox(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad, score=box.score, label=box.label).clamp(*image_size)


def propose_reference_box(reference_image: Image.Image, device: torch.device, cfg: dict) -> BoundingBox:
    detections = detect_objects(
        reference_image,
        labels=list(cfg["extraction"].get("reference_prompts", [])),
        device=device,
        cfg=cfg,
        threshold=cfg["detection"]["object_threshold"],
        text_threshold=cfg["detection"]["object_text_threshold"],
        max_side=min(960, cfg["detection"]["max_side"]),
    )
    if not detections:
        return BoundingBox(0, 0, reference_image.width, reference_image.height, label="reference")
    best = choose_best_detection(detections)
    high_conf = [d for d in detections if d.score >= max(0.18, best.score * 0.55)]
    roi = union_boxes(high_conf, reference_image.size) or best
    return expand_box(roi, reference_image.size, float(cfg["extraction"].get("reference_box_pad_ratio", 0.18)), int(cfg["extraction"].get("reference_box_min_pad_px", 24)))


def crop_with_box(image: Image.Image, box: BoundingBox) -> tuple[Image.Image, Tuple[int, int]]:
    x0, y0, x1, y1 = box.to_int_tuple()
    return image.crop((x0, y0, x1, y1)), (x0, y0)


def _mask_stats(mask: np.ndarray) -> dict:
    h, w = mask.shape[:2]
    area = float((mask > 0).sum())
    area_ratio = area / max(1.0, h * w)
    if area == 0:
        return {"area_ratio": 0.0, "border_touch": 1.0, "num_components": 0}
    border_touch = float(((mask[0, :] > 0).sum() + (mask[-1, :] > 0).sum() + (mask[:, 0] > 0).sum() + (mask[:, -1] > 0).sum())) / max(1.0, 2 * h + 2 * w)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    num_components = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 8:
            num_components += 1
    return {"area_ratio": area_ratio, "border_touch": border_touch, "num_components": num_components}


def combine_candidate_masks(primary_mask: Optional[Image.Image], sam_mask: Optional[Image.Image], mode: str = "auto", cfg: Optional[dict] = None) -> Optional[Image.Image]:
    if primary_mask is None and sam_mask is None:
        return None
    if primary_mask is None:
        return sam_mask
    if sam_mask is None:
        return primary_mask
    a = np.array(primary_mask, dtype=np.uint8) > 0
    b = np.array(sam_mask, dtype=np.uint8) > 0
    if mode == "intersection":
        out = a & b
    elif mode == "union":
        out = a | b
    elif mode == "primary_prefer":
        support = cv2.dilate(a.astype(np.uint8) * 255, np.ones((9, 9), np.uint8), iterations=1) > 0
        out = a | (b & support)
    elif mode == "sam_prefer":
        support = cv2.dilate(b.astype(np.uint8) * 255, np.ones((9, 9), np.uint8), iterations=1) > 0
        out = b | (a & support)
    else:
        sa = _mask_stats(a.astype(np.uint8) * 255)
        sb = _mask_stats(b.astype(np.uint8) * 255)
        border_limit = float(cfg["extraction"].get("max_border_touch_ratio", 0.22)) if cfg else 0.22
        good_a = sa["area_ratio"] > 0.01 and sa["border_touch"] <= border_limit
        good_b = sb["area_ratio"] > 0.01 and sb["border_touch"] <= border_limit
        inter = a & b
        if good_a and good_b:
            out = inter if inter.sum() > 0.65 * min(max(1, a.sum()), max(1, b.sum())) else (a | (b & (cv2.dilate(a.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=1) > 0)))
        elif good_a:
            out = a
        elif good_b:
            out = b
        else:
            out = a | b
    return Image.fromarray(out.astype(np.uint8) * 255, mode="L")


def _component_boxes_and_labels(mask: np.ndarray):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        comps.append((i, BoundingBox(x, y, x + w, y + h), area))
    return labels, comps


def _box_gap(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
    dy = max(0.0, max(a.y0 - b.y1, b.y0 - a.y1))
    return float((dx * dx + dy * dy) ** 0.5)


def find_seed_component(mask: Image.Image) -> BoundingBox:
    mask_np = np.array(mask, dtype=np.uint8)
    _, comps = _component_boxes_and_labels(mask_np)
    if not comps:
        return BoundingBox(0, 0, mask.width, mask.height)
    cx = mask.width * 0.5
    cy = mask.height * 0.5
    best = None
    best_score = -1e18
    for _, box, area in comps:
        bx, by = box.centre()
        center_dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        score = area - center_dist * 2.0
        if score > best_score:
            best_score = score
            best = box
    return best or BoundingBox(0, 0, mask.width, mask.height)


def group_subject_components(mask: Image.Image, seed_box: BoundingBox, cfg: dict) -> Image.Image:
    mask_np = np.array(mask, dtype=np.uint8)
    labels, comps = _component_boxes_and_labels(mask_np)
    if not comps:
        return mask
    img_area = float(mask_np.shape[0] * mask_np.shape[1])
    min_area = max(8.0, img_area * float(cfg["extraction"].get("component_min_area_ratio", 0.0012)))
    max_gap = float(cfg["extraction"].get("component_max_gap_px", 96))
    seed_idx = None
    seed_score = -1.0
    for i, box, area in comps:
        inter = intersection_area(seed_box, box)
        center_dist = ((box.centre()[0] - seed_box.centre()[0]) ** 2 + (box.centre()[1] - seed_box.centre()[1]) ** 2) ** 0.5
        score = inter * 2.0 + area - center_dist * 4.0
        if score > seed_score:
            seed_score = score
            seed_idx = i
    keep = set()
    frontier = []
    if seed_idx is not None:
        keep.add(seed_idx)
        frontier.append(seed_idx)
    boxes_by_idx = {i: (box, area) for i, box, area in comps}
    while frontier:
        current = frontier.pop()
        cur_box, _ = boxes_by_idx[current]
        for idx, (box, area) in boxes_by_idx.items():
            if idx in keep or area < min_area:
                continue
            if _box_gap(cur_box, box) <= max_gap:
                keep.add(idx)
                frontier.append(idx)
    out = np.zeros_like(mask_np, dtype=np.uint8)
    for idx in keep:
        out[labels == idx] = 255
    return Image.fromarray(out, mode="L") if out.sum() > 0 else mask


def refine_mask_with_grabcut(image: Image.Image, mask: Image.Image, iters: int = 1) -> Image.Image:
    rgb = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    m = np.array(mask, dtype=np.uint8)
    if m.max() == 0:
        return mask
    gc_mask = np.full(m.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[m > 0] = cv2.GC_PR_FGD
    sure_fg = cv2.erode((m > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    sure_bg = cv2.dilate((m == 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    gc_mask[sure_bg > 0] = cv2.GC_BGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, gc_mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)
        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return Image.fromarray(out, mode="L")
    except Exception:
        return mask


def extract_reference_object(reference_image: Image.Image, device: torch.device, cfg: dict) -> ExtractedObject:
    reference_box = propose_reference_box(reference_image, device, cfg)
    ref_crop, _ = crop_with_box(reference_image, reference_box)

    primary_local = mask_from_birefnet(ref_crop, device, cfg)
    if primary_local is None and cfg["extraction"].get("use_rembg_fallback", True):
        primary_local = mask_from_rembg(
            ref_crop,
            int(cfg["extraction"].get("rembg_alpha_threshold", 12)),
        )

    sam_full = refine_mask_with_sam(reference_image, reference_box, device, cfg)
    sam_local = None
    if sam_full is not None:
        x0, y0, x1, y1 = reference_box.to_int_tuple()
        sam_local = sam_full.crop((x0, y0, x1, y1))

    combine_mode = str(cfg["extraction"].get("combine_mode", "primary_prefer"))
    merged_local = combine_candidate_masks(primary_local, sam_local, combine_mode, cfg)

    if merged_local is None and cfg["extraction"].get("dark_bg_fallback", True):
        cleaned_rgba = clean_extracted_object(
            extract_dark_bg_foreground(reference_image, reference_box),
            cfg,
        )
        cleaned_mask = cleaned_rgba.getchannel("A")
        return ExtractedObject(
            cleaned_rgba,
            cleaned_mask,
            BoundingBox(0, 0, cleaned_rgba.width, cleaned_rgba.height),
            "reference",
        )

    if merged_local is None:
        raise RuntimeError("Failed to extract the foreground subject from the reference image.")

    merged_np = np.array(merged_local, dtype=np.uint8)
    labels, comps = _component_boxes_and_labels(merged_np)
    if not comps:
        raise RuntimeError("Foreground mask was empty after extraction.")

    img_area = float(merged_np.shape[0] * merged_np.shape[1])
    min_area = max(10.0, img_area * float(cfg["extraction"].get("component_min_area_ratio", 0.0012)))
    max_gap = float(cfg["extraction"].get("component_max_gap_px", 96))

    # Pick the dominant subject component, but keep nearby companion components
    comps_sorted = sorted(comps, key=lambda t: t[2], reverse=True)
    seed_idx, seed_box, _ = comps_sorted[0]

    keep = set([seed_idx])

    # Keep components close to the seed OR overlapping an expanded seed region.
    expanded_seed = BoundingBox(
        seed_box.x0 - max_gap,
        seed_box.y0 - max_gap,
        seed_box.x1 + max_gap,
        seed_box.y1 + max_gap,
    )

    for idx, box, area in comps:
        if idx == seed_idx or area < min_area:
            continue
        gap = _box_gap(seed_box, box)
        inter = intersection_area(expanded_seed, box)
        if gap <= max_gap or inter > 0:
            keep.add(idx)

    # One extra pass: bridge nearby kept components so mint/leaves don't get dropped
    changed = True
    boxes_by_idx = {i: (box, area) for i, box, area in comps}
    while changed:
        changed = False
        kept_now = list(keep)
        for current in kept_now:
            cur_box, _ = boxes_by_idx[current]
            for idx, (box, area) in boxes_by_idx.items():
                if idx in keep or area < min_area:
                    continue
                if _box_gap(cur_box, box) <= max_gap * 0.75:
                    keep.add(idx)
                    changed = True

    grouped_np = np.zeros_like(merged_np, dtype=np.uint8)
    for idx in keep:
        grouped_np[labels == idx] = 255

    grouped_local = Image.fromarray(grouped_np, mode="L")

    if cfg["extraction"].get("grabcut_refine", False):
        grouped_local = refine_mask_with_grabcut(
            ref_crop,
            grouped_local,
            int(cfg["extraction"].get("grabcut_iters", 1)),
        )

    local_np = np.array(grouped_local, dtype=np.uint8)
    rgba_np = np.array(ref_crop.convert("RGBA")).copy()
    rgba_np[:, :, 3] = local_np
    rgba_np[local_np == 0, :3] = 0

    out = Image.fromarray(rgba_np, mode="RGBA")
    out = clean_extracted_object(out, cfg)
    cropped_rgba, cropped_mask, _ = crop_to_alpha_bbox(out)

    return ExtractedObject(
        cropped_rgba,
        cropped_mask,
        BoundingBox(0, 0, cropped_rgba.width, cropped_rgba.height),
        "reference",
    )


def estimate_depth_map(image: Image.Image, device: torch.device, cfg: dict) -> np.ndarray:
    processor, model = get_depth_model(device, cfg)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    depth = predicted_depth.squeeze().float().detach().cpu().numpy()
    depth = cv2.resize(depth, image.size, interpolation=cv2.INTER_CUBIC)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    depth = (depth - dmin) / (dmax - dmin)
    h = depth.shape[0]
    if np.median(depth[-max(1, h // 5):, :]) < np.median(depth[: max(1, h // 5), :]):
        depth = 1.0 - depth
    return depth.astype(np.float32)


def estimate_normals_map(image: Image.Image, device: torch.device, cfg: dict) -> Optional[np.ndarray]:
    pipe = get_normals_pipe(device, cfg)
    if pipe is None:
        return None
    try:
        out = pipe(image, num_inference_steps=int(cfg["geometry"].get("normals_steps", 2)))
        pred = out.prediction
        if isinstance(pred, list):
            pred = pred[0]
        if torch.is_tensor(pred):
            pred = pred.detach().float().cpu().numpy()
        pred = np.array(pred)
        if pred.ndim == 4:
            pred = pred[0]
        if pred.shape[0] == 3 and pred.ndim == 3:
            pred = np.transpose(pred, (1, 2, 0))
        if pred.shape[:2] != (image.height, image.width):
            pred = cv2.resize(pred, image.size, interpolation=cv2.INTER_LINEAR)
        return pred.astype(np.float32)
    except Exception:
        return None


def estimate_intrinsics_lighting(image: Image.Image, device: torch.device, cfg: dict) -> Optional[np.ndarray]:
    pipe = get_intrinsics_pipe(device, cfg)
    if pipe is None:
        return None
    try:
        out = pipe(image, num_inference_steps=int(cfg["geometry"].get("intrinsics_steps", 2)))
        pred = out.prediction
        if isinstance(pred, list):
            pred = pred[0]
        if isinstance(pred, dict):
            shade = pred.get("shading") or pred.get("diffuse_shading")
            if shade is None:
                return None
            if torch.is_tensor(shade):
                shade = shade.detach().float().cpu().numpy()
            shade = np.array(shade)
            if shade.ndim == 3 and shade.shape[0] in (1, 3):
                shade = np.transpose(shade, (1, 2, 0))
            if shade.ndim == 3:
                shade = shade.mean(axis=2)
            if shade.shape[:2] != (image.height, image.width):
                shade = cv2.resize(shade, image.size, interpolation=cv2.INTER_LINEAR)
            smin, smax = float(shade.min()), float(shade.max())
            if smax - smin > 1e-6:
                shade = (shade - smin) / (smax - smin)
            return shade.astype(np.float32)
    except Exception:
        return None
    return None


def depth_at_box_base(depth_map: np.ndarray, box: BoundingBox) -> float:
    h, w = depth_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0 + box.width() * 0.15))))
    x1 = max(x0 + 1, min(w, int(round(box.x1 - box.width() * 0.15))))
    y0 = max(0, min(h - 1, int(round(box.y1 - max(2.0, box.height() * 0.10)))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))
    patch = depth_map[y0:y1, x0:x1]
    return 0.5 if patch.size == 0 else float(np.median(patch))


def local_depth_stats(depth_map: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[float, float]:
    H, W = depth_map.shape
    x0 = max(0, min(W - 1, x))
    y0 = max(0, min(H - 1, y))
    x1 = max(x0 + 1, min(W, x + w))
    y1 = max(y0 + 1, min(H, y + h))
    patch = depth_map[y0:y1, x0:x1]
    return (0.5, 1.0) if patch.size == 0 else (float(np.median(patch)), float(np.std(patch)))


def choose_target_size(
    scene_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    support_box: Optional[BoundingBox],
    existing_same_objects: List[BoundingBox],
    depth_map: Optional[np.ndarray] = None,
    candidate_depth: Optional[float] = None,
    candidate_center: Optional[Tuple[float, float]] = None,
    cfg: Optional[dict] = None,
) -> tuple[int, int]:
    scene_w, scene_h = scene_size
    obj_w, obj_h = obj_size
    aspect = obj_w / max(1.0, obj_h)

    # Same-object matching, but keep the result conservative.
    if existing_same_objects and candidate_center is not None:
        cx, cy = candidate_center
        scored = []
        for b in existing_same_objects:
            bx, by = b.centre()
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            scored.append((dist, b.height()))
        scored.sort(key=lambda t: t[0])
        top = [h for _, h in scored[:3]]
        if top:
            target_h = float(np.median(top))
            if depth_map is not None and candidate_depth is not None:
                ref_depth = np.median(
                    [depth_at_box_base(depth_map, b) for b in existing_same_objects[: min(3, len(existing_same_objects))]]
                )
                ratio = float(candidate_depth / max(0.12, ref_depth))
                scale = float(
                    np.clip(
                        ratio ** 0.35,
                        cfg["placement"]["min_scale_ratio"],
                        cfg["placement"]["max_scale_ratio"],
                    )
                ) if cfg else float(np.clip(ratio ** 0.35, 0.85, 1.05))
                target_h *= scale

            target_h = int(round(np.clip(target_h, scene_h * 0.035, scene_h * 0.10)))
            target_w = int(round(target_h * aspect))
            target_w = min(target_w, int(scene_w * 0.075))
            return max(18, target_w), max(18, target_h)

    # Conservative support-based sizing.
    if support_box is not None:
        support_width_ratio = float(np.clip(support_box.width() / max(1.0, scene_w), 0.18, 0.75))
        support_height_ratio = float(np.clip(support_box.height() / max(1.0, scene_h), 0.02, 0.20))

        # Much smaller base size.
        target_w = scene_w * (0.028 + 0.028 * support_width_ratio)
        target_h = target_w / max(0.1, aspect)

        # Hard cap from support geometry.
        support_height_cap = max(18.0, support_box.height() * 0.65)

        # Global realism caps for a countertop lemon.
        global_w_cap = scene_w * 0.075
        global_h_cap = scene_h * 0.10

        if candidate_depth is not None:
            depth_scale = float(np.interp(candidate_depth, [0.0, 1.0], [0.88, 1.02]))
            target_w *= depth_scale
            target_h *= depth_scale

        target_h = min(target_h, support_height_cap, global_h_cap)
        target_w = min(target_w, global_w_cap, target_h * aspect)

        target_w = int(round(max(18.0, target_w)))
        target_h = int(round(max(18.0, target_h)))
        return target_w, target_h

    # Fallback sizing.
    target_w = int(round(scene_w * float(cfg["placement"].get("default_object_width_ratio", 0.06)) if cfg else scene_w * 0.06))
    target_w = min(target_w, int(scene_w * 0.075))
    target_h = int(round(target_w / max(0.1, aspect)))
    target_h = min(target_h, int(scene_h * 0.10))
    return max(18, target_w), max(18, target_h)

def support_preference_adjustment(support: SupportGeometry, cfg: dict) -> float:
    prefs = cfg.get("support_preferences", {})
    preferred_labels = [str(x).strip().lower() for x in prefs.get("preferred_labels", [])]
    disfavored_labels = [str(x).strip().lower() for x in prefs.get("disfavored_labels", [])]
    prefer_mode = str(prefs.get("prefer_mode", "any")).strip().lower()
    label = (support.box.label or "").strip().lower()
    adjustment = 0.0
    if any(p in label for p in preferred_labels):
        adjustment -= float(prefs.get("label_match_bonus", 2.5))
    if any(p in label for p in disfavored_labels):
        adjustment += float(prefs.get("disfavored_label_penalty", 2.0))
    if prefer_mode in {"plane", "edge"} and support.mode == prefer_mode:
        adjustment -= float(prefs.get("mode_match_bonus", 1.5))
    return adjustment


def filter_support_boxes(boxes: List[BoundingBox], image_size: Tuple[int, int]) -> List[BoundingBox]:
    w, h = image_size
    min_area = w * h * 0.012
    min_width = w * 0.18
    min_height = h * 0.02
    max_height = h * 0.22

    filtered = []
    for box in boxes:
        cy = box.centre()[1]
        width_ratio = box.width() / max(1.0, w)
        height_ratio = box.height() / max(1.0, h)

        if box.area() < min_area:
            continue
        if box.width() < min_width:
            continue
        if box.height() < min_height or box.height() > max_height:
            continue
        if cy < h * 0.28 or cy > h * 0.78:
            continue
        if width_ratio < 0.20:
            continue
        if height_ratio > 0.24:
            continue

        filtered.append(box)

    return filtered


def support_depth_profile(depth_map: np.ndarray, box: BoundingBox) -> tuple[float, float, float]:
    h, w = depth_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0))))
    x1 = max(x0 + 1, min(w, int(round(box.x1))))
    y0 = max(0, min(h - 1, int(round(box.y0))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0 or patch.shape[0] < 3:
        return 0.0, 0.0, 0.0
    row_medians = np.median(patch, axis=1)
    return float(row_medians[-1] - row_medians[0]), float(np.std(row_medians)), float(np.std(patch))


def normal_score_for_box(normals_map: Optional[np.ndarray], box: BoundingBox, cfg: dict) -> float:
    if normals_map is None:
        return 0.0
    h, w, _ = normals_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0))))
    y0 = max(0, min(h - 1, int(round(box.y0))))
    x1 = max(x0 + 1, min(w, int(round(box.x1))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))
    patch = normals_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    # Marigold normals are screen-space [-1,1]; favor surfaces with upward-ish Y
    mean_n = patch.reshape(-1, 3).mean(axis=0)
    up = float(mean_n[1])
    return up


def build_support_surface_mask(scene_image: Image.Image, box: BoundingBox, depth_map: Optional[np.ndarray], cfg: dict, scene_parse: Optional[dict] = None) -> np.ndarray:
    if scene_parse is not None:
        h, w = scene_image.height, scene_image.width
        full = np.zeros((h, w), dtype=np.float32)
        for m in scene_parse.get("support_masks", []):
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            seg_box = BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            if iou(seg_box, box) > 0.35:
                full = np.maximum(full, m.astype(np.float32))
        if full.max() > 0:
            blur_px = float(cfg["support_geometry"].get("surface_mask_blur_px", 7))
            if blur_px > 0:
                full = cv2.GaussianBlur(full, (0, 0), sigmaX=blur_px, sigmaY=blur_px)
            return np.clip(full, 0.0, 1.0)

    scene_arr = np.array(scene_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(scene_arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    x0 = max(0, min(w - 1, int(round(box.x0))))
    y0 = max(0, min(h - 1, int(round(box.y0))))
    x1 = max(x0 + 1, min(w, int(round(box.x1))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))
    roi = gray[y0:y1, x0:x1]
    mask = np.zeros((h, w), dtype=np.float32)
    if roi.size == 0:
        return mask
    edges = cv2.Canny(roi, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    inv_edges = (255 - edges).astype(np.float32) / 255.0
    inv_edges = cv2.GaussianBlur(inv_edges, (0, 0), sigmaX=2.0, sigmaY=2.0)
    local_mask = inv_edges
    if depth_map is not None:
        depth_roi = depth_map[y0:y1, x0:x1]
        if depth_roi.size > 0:
            gx = cv2.Sobel(depth_roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(depth_roi, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)
            grad = grad / max(1e-6, float(np.quantile(grad, 0.95)))
            grad = np.clip(grad, 0.0, 1.0)
            flatness = 1.0 - grad
            row_weight = np.linspace(0.7, 1.0, depth_roi.shape[0], dtype=np.float32)[:, None]
            local_mask *= np.clip(0.6 * flatness + 0.4 * row_weight, 0.0, 1.0)
    blur_px = float(cfg["support_geometry"].get("surface_mask_blur_px", 7))
    if blur_px > 0:
        local_mask = cv2.GaussianBlur(local_mask, (0, 0), sigmaX=blur_px, sigmaY=blur_px)
    mask[y0:y1, x0:x1] = local_mask
    return np.clip(mask, 0.0, 1.0)


def classify_support_geometry(box: BoundingBox, image_size: Tuple[int, int], depth_map: Optional[np.ndarray], cfg: dict, scene_image: Optional[Image.Image] = None, normals_map: Optional[np.ndarray] = None, scene_parse: Optional[dict] = None) -> SupportGeometry:
    scene_w, scene_h = image_size
    height_ratio = box.height() / max(1.0, scene_h)
    width_ratio = box.width() / max(1.0, scene_w)
    cy_ratio = box.centre()[1] / max(1.0, scene_h)
    if depth_map is not None:
        depth_slope, depth_variance, _ = support_depth_profile(depth_map, box)
    else:
        depth_slope, depth_variance = 0.0, 0.0

    geom_cfg = cfg["support_geometry"]
    is_thin = height_ratio <= float(geom_cfg["thin_height_ratio"])
    is_low_in_image = cy_ratio >= 0.42
    has_plane_depth = (depth_slope >= float(geom_cfg["plane_depth_slope_min"])) or (depth_variance >= float(geom_cfg["plane_depth_variance_min"]))
    nscore = normal_score_for_box(normals_map, box, cfg)

    if is_thin and depth_slope <= float(geom_cfg["edge_depth_slope_max"]):
        mode = "edge"
    elif height_ratio >= float(geom_cfg["plane_min_height_ratio"]) and is_low_in_image and has_plane_depth:
        mode = "plane"
    elif width_ratio > 0.22 and is_low_in_image:
        mode = "plane"
    else:
        mode = "edge"

    if mode == "plane":
        y_min = int(round(box.y0 + box.height() * float(geom_cfg["plane_back_start_ratio"])))
        y_max = int(round(box.y0 + box.height() * float(geom_cfg["plane_front_end_ratio"])))
        y_min = max(int(round(box.y0)), min(y_min, int(round(box.y1)) - 1))
        y_max = max(y_min + 1, min(int(round(box.y1)), y_max))
        score = width_ratio * 2.0 + height_ratio * 2.5 + max(0.0, depth_slope) * 4.0 + depth_variance * 3.0 + cy_ratio + max(0.0, nscore) * 1.5
    else:
        edge_y = int(round(box.y0 + float(geom_cfg["edge_contact_offset_px"])))
        y_min = edge_y
        y_max = edge_y
        score = width_ratio * 2.2 + (1.0 - min(0.2, height_ratio)) * 1.5 + (1.0 - min(0.2, max(0.0, depth_slope))) - max(0.0, nscore) * 0.5

    surface_mask = build_support_surface_mask(scene_image, box, depth_map, cfg, scene_parse=scene_parse) if scene_image is not None else None
    return SupportGeometry(box=box, mode=mode, plane_y_min=y_min, plane_y_max=y_max, depth_slope=float(depth_slope), depth_variance=float(depth_variance), score=float(score), surface_mask=surface_mask, normal_score=float(nscore))


def build_support_geometries(boxes: List[BoundingBox], image_size: Tuple[int, int], depth_map: Optional[np.ndarray], cfg: dict, scene_image: Image.Image, normals_map: Optional[np.ndarray] = None, scene_parse: Optional[dict] = None) -> List[SupportGeometry]:
    geoms = [classify_support_geometry(box, image_size, depth_map, cfg, scene_image, normals_map=normals_map, scene_parse=scene_parse) for box in boxes]
    geoms.sort(key=lambda g: g.score, reverse=True)
    return geoms


def support_mask_coverage(support_mask: Optional[np.ndarray], candidate_box: BoundingBox, threshold: float = 0.28) -> float:
    if support_mask is None:
        return 1.0
    h, w = support_mask.shape
    x0 = max(0, min(w - 1, int(round(candidate_box.x0))))
    x1 = max(x0 + 1, min(w, int(round(candidate_box.x1))))
    foot_h = max(2, int(round(candidate_box.height() * 0.12)))
    y0 = max(0, min(h - 1, int(round(candidate_box.y1 - foot_h))))
    y1 = max(y0 + 1, min(h, int(round(candidate_box.y1))))
    patch = support_mask[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch >= threshold))


def candidate_positions_on_support(support: SupportGeometry, target_w: int, target_h: int, scene_w: int, scene_h: int, cfg: dict) -> List[tuple[int, int]]:
    box = support.box
    edge_margin = box.width() * float(cfg["placement"].get("edge_margin_ratio", 0.05))
    usable_left = int(round(box.x0 + edge_margin))
    usable_right = int(round(box.x1 - edge_margin))
    xs = []
    step_x = max(8, target_w // max(2, int(cfg["placement"].get("candidate_step_x_divisor", 7))))
    x = usable_left
    while x + target_w <= usable_right:
        xs.append(x)
        x += step_x
    if usable_right - target_w >= usable_left:
        xs.append(usable_right - target_w)
    xs = list(dict.fromkeys(xs))

    ys: List[int] = []
    if support.mode == "edge":
        ys = [support.plane_y_min]
    else:
        step_y = max(5, target_h // max(2, int(cfg["support_geometry"].get("plane_candidate_step_y_divisor", 7))))
        y = support.plane_y_min
        while y <= support.plane_y_max:
            ys.append(y)
            y += step_y
        ys.append(support.plane_y_max)
        ys = list(dict.fromkeys(ys))

    out = []
    for foot_y in ys:
        for xx in xs:
            out.append((max(0, min(scene_w - target_w, xx)), max(0, min(scene_h - 1, foot_y))))
    return out


def build_fallback_support_boxes(image_size: Tuple[int, int]) -> List[BoundingBox]:
    w, h = image_size
    return [
        BoundingBox(int(w * 0.08), int(h * 0.34), int(w * 0.92), int(h * 0.63), label="fallback_counter"),
    ]

def rank_global_fallback_placements(scene_image: Image.Image, extracted_object: ExtractedObject, avoid_boxes: List[BoundingBox], depth_map: Optional[np.ndarray], cfg: dict) -> List[PlacementCandidate]:
    scene_w, scene_h = scene_image.size
    out: List[PlacementCandidate] = []
    obj_aspect = extracted_object.rgba.size[0] / max(1.0, extracted_object.rgba.size[1])
    base_w = max(int(cfg["placement"].get("min_object_width_px", 24)), int(scene_w * float(cfg["placement"].get("default_object_width_ratio", 0.12))))
    scales = [0.85, 1.0, 1.15]
    y_fracs = [0.62, 0.70, 0.78, 0.84]
    for scale in scales:
        w = max(24, int(round(base_w * scale)))
        h = max(24, int(round(w / max(0.1, obj_aspect))))
        step_x = max(10, w // 4)
        margin = int(scene_w * float(cfg["placement"].get("fallback_support_margin_ratio", 0.08)))
        for foot_frac in y_fracs:
            foot_y = int(scene_h * foot_frac)
            for x in range(margin, max(margin + 1, scene_w - w - margin + 1), step_x):
                y = foot_y - h
                candidate_box = BoundingBox(x, y, x + w, y + h)
                if candidate_box.y0 < 0 or candidate_box.x1 > scene_w or candidate_box.y1 > scene_h:
                    continue
                overlaps = [iou(candidate_box, other) for other in avoid_boxes]
                inter_ratios = [candidate_intersection_ratio(candidate_box, other) for other in avoid_boxes]
                max_overlap = max(overlaps, default=0.0)
                max_inter = max(inter_ratios, default=0.0)
                if max_overlap > 0.08 or max_inter > 0.18:
                    continue
                depth_median, depth_std = local_depth_stats(depth_map, x=x + int(w * 0.1), y=max(0, y + int(h * 0.72)), w=max(6, int(w * 0.8)), h=max(4, int(h * 0.2))) if depth_map is not None else (0.5, 0.25)
                bottom_bias = abs(0.80 - foot_frac)
                center_offset = abs((x + w * 0.5) - scene_w * 0.5) / max(1.0, scene_w)
                clutter = sum(1.0 / max(30.0, (((other.centre()[0] - candidate_box.centre()[0]) ** 2 + (other.centre()[1] - candidate_box.centre()[1]) ** 2) ** 0.5)) for other in avoid_boxes)
                score = max_overlap * 8.0 + sum(overlaps) * 2.5 + max_inter * 5.0 + depth_std * 1.2 + bottom_bias * 1.5 + center_offset * 0.5 + clutter * 0.8
                out.append(PlacementCandidate(Placement(x, y, w, h, None), float(score), debug=f"global_fallback overlap={max_overlap:.3f} inter={max_inter:.3f} depth_std={depth_std:.3f}"))
    out.sort(key=lambda c: c.score)
    return out[: int(cfg["placement"].get("top_k_to_keep", 16))]


def iou(a: BoundingBox, b: BoundingBox) -> float:
    inter = intersection_area(a, b)
    union = a.area() + b.area() - inter
    return 0.0 if union <= 0.0 else inter / union


def scene_parse_oneformer(scene_image: Image.Image, device: torch.device, cfg: dict) -> Optional[dict]:
    if not cfg["segmentation"].get("use_oneformer", True):
        return None
    try:
        processor, model = get_oneformer(device, cfg)
        task = str(cfg["segmentation"].get("oneformer_task", "panoptic"))
        inputs = processor(images=scene_image, task_inputs=[task], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        processed = processor.post_process_panoptic_segmentation(outputs, target_sizes=[scene_image.size[::-1]])[0]
        seg = processed["segmentation"].cpu().numpy()
        segments_info = processed["segments_info"]
        id2label = model.config.id2label
        h, w = seg.shape
        min_area = int(h * w * float(cfg["segmentation"].get("min_segment_area_ratio", 0.002)))

        support_masks = []
        support_boxes = []
        avoid_masks = []
        avoid_boxes = []

        support_keywords = [s.lower() for s in cfg["segmentation"].get("support_label_keywords", [])]
        avoid_keywords = [s.lower() for s in cfg["segmentation"].get("avoid_label_keywords", [])]

        for info in segments_info:
            sid = info["id"]
            label_id = info["label_id"]
            label = str(id2label.get(label_id, label_id)).lower()
            mask = (seg == sid).astype(np.uint8)
            area = int(mask.sum())
            if area < min_area:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            box = BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), label=label)

            if any(k in label for k in support_keywords):
                support_masks.append(mask.astype(np.float32))
                support_boxes.append(box)
            if any(k in label for k in avoid_keywords):
                avoid_masks.append(mask.astype(np.float32))
                avoid_boxes.append(box)

        return {
            "segmentation": seg,
            "segments_info": segments_info,
            "support_masks": support_masks,
            "support_boxes": support_boxes,
            "avoid_masks": avoid_masks,
            "avoid_boxes": avoid_boxes,
        }
    except Exception:
        return None


def rank_placements(
    scene_image: Image.Image,
    extracted_object: ExtractedObject,
    support_boxes: List[BoundingBox],
    avoid_boxes: List[BoundingBox],
    existing_same_objects: List[BoundingBox],
    depth_map: Optional[np.ndarray],
    cfg: dict,
    normals_map: Optional[np.ndarray] = None,
    scene_parse: Optional[dict] = None,
) -> tuple[List[PlacementCandidate], List[SupportGeometry]]:
    scene_w, scene_h = scene_image.size
    if not support_boxes:
        support_boxes = build_fallback_support_boxes(scene_image.size)

    support_geometries = build_support_geometries(
        support_boxes,
        scene_image.size,
        depth_map,
        cfg,
        scene_image,
        normals_map=normals_map,
        scene_parse=scene_parse,
    )

    collision_cfg = cfg.get("collision", {})
    use_occ_map = bool(collision_cfg.get("use_occupancy_map", True))
    occupancy_map = (
        build_occupancy_map(
            scene_image.size,
            avoid_boxes,
            blur_px=int(collision_cfg.get("occupancy_blur_px", 9)),
        )
        if use_occ_map
        else None
    )

    def generate_candidates(relaxed: bool) -> List[PlacementCandidate]:
        out: List[PlacementCandidate] = []

        # strict thresholds from config
        strict_max_iou = float(collision_cfg.get("max_iou", 0.01))
        strict_max_inter = float(collision_cfg.get("max_intersection_ratio_of_candidate", 0.02))
        strict_occ_thresh = float(collision_cfg.get("occupancy_threshold", 0.20))
        strict_occ_weight = float(collision_cfg.get("occupancy_penalty_weight", 10.0))

        # relaxed thresholds if strict search fails
        if relaxed:
            max_iou = max(strict_max_iou, 0.18)
            max_inter = max(strict_max_inter, 0.32)
            occ_thresh = max(strict_occ_thresh, 0.55)
            occ_weight = min(strict_occ_weight, 2.0)
            max_supports = max(int(cfg["placement"].get("max_supports_to_try", 8)), len(support_geometries))
            hard_occ_reject = False
        else:
            max_iou = strict_max_iou
            max_inter = strict_max_inter
            occ_thresh = strict_occ_thresh
            occ_weight = strict_occ_weight
            max_supports = int(cfg["placement"].get("max_supports_to_try", 8))
            hard_occ_reject = bool(collision_cfg.get("hard_occupancy_reject", False))

        stats = {
            "supports": 0,
            "positions": 0,
            "oob": 0,
            "collision": 0,
            "occupancy": 0,
            "accepted": 0,
        }

        for support in support_geometries[:max_supports]:
            stats["supports"] += 1

            seed_depth = depth_at_box_base(depth_map, support.box) if depth_map is not None else 0.5
            base_w, base_h = choose_target_size(
                scene_image.size,
                extracted_object.rgba.size,
                support.box,
                existing_same_objects,
                depth_map=depth_map,
                candidate_depth=seed_depth,
                candidate_center=support.box.centre(),
                cfg=cfg,
            )

            positions = candidate_positions_on_support(
                support,
                base_w,
                base_h,
                scene_w,
                scene_h,
                cfg,
            )

            # relaxed mode also tries slightly smaller objects, which helps avoid-box rejection a lot
            scale_trials = [1.0] if not relaxed else [0.82, 0.92, 1.0]

            for x, foot_y in positions:
                for scale_mul in scale_trials:
                    stats["positions"] += 1

                    scaled_w = max(24, int(round(base_w * scale_mul)))
                    scaled_h = max(24, int(round(base_h * scale_mul)))

                    sample_y = foot_y - int(scaled_h * (0.20 if support.mode == "edge" else 0.08))
                    depth_median, depth_std = (
                        local_depth_stats(
                            depth_map,
                            x=x + int(scaled_w * 0.1),
                            y=max(0, sample_y),
                            w=max(6, int(scaled_w * 0.8)),
                            h=max(4, int(scaled_h * 0.22)),
                        )
                        if depth_map is not None
                        else (0.5, 0.25)
                    )

                    center_guess = (x + scaled_w * 0.5, foot_y - scaled_h * 0.5)
                    target_w, target_h = choose_target_size(
                        scene_image.size,
                        extracted_object.rgba.size,
                        support.box,
                        existing_same_objects,
                        depth_map=depth_map,
                        candidate_depth=depth_median,
                        candidate_center=center_guess,
                        cfg=cfg,
                    )

                    target_w = max(24, int(round(target_w * scale_mul)))
                    target_h = max(24, int(round(target_h * scale_mul)))

                    persp = (
                        float(
                            np.interp(
                                np.clip(
                                    (foot_y - support.plane_y_min)
                                    / max(1.0, support.plane_y_max - support.plane_y_min),
                                    0.0,
                                    1.0,
                                ),
                                [0.0, 1.0],
                                [0.92, 1.18],
                            )
                        )
                        if support.mode == "plane"
                        else 1.0
                    )

                    target_w = int(round(target_w * persp))
                    target_h = int(round(target_h * persp))
                    obj_y = int(round(foot_y - target_h))
                    candidate_box = BoundingBox(x, obj_y, x + target_w, obj_y + target_h)

                    if (
                        candidate_box.y0 < 0
                        or candidate_box.x0 < 0
                        or candidate_box.x1 > scene_w
                        or candidate_box.y1 > scene_h
                    ):
                        stats["oob"] += 1
                        continue

                    overlaps = [iou(candidate_box, other) for other in avoid_boxes]
                    inter_ratios = [candidate_intersection_ratio(candidate_box, other) for other in avoid_boxes]
                    max_overlap = max(overlaps, default=0.0)
                    max_intersection = max(inter_ratios, default=0.0)

                    if max_overlap > max_iou or max_intersection > max_inter:
                        stats["collision"] += 1
                        continue

                    occ_score = occupancy_score_for_box(occupancy_map, candidate_box) if occupancy_map is not None else 0.0
                    if occupancy_map is not None and hard_occ_reject and occ_score > occ_thresh:
                        stats["occupancy"] += 1
                        continue

                    occupancy_penalty = (
                        max(0.0, occ_score - occ_thresh * 0.5) * occ_weight
                        if occupancy_map is not None
                        else 0.0
                    )

                    support_cov = support_mask_coverage(
                        support.surface_mask,
                        candidate_box,
                        float(cfg["support_geometry"].get("surface_valid_threshold", 0.20)),
                    )

                    if relaxed:
                        support_penalty = max(0.0, 0.25 - support_cov) * float(
                            cfg["placement"].get("support_mask_penalty_weight", 2.5)
                        ) * 0.35
                    else:
                        support_penalty = max(0.0, 0.45 - support_cov) * float(
                            cfg["placement"].get("support_mask_penalty_weight", 2.5)
                        )

                    center_offset = abs(candidate_box.centre()[0] - support.box.centre()[0]) / max(1.0, support.box.width())
                    support_band_pref = (
                        abs(
                            (foot_y - support.plane_y_min)
                            / max(1.0, support.plane_y_max - support.plane_y_min)
                            - 0.35
                        )
                        if support.mode == "plane"
                        else 0.0
                    )

                    predicted_w, predicted_h = choose_target_size(
                        scene_image.size,
                        extracted_object.rgba.size,
                        support.box,
                        existing_same_objects,
                        depth_map=depth_map,
                        candidate_depth=depth_median,
                        candidate_center=candidate_box.centre(),
                        cfg=cfg,
                    )

                    size_consistency = (
                        abs(predicted_w - target_w) / max(1.0, predicted_w)
                        + abs(predicted_h - target_h) / max(1.0, predicted_h)
                    )

                    empty_space_score = 0.0
                    ccx, ccy = candidate_box.centre()
                    for other in avoid_boxes:
                        ocx, ocy = other.centre()
                        dist = ((ocx - ccx) ** 2 + (ocy - ccy) ** 2) ** 0.5
                        empty_space_score += 1.0 / max(30.0, dist)

                    support_depth = depth_at_box_base(depth_map, support.box) if depth_map is not None else depth_median
                    support_depth_mismatch = abs(depth_median - support_depth) * float(
                        cfg["placement"].get("support_depth_mismatch_weight", 2.0)
                    )

                    normal_penalty = (
                        max(0.0, float(cfg["geometry"].get("plane_normal_min_up", 0.35)) - support.normal_score)
                        * float(cfg["placement"].get("normal_penalty_weight", 2.5))
                        if support.mode == "plane"
                        else 0.0
                    )

                    score = (
                        max_overlap * cfg["placement"]["avoid_overlap_weight"]
                        + sum(overlaps) * cfg["placement"]["total_overlap_weight"]
                        + depth_std * cfg["placement"]["depth_std_weight"]
                        + center_offset * cfg["placement"]["center_offset_weight"]
                        + support_band_pref * cfg["placement"]["support_band_weight"]
                        + (1.0 - min(1.0, persp)) * cfg["placement"]["perspective_weight"]
                        + empty_space_score * cfg["placement"]["favor_empty_space_weight"]
                        + size_consistency * cfg["placement"]["size_consistency_weight"]
                        + support_preference_adjustment(support, cfg)
                        + occupancy_penalty
                        + support_penalty
                        + support_depth_mismatch
                        + normal_penalty
                    )

                    out.append(
                        PlacementCandidate(
                            Placement(
                                int(candidate_box.x0),
                                int(candidate_box.y0),
                                target_w,
                                target_h,
                                support.box,
                            ),
                            float(score),
                            debug=(
                                f"{'relaxed' if relaxed else 'strict'} "
                                f"label={support.box.label} mode={support.mode} "
                                f"iou={max_overlap:.3f} inter={max_intersection:.3f} "
                                f"occ={occ_score:.3f} cov={support_cov:.3f} "
                                f"depth_std={depth_std:.3f} n={support.normal_score:.3f}"
                            ),
                        )
                    )
                    stats["accepted"] += 1

        out.sort(key=lambda c: c.score)
        out = out[: int(cfg["placement"].get("top_k_to_keep", 16))]
        print(
            f"[{'relaxed' if relaxed else 'strict'} placement] "
            f"supports={stats['supports']} positions={stats['positions']} "
            f"accepted={stats['accepted']} oob={stats['oob']} "
            f"collision={stats['collision']} occupancy={stats['occupancy']}"
        )
        return out

    ranked = generate_candidates(relaxed=False)

    if not ranked:
        print("Strict placement found no candidates; retrying with relaxed overlap and occupancy limits.")
        ranked = generate_candidates(relaxed=True)

    if not ranked and bool(cfg["placement"].get("fallback_global_search", True)):
        print("Relaxed support placement still found no candidates; trying global fallback search.")
        ranked = rank_global_fallback_placements(scene_image, extracted_object, avoid_boxes, depth_map, cfg)

    return ranked, support_geometries

def choose_placement_from_ranked(candidates: List[PlacementCandidate], attempt_index: int) -> Placement:
    if not candidates:
        raise RuntimeError("No plausible placement candidates found.")
    idx = max(0, min(attempt_index, len(candidates) - 1))
    return candidates[idx].placement


def sample_surface_patch(scene_image: Image.Image, placement: Placement) -> Image.Image:
    scene_w, scene_h = scene_image.size
    x0 = max(0, placement.x - int(placement.width * 0.15))
    x1 = min(scene_w, placement.x + int(placement.width * 1.15))
    y0 = max(0, placement.y + int(placement.height * 0.68))
    y1 = min(scene_h, placement.y + int(placement.height * 1.05))
    if x1 <= x0 or y1 <= y0:
        return scene_image.crop((0, 0, min(scene_w, 32), min(scene_h, 32))).convert("RGB")
    return scene_image.crop((x0, y0, x1, y1)).convert("RGB")


def estimate_light_direction(scene_patch: Image.Image) -> Tuple[float, float]:
    gray = np.array(scene_patch.convert("L"), dtype=np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    weight = cv2.GaussianBlur(gray, (0, 0), 1.4) + 1e-3
    vx = float(np.mean(gx * weight))
    vy = float(np.mean(gy * weight))
    norm = max(1e-6, (vx * vx + vy * vy) ** 0.5)
    sx, sy = -vx / norm, max(0.12, -vy / norm)
    norm2 = max(1e-6, (sx * sx + sy * sy) ** 0.5)
    return sx / norm2, sy / norm2


def surface_shadow_color(scene_patch: Image.Image) -> Tuple[int, int, int]:
    med = np.median(np.array(scene_patch, dtype=np.uint8).reshape(-1, 3), axis=0)
    tinted = np.clip(med * 0.34, 0, 255).astype(np.uint8)
    return int(tinted[0]), int(tinted[1]), int(tinted[2])


def surface_patch_stats(scene_patch: Image.Image) -> tuple[float, float, np.ndarray, float]:
    arr = np.array(scene_patch.convert("RGB"), dtype=np.float32)
    gray = np.array(scene_patch.convert("L"), dtype=np.float32) / 255.0
    rgb_mean = arr.reshape(-1, 3).mean(axis=0) / 255.0
    sat = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32) / 255.0
    return float(np.mean(gray)), float(np.std(gray)), rgb_mean.astype(np.float32), float(np.mean(sat))


def relight_object_to_patch(obj_rgba: Image.Image, scene_patch: Image.Image, light_dir: Tuple[float, float], light_strength: float, cfg: dict, scene_shading_patch: Optional[np.ndarray] = None) -> Image.Image:
    if not cfg.get("relighting", {}).get("enabled", True):
        return obj_rgba
    arr = np.array(obj_rgba.convert("RGBA"), dtype=np.uint8)
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    if alpha.max() <= 0:
        return obj_rgba

    obj_gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    patch_mean, patch_std, patch_rgb_mean, patch_sat = surface_patch_stats(scene_patch)
    obj_mask = alpha > 0.03
    obj_mean = float(obj_gray[obj_mask].mean()) if np.any(obj_mask) else 0.5
    obj_std = float(obj_gray[obj_mask].std()) if np.any(obj_mask) else 0.2

    mean_strength = float(cfg["relighting"].get("mean_match_strength", 0.50))
    std_strength = float(cfg["relighting"].get("std_match_strength", 0.35))
    scale = (patch_std / max(0.04, obj_std)) ** std_strength
    shift = (patch_mean - obj_mean) * mean_strength
    rgb = np.clip((rgb - obj_mean) * scale + obj_mean + shift, 0.0, 1.0)

    obj_rgb_mean = np.array([rgb[:, :, c][obj_mask].mean() if np.any(obj_mask) else 0.5 for c in range(3)], dtype=np.float32)
    color_strength = float(cfg["relighting"].get("color_match_strength", 0.25))
    rgb = np.clip(rgb * (1.0 - color_strength) + rgb * (patch_rgb_mean / np.maximum(0.08, obj_rgb_mean))[None, None, :] * color_strength, 0.0, 1.0)

    sat_strength = float(cfg["relighting"].get("saturation_match_strength", 0.12))
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= float(np.clip(1.0 + (patch_sat - 0.5) * sat_strength * 2.0, 0.8, 1.15))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    h, w = alpha.shape
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    lx, ly = light_dir
    shade = 1.0 + (-lx * xs * 0.5 - ly * ys * 0.8) * float(cfg["relighting"].get("directional_shading_strength", 0.18)) * float(np.clip(light_strength, 0.2, 1.0))
    bottom_occlusion = 1.0 - np.clip((ys + 1.0) * 0.5, 0.0, 1.0) * float(cfg["relighting"].get("bottom_occlusion_strength", 0.10))
    highlight = 1.0 + np.clip((-lx * xs - ly * ys), 0.0, 1.0) * float(cfg["relighting"].get("highlight_strength", 0.05)) * light_strength
    field = np.clip(shade * bottom_occlusion * highlight, 0.75, 1.25)

    if scene_shading_patch is not None and cfg["relighting"].get("use_intrinsics_shading", True):
        sp = scene_shading_patch
        if sp.shape[:2] != (h, w):
            sp = cv2.resize(sp, (w, h), interpolation=cv2.INTER_LINEAR)
        if sp.ndim == 3:
            sp = sp.mean(axis=2)
        smin, smax = float(sp.min()), float(sp.max())
        if smax - smin > 1e-6:
            sp = (sp - smin) / (smax - smin)
            field *= np.clip(0.9 + 0.25 * sp, 0.8, 1.2)

    rgb = np.clip(rgb * field[:, :, None], 0.0, 1.0)
    out = np.dstack([(rgb * 255).astype(np.uint8), (alpha * 255).astype(np.uint8)])
    return Image.fromarray(out, mode="RGBA")


def build_contact_shadow_from_alpha(alpha: Image.Image, placement: Placement, shadow_rgb: Tuple[int, int, int], cfg: dict, brightness: float, contrast: float) -> Image.Image:
    a = np.array(alpha, dtype=np.uint8)
    h, w = a.shape
    ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))
    band_h = max(2, int(round(h * float(cfg["shadow"].get("ambient_occlusion_band_ratio", 0.08)))))
    footprint = np.zeros((h, w), dtype=np.uint8)
    for xx in range(w):
        col = np.where(a[:, xx] > 10)[0]
        if len(col) == 0:
            continue
        yb = int(col.max())
        y0 = max(0, yb - band_h // 2)
        y1 = min(h, yb + band_h // 2 + 1)
        footprint[y0:y1, xx] = 255
    footprint = cv2.GaussianBlur(footprint, (0, 0), sigmaX=float(cfg["shadow"].get("contact_blur_px", 2.4)) * (0.9 + 0.2 * (1.0 - brightness)), sigmaY=max(0.8, float(cfg["shadow"].get("contact_blur_px", 2.4)) * 0.7))
    opacity = float(cfg["shadow"].get("contact_opacity", 0.28)) * float(np.clip(0.9 + contrast, 0.8, 1.25))
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.array(shadow_rgb, dtype=np.uint8)
    rgba[:, :, 3] = np.clip(footprint.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def build_projected_shadow(alpha: Image.Image, placement: Placement, light_dir: Tuple[float, float], light_strength: float, cfg: dict, scene_patch: Image.Image, support_box: Optional[BoundingBox] = None, depth_map: Optional[np.ndarray] = None) -> Image.Image:
    a = np.array(alpha, dtype=np.uint8)
    brightness, contrast, _, _ = surface_patch_stats(scene_patch)
    support_factor = float(np.clip((support_box.height() / max(1.0, placement.height)) if support_box is not None else 1.0, 0.7, 1.4))
    depth_factor = 1.0
    if depth_map is not None:
        d = depth_at_box_base(depth_map, BoundingBox(placement.x, placement.y, placement.x + placement.width, placement.y + placement.height))
        depth_factor = float(np.interp(d, [0.0, 1.0], [0.78, 1.22]))
    directional = float(np.clip(light_strength, 0.18, 1.0))
    adaptive_squash = float(np.clip(float(cfg["shadow"]["squash_ratio"]) + 0.04 * (1.0 - brightness), 0.22, 0.50))
    adaptive_shear = float(np.clip(float(cfg["shadow"]["shear_strength"]) * (0.7 + 0.8 * directional), 0.08, 0.50))
    adaptive_length = float(np.clip(float(cfg["shadow"]["cast_length_scale"]) * support_factor * depth_factor * (0.65 + 0.8 * directional), 0.25, 1.35))
    new_h = max(1, int(round(a.shape[0] * adaptive_squash)))
    squashed = cv2.resize(a, (a.shape[1], new_h), interpolation=cv2.INTER_LINEAR)
    dx = int(round(light_dir[0] * placement.width * adaptive_shear))
    dy = int(round(light_dir[1] * placement.height * adaptive_length))
    canvas_h = max(new_h + abs(dy) + 8, placement.height)
    canvas_w = max(a.shape[1] + abs(dx) + 8, placement.width)
    src = np.float32([[0, 0], [squashed.shape[1] - 1, 0], [0, squashed.shape[0] - 1]])
    dst = np.float32([[max(0, dx), 0], [max(0, dx) + squashed.shape[1] - 1, 0], [0, squashed.shape[0] - 1 + max(0, dy)]])
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(squashed, M, (canvas_w, canvas_h))
    blur_sigma = float(cfg["shadow"]["cast_blur_px"]) * (0.75 + float(cfg["shadow"].get("shadow_softness_influence", 0.35)) * (1.0 - contrast) + 0.45 * adaptive_length)
    warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=blur_sigma, sigmaY=max(0.8, blur_sigma * 0.75))
    opacity = float(cfg["shadow"]["cast_opacity"]) * float(np.clip(0.72 + 0.55 * directional + 0.12 * contrast, 0.65, 1.35))
    warped = np.clip(warped.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    return Image.fromarray(warped, mode="L")


def build_scene_occlusion_alpha(scene_depth: np.ndarray, placement: Placement, obj_alpha: Image.Image, cfg: dict, occluder_mask_full: Optional[np.ndarray] = None) -> Image.Image:
    occ_cfg = cfg.get("occlusion", {})
    if not occ_cfg.get("enabled", True):
        return Image.new("L", (placement.width, placement.height), 0)

    H, W = scene_depth.shape
    x0 = max(0, min(W - 1, placement.x))
    y0 = max(0, min(H - 1, placement.y))
    x1 = max(x0 + 1, min(W, placement.x + placement.width))
    y1 = max(y0 + 1, min(H, placement.y + placement.height))
    scene_patch = scene_depth[y0:y1, x0:x1]
    if scene_patch.size == 0:
        return Image.new("L", (placement.width, placement.height), 0)
    if scene_patch.shape[::-1] != (placement.width, placement.height):
        scene_patch = cv2.resize(scene_patch, (placement.width, placement.height), interpolation=cv2.INTER_LINEAR)

    alpha = np.array(obj_alpha, dtype=np.uint8).astype(np.float32) / 255.0
    ys = np.linspace(0.0, 1.0, placement.height, dtype=np.float32)[:, None]
    object_depth = np.full_like(scene_patch, depth_at_box_base(scene_depth, BoundingBox(placement.x, placement.y, placement.x + placement.width, placement.y + placement.height)), dtype=np.float32)
    object_depth -= ys * float(occ_cfg.get("object_depth_top_offset", 0.08))
    bias = float(occ_cfg.get("depth_bias", 0.03))
    hardness = float(occ_cfg.get("foreground_hardness", 0.70))
    closer = (scene_patch + bias) < object_depth
    depth_delta = np.clip((object_depth - scene_patch - bias) / max(1e-6, 0.10), 0.0, 1.0)
    occ = np.where(closer, depth_delta * hardness, 0.0) * alpha

    if occluder_mask_full is not None and occ_cfg.get("prefer_segment_occluders", True):
        occ_mask = occluder_mask_full[y0:y1, x0:x1]
        if occ_mask.shape != occ.shape:
            occ_mask = cv2.resize(occ_mask, (placement.width, placement.height), interpolation=cv2.INTER_LINEAR)
        occ = np.maximum(occ, occ_mask * alpha * 0.85)

    feather = float(occ_cfg.get("feather_px", 2.0))
    if feather > 0:
        occ = cv2.GaussianBlur(occ, (0, 0), sigmaX=feather, sigmaY=feather)
    return Image.fromarray(np.clip(occ * 255, 0, 255).astype(np.uint8), mode="L")

def resize_rgba_premultiplied(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    arr = np.array(img.convert("RGBA"), dtype=np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    rgb = arr[:, :, :3] * alpha

    rgb_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").resize(size, Image.LANCZOS)
    a_img = Image.fromarray(arr[:, :, 3].astype(np.uint8), mode="L").resize(size, Image.LANCZOS)

    rgb_r = np.array(rgb_img, dtype=np.float32)
    a_r = np.array(a_img, dtype=np.float32) / 255.0
    out_rgb = np.zeros_like(rgb_r)
    nz = a_r > 1e-6
    out_rgb[nz] = rgb_r[nz] / a_r[nz, None]
    out = np.dstack([np.clip(out_rgb, 0, 255).astype(np.uint8), (a_r * 255).astype(np.uint8)])
    return Image.fromarray(out, mode="RGBA")

def composite_object(
    scene_image: Image.Image,
    extracted_object: ExtractedObject,
    placement: Placement,
    cfg: dict,
    depth_map: Optional[np.ndarray] = None,
    light_sources: Optional[List[LightSource]] = None,
    scene_shading: Optional[np.ndarray] = None,
    occluder_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    canvas = scene_image.convert("RGBA")

    # Resize object.
    obj = resize_rgba_premultiplied(extracted_object.rgba, (placement.width, placement.height))

    # Force object alpha to be fully opaque wherever it exists.
    # This completely blocks translucent / transparent lemon interiors.
    alpha_np = np.array(obj.getchannel("A"), dtype=np.uint8)
    obj.putalpha(Image.fromarray(alpha_np, mode="L"))
    alpha = obj.getchannel("A")

    surface_patch = sample_surface_patch(scene_image, placement)

    if cfg.get("lighting", {}).get("enabled", True):
        lx, ly, light_strength = estimate_light_vector_from_sources(
            placement, canvas.size, light_sources or [], cfg
        )
        light_dir = (lx, ly)
    else:
        light_dir = estimate_light_direction(surface_patch)
        light_strength = 0.5

    scene_shading_patch = None
    if scene_shading is not None:
        x0 = max(0, placement.x)
        y0 = max(0, placement.y)
        x1 = min(scene_image.width, placement.x + placement.width)
        y1 = min(scene_image.height, placement.y + placement.height)
        scene_shading_patch = scene_shading[y0:y1, x0:x1]

    obj = relight_object_to_patch(
        obj,
        surface_patch,
        light_dir,
        light_strength,
        cfg,
        scene_shading_patch=scene_shading_patch,
    )

    # Re-harden alpha after relighting too, in case anything softened it.
    alpha_np = np.array(obj.getchannel("A"), dtype=np.uint8)
    alpha_np = np.where(alpha_np > 0, 255, 0).astype(np.uint8)
    obj.putalpha(Image.fromarray(alpha_np, mode="L"))
    alpha = obj.getchannel("A")

    if cfg["shadow"].get("enabled", True):
        brightness, contrast, _, _ = surface_patch_stats(surface_patch)
        shadow_rgb = (
            surface_shadow_color(surface_patch)
            if cfg["shadow"].get("shadow_color_mode", "surface_tinted") == "surface_tinted"
            else (0, 0, 0)
        )
        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        contact_rgba = build_contact_shadow_from_alpha(
            alpha, placement, shadow_rgb, cfg, brightness, contrast
        )
        projected = build_projected_shadow(
            alpha,
            placement,
            light_dir,
            light_strength,
            cfg,
            surface_patch,
            support_box=placement.support_box,
            depth_map=depth_map,
        )
        proj_rgba = Image.new("RGBA", projected.size, shadow_rgb + (0,))
        proj_rgba.putalpha(projected)
        proj_x = placement.x + int(round(light_dir[0] * placement.width * 0.08))
        proj_y = placement.y + int(round(placement.height * 0.72))
        shadow_layer.alpha_composite(proj_rgba, dest=(proj_x, proj_y))
        shadow_layer.alpha_composite(contact_rgba, dest=(placement.x, placement.y))
        canvas = Image.alpha_composite(canvas, shadow_layer)

    obj_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    obj_layer.alpha_composite(obj, dest=(placement.x, placement.y))
    canvas = Image.alpha_composite(canvas, obj_layer)

    if depth_map is not None and cfg.get("occlusion", {}).get("enabled", True):
        occ_alpha = build_scene_occlusion_alpha(
            depth_map,
            placement,
            alpha,
            cfg,
            occluder_mask_full=occluder_mask,
        )
        occ_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        scene_crop = scene_image.crop(
            (placement.x, placement.y, placement.x + placement.width, placement.y + placement.height)
        ).convert("RGBA")
        scene_crop.putalpha(occ_alpha)
        occ_layer.alpha_composite(scene_crop, dest=(placement.x, placement.y))
        canvas = Image.alpha_composite(canvas, occ_layer)

    return canvas.convert("RGB")

def build_output_path(output_arg: str | Path, timestamp_outputs: bool = True) -> Path:
    output_path = Path(output_arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not timestamp_outputs:
        if output_path.suffix:
            return output_path
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / "output.jpg"
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"output_{timestamp}.jpg"


def save_debug_overlay(scene_image: Image.Image, output_path: Path, support_boxes: List[BoundingBox], placement: Placement, candidates: Optional[List[PlacementCandidate]] = None, depth_map: Optional[np.ndarray] = None, cfg: Optional[dict] = None, support_geometries: Optional[List[SupportGeometry]] = None) -> None:
    debug = scene_image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug)
    geoms = support_geometries or []
    if geoms:
        for geom in geoms:
            col = (0, 255, 0) if geom.mode == "plane" else (0, 180, 255)
            draw.rectangle(geom.box.to_int_tuple(), outline=col, width=3)
            draw.text((geom.box.x0 + 4, geom.box.y0 + 4), f"{geom.mode} ds={geom.depth_slope:.02f} n={geom.normal_score:.02f}", fill=col)
            draw.line((geom.box.x0, geom.plane_y_min, geom.box.x1, geom.plane_y_min), fill=(255, 255, 0), width=2)
            if geom.mode == "plane":
                draw.line((geom.box.x0, geom.plane_y_max, geom.box.x1, geom.plane_y_max), fill=(255, 200, 0), width=2)
    else:
        for box in support_boxes:
            draw.rectangle(box.to_int_tuple(), outline=(0, 255, 0), width=3)
            if box.label:
                draw.text((box.x0 + 4, box.y0 + 4), box.label, fill=(0, 255, 0))
    if candidates:
        for i, cand in enumerate(candidates[:8]):
            p = cand.placement
            col = (255, 180, 0) if i > 0 else (255, 0, 0)
            draw.rectangle((p.x, p.y, p.x + p.width, p.y + p.height), outline=col, width=2)
    place_box = BoundingBox(placement.x, placement.y, placement.x + placement.width, placement.y + placement.height)
    draw.rectangle(place_box.to_int_tuple(), outline=(255, 0, 0), width=4)
    debug.save(output_path)


def merge_support_boxes(det_boxes: List[BoundingBox], seg_boxes: List[BoundingBox], image_size: Tuple[int, int]) -> List[BoundingBox]:
    all_boxes = list(det_boxes) + list(seg_boxes)
    if not all_boxes:
        return []
    kept: List[BoundingBox] = []
    for b in sorted(all_boxes, key=lambda x: (x.area(), x.score), reverse=True):
        if all(iou(b, k) < 0.65 for k in kept):
            kept.append(b)
    return filter_support_boxes(kept, image_size)


def merge_avoid_boxes(det_boxes: List[BoundingBox], seg_boxes: List[BoundingBox]) -> List[BoundingBox]:
    all_boxes = list(det_boxes) + list(seg_boxes)
    kept: List[BoundingBox] = []
    for b in sorted(all_boxes, key=lambda x: x.area(), reverse=True):
        if all(iou(b, k) < 0.75 for k in kept):
            kept.append(b)
    return kept


def build_occluder_mask(scene_parse: Optional[dict], image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if scene_parse is None:
        return None
    w, h = image_size
    full = np.zeros((h, w), dtype=np.float32)
    for m in scene_parse.get("avoid_masks", []):
        full = np.maximum(full, m.astype(np.float32))
    if full.max() > 0:
        full = cv2.GaussianBlur(full, (0, 0), sigmaX=1.0, sigmaY=1.0)
        return np.clip(full, 0.0, 1.0)
    return None

def run_pipeline(
    scene_path: str | Path,
    object_image_path: str | Path,
    output_path: str | Path,
    *,
    device: torch.device,
    cfg: dict,
    debug_overlay_path: Optional[str | Path] = None,
    no_sam: bool = False,
) -> dict:
    global _SAM_AVAILABLE

    prev_sam_available = _SAM_AVAILABLE
    if no_sam:
        _SAM_AVAILABLE = False

    try:
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Using device: {device}")

        scene = open_rgb(scene_path)
        ref = open_rgb(object_image_path)

        depth_map = estimate_depth_map(scene, device, cfg)
        print("Estimated scene depth map with DepthPro.")

        normals_map = estimate_normals_map(scene, device, cfg)
        print(f"Normals {'enabled' if normals_map is not None else 'unavailable'}.")

        scene_shading = estimate_intrinsics_lighting(scene, device, cfg)
        print(f"Intrinsic lighting {'enabled' if scene_shading is not None else 'unavailable'}.")

        light_sources = detect_light_sources(scene, depth_map, cfg)
        print(f"Detected {len(light_sources)} likely light source(s).")

        extracted = extract_reference_object(ref, device, cfg)
        print(f"Extracted reference foreground: {extracted.rgba.size[0]}x{extracted.rgba.size[1]}")

        scene_parse = scene_parse_oneformer(scene, device, cfg)
        if scene_parse is not None:
            print(
                f"OneFormer scene parsing found {len(scene_parse['support_boxes'])} support segment(s) "
                f"and {len(scene_parse['avoid_boxes'])} occluder segment(s)."
            )
        else:
            print("OneFormer scene parsing unavailable; falling back to detector-only supports and obstacles.")

        det_support_boxes = detect_objects(
            scene,
            SURFACE_LABELS,
            device,
            cfg,
            threshold=cfg["detection"]["support_threshold"],
            text_threshold=cfg["detection"]["support_text_threshold"],
        )
        seg_support_boxes = scene_parse["support_boxes"] if scene_parse is not None else []
        support_boxes = merge_support_boxes(det_support_boxes, seg_support_boxes, scene.size)
        if not support_boxes:
            support_boxes = build_fallback_support_boxes(scene.size)
            print("No support surfaces detected; using fallback lower-scene support regions.")
        else:
            print(f"Using {len(support_boxes)} merged support surfaces.")

        existing_same_objects: List[BoundingBox] = []
        print("Skipped same-object scene matching because label-free extraction is enabled.")

        det_context_boxes = detect_objects(
            scene,
            CONTEXT_OBJECT_LABELS,
            device,
            cfg,
            threshold=cfg["detection"]["context_threshold"],
            text_threshold=cfg["detection"]["context_text_threshold"],
        )
        seg_context_boxes = scene_parse["avoid_boxes"] if scene_parse is not None else []
        avoid_boxes = merge_avoid_boxes(det_context_boxes, seg_context_boxes)

        ranked_candidates, support_geometries = rank_placements(
            scene,
            extracted,
            support_boxes,
            avoid_boxes,
            existing_same_objects,
            depth_map,
            cfg,
            normals_map=normals_map,
            scene_parse=scene_parse,
        )
        placement = choose_placement_from_ranked(
            ranked_candidates,
            int(cfg["placement"].get("attempt_index", 0)),
        )
        print(
            f"Placement chosen at x={placement.x}, y={placement.y}, "
            f"w={placement.width}, h={placement.height}"
        )

        occluder_mask = build_occluder_mask(scene_parse, scene.size)
        result = composite_object(
            scene,
            extracted,
            placement,
            cfg,
            depth_map=depth_map,
            light_sources=light_sources,
            scene_shading=scene_shading,
            occluder_mask=occluder_mask,
        )

        output_path = build_output_path(output_path, bool(cfg["output"].get("timestamp_outputs", True)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        print(f"Saved output image to {output_path}")

        debug_written = None
        if debug_overlay_path:
            debug_path = build_output_path(debug_overlay_path, bool(cfg["output"].get("timestamp_outputs", True)))
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            save_debug_overlay(
                scene,
                debug_path,
                support_boxes,
                placement,
                candidates=ranked_candidates,
                depth_map=depth_map,
                cfg=cfg,
                support_geometries=support_geometries,
            )
            print(f"Saved debug overlay to {debug_path}")
            debug_written = debug_path

        return {
            "output_path": str(output_path),
            "debug_overlay_path": str(debug_written) if debug_written else None,
            "placement": {
                "x": placement.x,
                "y": placement.y,
                "width": placement.width,
                "height": placement.height,
                "support_box": None if placement.support_box is None else {
                    "x0": placement.support_box.x0,
                    "y0": placement.support_box.y0,
                    "x1": placement.support_box.x1,
                    "y1": placement.support_box.y1,
                    "label": placement.support_box.label,
                    "score": placement.support_box.score,
                },
            },
            "num_candidates": len(ranked_candidates),
            "best_candidate_score": None if not ranked_candidates else ranked_candidates[0].score,
        }
    finally:
        _SAM_AVAILABLE = prev_sam_available

def sample_optuna_config(trial: optuna.trial.Trial, base_cfg: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)

    # High-level behavioral profile first.
    profile = trial.suggest_categorical(
        "profile",
        [
            "strict_realism",
            "balanced",
            "aggressive_placement",
            "loose_extraction",
            "dramatic_shadow",
        ],
    )

    # -------------------------
    # Extraction: much wider
    # -------------------------
    cfg["extraction"]["matte_edge_blur"] = 0.0

    cfg["extraction"]["erode_px"] = 0
    cfg["extraction"]["dilate_px"] = 1
    cfg["extraction"]["grabcut_refine"] = trial.suggest_categorical(
        "extraction.grabcut_refine", [True]
    )
    cfg["extraction"]["grabcut_iters"] = 1
    cfg["extraction"]["combine_mode"] = "primary_prefer"
    cfg["extraction"]["reference_box_pad_ratio"] = trial.suggest_float(
        "extraction.reference_box_pad_ratio", 0.24, 0.60, step=0.04
    )
    cfg["extraction"]["reference_box_min_pad_px"] = trial.suggest_int(
        "extraction.reference_box_min_pad_px", 48, 96, step=8
    )
    cfg["extraction"]["component_max_gap_px"] = trial.suggest_int(
        "extraction.component_max_gap_px", 16, 48, step=8
    )
    cfg["extraction"]["component_min_area_ratio"] = trial.suggest_float(
        "extraction.component_min_area_ratio", 0.00008, 0.0008, log=True
    )
    cfg["extraction"]["max_border_touch_ratio"] = trial.suggest_float(
        "extraction.max_border_touch_ratio", 0.10, 0.22, step=0.02
    )

    # -------------------------
    # Detection / segmentation
    # -------------------------
    cfg["detection"]["object_threshold"] = trial.suggest_float(
        "detection.object_threshold", 0.12, 0.40, step=0.02
    )
    cfg["detection"]["object_text_threshold"] = trial.suggest_float(
        "detection.object_text_threshold", 0.08, 0.30, step=0.02
    )
    cfg["detection"]["support_threshold"] = trial.suggest_float(
        "detection.support_threshold", 0.10, 0.38, step=0.02
    )
    cfg["detection"]["support_text_threshold"] = trial.suggest_float(
        "detection.support_text_threshold", 0.08, 0.28, step=0.02
    )
    cfg["detection"]["context_threshold"] = trial.suggest_float(
        "detection.context_threshold", 0.10, 0.38, step=0.02
    )
    cfg["detection"]["context_text_threshold"] = trial.suggest_float(
        "detection.context_text_threshold", 0.08, 0.28, step=0.02
    )

    cfg["segmentation"]["birefnet_threshold"] = 0.2
    cfg["segmentation"]["use_oneformer"] = trial.suggest_categorical(
        "segmentation.use_oneformer", [True]
    )
    cfg["segmentation"]["min_segment_area_ratio"] = trial.suggest_float(
        "segmentation.min_segment_area_ratio", 0.0005, 0.008, log=True
    )

    # -------------------------
    # Geometry / support interpretation
    # -------------------------
    cfg["geometry"]["use_normals"] = trial.suggest_categorical(
        "geometry.use_normals", [True]
    )
    cfg["geometry"]["use_intrinsics"] = trial.suggest_categorical(
        "geometry.use_intrinsics", [True, False]
    )
    cfg["geometry"]["plane_normal_min_up"] = trial.suggest_float(
        "geometry.plane_normal_min_up", 0.10, 0.65, step=0.05
    )

    cfg["support_geometry"]["thin_height_ratio"] = trial.suggest_float(
        "support_geometry.thin_height_ratio", 0.01, 0.08, step=0.01
    )
    cfg["support_geometry"]["plane_min_height_ratio"] = trial.suggest_float(
        "support_geometry.plane_min_height_ratio", 0.02, 0.12, step=0.01
    )
    cfg["support_geometry"]["edge_depth_slope_max"] = trial.suggest_float(
        "support_geometry.edge_depth_slope_max", 0.01, 0.08, step=0.01
    )
    cfg["support_geometry"]["plane_depth_slope_min"] = trial.suggest_float(
        "support_geometry.plane_depth_slope_min", 0.01, 0.08, step=0.01
    )
    cfg["support_geometry"]["plane_depth_variance_min"] = trial.suggest_float(
        "support_geometry.plane_depth_variance_min", 0.003, 0.030, step=0.003
    )
    cfg["support_geometry"]["plane_back_start_ratio"] = trial.suggest_float(
        "support_geometry.plane_back_start_ratio", 0.00, 0.25, step=0.025
    )
    cfg["support_geometry"]["plane_front_end_ratio"] = trial.suggest_float(
        "support_geometry.plane_front_end_ratio", 0.40, 0.95, step=0.05
    )
    cfg["support_geometry"]["surface_valid_threshold"] = trial.suggest_float(
        "support_geometry.surface_valid_threshold", 0.08, 0.55, step=0.04
    )

    # -------------------------
    # Support preferences
    # -------------------------
    cfg["support_preferences"]["prefer_mode"] = trial.suggest_categorical(
        "support_preferences.prefer_mode", ["plane", "edge", "any"]
    )
    cfg["support_preferences"]["label_match_bonus"] = trial.suggest_float(
        "support_preferences.label_match_bonus", 0.0, 5.0, step=0.5
    )
    cfg["support_preferences"]["mode_match_bonus"] = trial.suggest_float(
        "support_preferences.mode_match_bonus", 0.0, 3.0, step=0.5
    )
    cfg["support_preferences"]["disfavored_label_penalty"] = trial.suggest_float(
        "support_preferences.disfavored_label_penalty", 0.0, 4.0, step=0.5
    )

    # -------------------------
    # Placement: much wider
    # -------------------------
    cfg["placement"]["max_supports_to_try"] = trial.suggest_int(
        "placement.max_supports_to_try", 2, 16
    )
    cfg["placement"]["candidate_step_x_divisor"] = trial.suggest_int(
        "placement.candidate_step_x_divisor", 3, 14
    )
    cfg["placement"]["top_k_to_keep"] = trial.suggest_int(
        "placement.top_k_to_keep", 8, 32, step=4
    )
    cfg["placement"]["edge_margin_ratio"] = trial.suggest_float(
        "placement.edge_margin_ratio", 0.00, 0.18, step=0.02
    )
    cfg["placement"]["avoid_overlap_weight"] = trial.suggest_float(
        "placement.avoid_overlap_weight", 0.5, 12.0, step=0.5
    )
    cfg["placement"]["total_overlap_weight"] = trial.suggest_float(
        "placement.total_overlap_weight", 0.0, 6.0, step=0.5
    )
    cfg["placement"]["depth_std_weight"] = trial.suggest_float(
        "placement.depth_std_weight", 0.0, 4.0, step=0.25
    )
    cfg["placement"]["center_offset_weight"] = trial.suggest_float(
        "placement.center_offset_weight", 0.0, 1.5, step=0.1
    )
    cfg["placement"]["support_band_weight"] = trial.suggest_float(
        "placement.support_band_weight", 0.0, 3.0, step=0.25
    )
    cfg["placement"]["perspective_weight"] = trial.suggest_float(
        "placement.perspective_weight", 0.0, 4.0, step=0.25
    )
    cfg["placement"]["favor_empty_space_weight"] = trial.suggest_float(
        "placement.favor_empty_space_weight", 0.0, 3.0, step=0.25
    )
    cfg["placement"]["size_consistency_weight"] = trial.suggest_float(
        "placement.size_consistency_weight", 0.0, 4.0, step=0.25
    )
    cfg["placement"]["support_mask_penalty_weight"] = trial.suggest_float(
        "placement.support_mask_penalty_weight", 0.0, 10.0, step=0.5
    )
    cfg["placement"]["support_depth_mismatch_weight"] = trial.suggest_float(
        "placement.support_depth_mismatch_weight", 0.0, 4.0, step=0.25
    )
    cfg["placement"]["normal_penalty_weight"] = trial.suggest_float(
        "placement.normal_penalty_weight", 0.0, 5.0, step=0.25
    )
    cfg["placement"]["default_object_width_ratio"] = trial.suggest_float(
        "placement.default_object_width_ratio", 0.045, 0.065, step=0.005
    )
    cfg["placement"]["min_scale_ratio"] = trial.suggest_float(
        "placement.min_scale_ratio", 0.40, 0.50, step=0.05
    )
    cfg["placement"]["max_scale_ratio"] = trial.suggest_float(
        "placement.max_scale_ratio", 0.60, 0.75, step=0.05
    )
    cfg["placement"]["min_object_width_px"] = trial.suggest_int(
        "placement.min_object_width_px", 18, 24, step=2
    )
    cfg["placement"]["fallback_global_search"] = trial.suggest_categorical(
        "placement.fallback_global_search", [True, False]
    )
    cfg["placement"]["fallback_support_margin_ratio"] = trial.suggest_float(
        "placement.fallback_support_margin_ratio", 0.00, 0.18, step=0.02
    )

    # -------------------------
    # Collision: much looser possible
    # -------------------------
    cfg["collision"]["enabled"] = trial.suggest_categorical(
        "collision.enabled", [True]
    )
    cfg["collision"]["max_iou"] = trial.suggest_float(
        "collision.max_iou", 0.00, 0.30, step=0.02
    )
    cfg["collision"]["max_intersection_ratio_of_candidate"] = trial.suggest_float(
        "collision.max_intersection_ratio_of_candidate", 0.00, 0.45, step=0.03
    )
    cfg["collision"]["use_occupancy_map"] = trial.suggest_categorical(
        "collision.use_occupancy_map", [True, False]
    )
    cfg["collision"]["occupancy_blur_px"] = trial.suggest_int(
        "collision.occupancy_blur_px", 0, 17, step=2
    )
    cfg["collision"]["occupancy_threshold"] = trial.suggest_float(
        "collision.occupancy_threshold", 0.05, 0.80, step=0.05
    )
    cfg["collision"]["occupancy_penalty_weight"] = trial.suggest_float(
        "collision.occupancy_penalty_weight", 0.0, 12.0, step=0.5
    )
    cfg["collision"]["hard_occupancy_reject"] = trial.suggest_categorical(
        "collision.hard_occupancy_reject", [True, False]
    )

    # -------------------------
    # Lighting / relighting / occlusion toggles
    # -------------------------
    cfg["lighting"]["enabled"] = trial.suggest_categorical(
        "lighting.enabled", [True, False]
    )
    cfg["relighting"]["enabled"] = trial.suggest_categorical(
        "relighting.enabled", [True, False]
    )
    cfg["occlusion"]["enabled"] = trial.suggest_categorical(
        "occlusion.enabled", [True, False]
    )

    cfg["relighting"]["mean_match_strength"] = trial.suggest_float(
        "relighting.mean_match_strength", 0.0, 0.9, step=0.1
    )
    cfg["relighting"]["std_match_strength"] = trial.suggest_float(
        "relighting.std_match_strength", 0.0, 0.8, step=0.1
    )
    cfg["relighting"]["color_match_strength"] = trial.suggest_float(
        "relighting.color_match_strength", 0.0, 0.6, step=0.05
    )
    cfg["relighting"]["saturation_match_strength"] = trial.suggest_float(
        "relighting.saturation_match_strength", 0.0, 0.3, step=0.03
    )
    cfg["relighting"]["directional_shading_strength"] = trial.suggest_float(
        "relighting.directional_shading_strength", 0.0, 0.4, step=0.04
    )
    cfg["relighting"]["bottom_occlusion_strength"] = trial.suggest_float(
        "relighting.bottom_occlusion_strength", 0.0, 0.25, step=0.025
    )
    cfg["relighting"]["highlight_strength"] = trial.suggest_float(
        "relighting.highlight_strength", 0.0, 0.15, step=0.015
    )

    cfg["occlusion"]["feather_px"] = trial.suggest_float(
        "occlusion.feather_px", 0.0, 5.0, step=0.5
    )
    cfg["occlusion"]["depth_bias"] = trial.suggest_float(
        "occlusion.depth_bias", 0.0, 0.10, step=0.01
    )
    cfg["occlusion"]["foreground_hardness"] = trial.suggest_float(
        "occlusion.foreground_hardness", 0.2, 1.0, step=0.1
    )
    cfg["occlusion"]["object_depth_top_offset"] = trial.suggest_float(
        "occlusion.object_depth_top_offset", 0.0, 0.16, step=0.02
    )

    # -------------------------
    # Shadow: much wider
    # -------------------------
    cfg["shadow"]["enabled"] = trial.suggest_categorical(
        "shadow.enabled", [True, False]
    )
    cfg["shadow"]["contact_opacity"] = trial.suggest_float(
        "shadow.contact_opacity", 0.05, 0.55, step=0.05
    )
    cfg["shadow"]["contact_blur_px"] = trial.suggest_float(
        "shadow.contact_blur_px", 0.5, 6.0, step=0.5
    )
    cfg["shadow"]["cast_opacity"] = trial.suggest_float(
        "shadow.cast_opacity", 0.02, 0.35, step=0.03
    )
    cfg["shadow"]["cast_blur_px"] = trial.suggest_float(
        "shadow.cast_blur_px", 1.0, 18.0, step=1.0
    )
    cfg["shadow"]["cast_length_scale"] = trial.suggest_float(
        "shadow.cast_length_scale", 0.10, 1.40, step=0.10
    )
    cfg["shadow"]["squash_ratio"] = trial.suggest_float(
        "shadow.squash_ratio", 0.12, 0.65, step=0.03
    )
    cfg["shadow"]["shear_strength"] = trial.suggest_float(
        "shadow.shear_strength", 0.00, 0.50, step=0.05
    )
    cfg["shadow"]["shadow_color_mode"] = trial.suggest_categorical(
        "shadow.shadow_color_mode", ["surface_tinted", "black"]
    )
    cfg["shadow"]["ambient_occlusion_band_ratio"] = trial.suggest_float(
        "shadow.ambient_occlusion_band_ratio", 0.02, 0.18, step=0.02
    )
    cfg["shadow"]["shadow_softness_influence"] = trial.suggest_float(
        "shadow.shadow_softness_influence", 0.0, 0.8, step=0.1
    )

    # -------------------------
    # Profile overrides
    # -------------------------
    if profile == "strict_realism":
        cfg["collision"]["enabled"] = True
        cfg["collision"]["max_iou"] = min(cfg["collision"]["max_iou"], 0.08)
        cfg["collision"]["max_intersection_ratio_of_candidate"] = min(
            cfg["collision"]["max_intersection_ratio_of_candidate"], 0.16
        )
        cfg["collision"]["use_occupancy_map"] = True
        cfg["shadow"]["enabled"] = True
        cfg["lighting"]["enabled"] = True
        cfg["relighting"]["enabled"] = trial.suggest_categorical(
            "strict_realism.relighting.enabled", [True, False]
        )
        cfg["occlusion"]["enabled"] = trial.suggest_categorical(
            "strict_realism.occlusion.enabled", [True, False]
        )
        cfg["placement"]["default_object_width_ratio"] = min(
            cfg["placement"]["default_object_width_ratio"], 0.12
        )

    elif profile == "balanced":
        pass

    elif profile == "aggressive_placement":
        cfg["collision"]["enabled"] = True
        cfg["collision"]["max_iou"] = max(cfg["collision"]["max_iou"], 0.12)
        cfg["collision"]["max_intersection_ratio_of_candidate"] = max(
            cfg["collision"]["max_intersection_ratio_of_candidate"], 0.20
        )
        cfg["placement"]["default_object_width_ratio"] = max(
            cfg["placement"]["default_object_width_ratio"], 0.065
        )
        cfg["placement"]["fallback_global_search"] = True
        cfg["placement"]["support_mask_penalty_weight"] = min(
            cfg["placement"]["support_mask_penalty_weight"], 4.0
        )

    elif profile == "loose_extraction":
        cfg["extraction"]["combine_mode"] = trial.suggest_categorical(
            "loose_extraction.combine_mode",
            ["union", "sam_prefer", "primary_prefer"],
        )
        cfg["extraction"]["component_max_gap_px"] = max(
            cfg["extraction"]["component_max_gap_px"], 120
        )
        cfg["extraction"]["max_border_touch_ratio"] = max(
            cfg["extraction"]["max_border_touch_ratio"], 0.30
        )

    elif profile == "dramatic_shadow":
        cfg["shadow"]["enabled"] = True
        cfg["lighting"]["enabled"] = True
        cfg["shadow"]["cast_opacity"] = max(cfg["shadow"]["cast_opacity"], 0.16)
        cfg["shadow"]["cast_length_scale"] = max(cfg["shadow"]["cast_length_scale"], 0.70)
        cfg["shadow"]["shear_strength"] = max(cfg["shadow"]["shear_strength"], 0.18)
        cfg["shadow"]["contact_opacity"] = max(cfg["shadow"]["contact_opacity"], 0.22)

    cfg["placement"]["attempt_index"] = 0
    return cfg

def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_trial_dir(root_dir: Path, trial_number: int) -> Path:
    return root_dir / f"trial_{trial_number:03d}"

def run_optuna_sweep(
    *,
    scene_path: str,
    object_image_path: str,
    output_root: str,
    config_path: Optional[str],
    device: torch.device,
    no_sam: bool,
    n_trials: int,
    seed: int,
    study_name: str,
) -> None:
    base_cfg = load_config(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path(output_root) / f"{study_name}_{timestamp}"
    root_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )

    run_manifest = {
        "study_name": study_name,
        "timestamp": timestamp,
        "scene": str(scene_path),
        "object_image": str(object_image_path),
        "config_path": str(config_path) if config_path else None,
        "device": str(device),
        "n_trials": n_trials,
        "seed": seed,
    }
    save_json(root_dir / "run_manifest.json", run_manifest)
    save_yaml(root_dir / "base_config.yaml", base_cfg)

    for _ in range(n_trials):
        trial = study.ask()
        trial_cfg = sample_optuna_config(trial, base_cfg)
        trial_dir = build_trial_dir(root_dir, trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config_file = trial_dir / "config.yaml"
        output_file = trial_dir / "output.jpg"
        debug_file = trial_dir / "debug_overlay.jpg"
        meta_file = trial_dir / "meta.json"

        save_yaml(config_file, trial_cfg)

        print("=" * 80)
        print(f"Running trial {trial.number}/{n_trials - 1}")
        print(f"Trial directory: {trial_dir}")

        result_meta = run_pipeline(
            scene_path=scene_path,
            object_image_path=object_image_path,
            output_path=output_file,
            device=device,
            cfg=trial_cfg,
            debug_overlay_path=debug_file,
            no_sam=no_sam,
        )

        trial_meta = {
            "trial_number": trial.number,
            "params": trial.params,
            "result": result_meta,
            "config_file": str(config_file),
            "output_file": str(output_file),
            "debug_overlay_file": str(debug_file),
        }
        save_json(meta_file, trial_meta)

        # Manual browsing workflow: all trials are considered valid.
        # Dummy objective lets Optuna record the trial cleanly.
        study.tell(trial, float(trial.number))

    summary = []
    for t in study.trials:
        summary.append({
            "trial_number": t.number,
            "params": t.params,
            "value": t.value,
        })
    save_json(root_dir / "study_trials.json", summary)

    print("=" * 80)
    print(f"Completed {n_trials} trials.")
    print(f"Results saved under: {root_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Insert a real reference object into a scene using local open models.")
    parser.add_argument("--scene", required=True, help="Path to target scene image")
    parser.add_argument("--object-image", required=True, help="Path to reference object image")
    parser.add_argument("--output", required=True, help="Path to output image or output root dir for Optuna mode")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--debug-overlay", default=None, help="Optional path to save placement overlay image")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM2 even if installed")
    parser.add_argument("--config", default="config_generalized_v2.yaml", help="Path to YAML config file")
    parser.add_argument("--attempt", type=int, default=None, help="Placement attempt index override")

    # New optional batch/Optuna args
    parser.add_argument("--optuna-trials", type=int, default=0, help="If > 0, run an Optuna sweep with this many trials")
    parser.add_argument("--optuna-seed", type=int, default=42, help="Random seed for Optuna sampling")
    parser.add_argument("--optuna-study-name", default="image_config_sweep", help="Study name for Optuna run")

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.attempt is not None:
        cfg["placement"]["attempt_index"] = int(args.attempt)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )

    if args.optuna_trials and args.optuna_trials > 0:
        run_optuna_sweep(
            scene_path=args.scene,
            object_image_path=args.object_image,
            output_root=args.output,
            config_path=args.config,
            device=device,
            no_sam=args.no_sam,
            n_trials=int(args.optuna_trials),
            seed=int(args.optuna_seed),
            study_name=str(args.optuna_study_name),
        )
        return

    run_pipeline(
        scene_path=args.scene,
        object_image_path=args.object_image,
        output_path=args.output,
        device=device,
        cfg=cfg,
        debug_overlay_path=args.debug_overlay,
        no_sam=args.no_sam,
    )


if __name__ == "__main__":
    main()