from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
    pipeline,
)

try:
    from diffusers import MarigoldNormalsPipeline  # type: ignore

    _MARIGOLD_AVAILABLE = True
except Exception:
    MarigoldNormalsPipeline = None  # type: ignore
    _MARIGOLD_AVAILABLE = False


# -----------------------------------------------------------------------------
# Default config
# -----------------------------------------------------------------------------

DEFAULT_SCENE_UNDERSTANDING_CFG: Dict[str, Any] = {
    "models": {
        "detector_id": "IDEA-Research/grounding-dino-tiny",
        "depth_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "oneformer_id": "shi-labs/oneformer_ade20k_swin_large",
        "normals_id": "prs-eth/marigold-normals-v1-1",
    },
    "runtime": {
        "device": "cuda",
        "use_fp16": True,
    },
    "input": {
        "max_side": 1280,
    },
    "support_instances": {
        "enabled_sources": ["detector", "panoptic"],
        "canonical_labels": [
            "countertop",
            "table",
            "desk",
            "shelf",
            "bench",
            "nightstand",
            "dresser",
            "cabinet",
            "windowsill",
            "floor",
        ],
        "detector_labels": [
            "countertop",
            "kitchen counter",
            "table",
            "desk",
            "shelf",
            "bookshelf",
            "bench",
            "nightstand",
            "dresser",
            "cabinet top",
            "windowsill",
            "floor",
        ],
        "box_threshold": 0.22,
        "text_threshold": 0.18,
        "detector_nms_iou": 0.45,
        "max_detector_instances": 16,
        "max_panoptic_instances": 20,
        "max_instances_total": 20,
        "min_instance_area_ratio": 0.0020,
        "max_instance_area_ratio": 0.35,
        "reject_if_touches_left_right": True,
        "reject_if_touches_top_bottom": True,
        "max_border_touch_count": 1,
        "min_box_width_ratio": 0.06,
        "min_box_height_ratio": 0.03,
        "panoptic_label_allowlist": [
            "table",
            "desk",
            "counter",
            "countertop",
            "shelf",
            "cabinet",
            "bench",
            "nightstand",
            "dresser",
            "bookcase",
            "floor",
            "window",
            "windowsill",
        ],
        "panoptic_label_map": {
            "table": "table",
            "desk": "desk",
            "counter": "countertop",
            "countertop": "countertop",
            "shelf": "shelf",
            "bookcase": "shelf",
            "cabinet": "cabinet",
            "bench": "bench",
            "nightstand": "nightstand",
            "dresser": "dresser",
            "floor": "floor",
            "window": "windowsill",
            "windowsill": "windowsill",
        },
    },
    "segmentation": {
        "enabled": True,
        "task": "panoptic",
    },
    "normals": {
        "enabled": True,
    },
    "instance_refinement": {
        "close_px": 4,
        "open_px": 2,
        "min_component_area_ratio": 0.0015,
        "max_component_area_ratio": 0.35,
    },
    "top_surface": {
        "min_abs_nz": 0.82,
        "max_abs_ny": 0.42,
        "min_pixels": 240,
        "close_px": 3,
        "open_px": 2,
        "top_band_ratio": 0.22,
        "eye_level_band_ratio": 0.16,
        "horizon_y_ratio": 0.46,
        "min_top_band_fill_below_eye": 0.10,
        "min_top_band_fill_eye_level": 0.05,
        "min_visible_ratio_below_eye": 0.08,
        "max_visible_ratio_eye_level": 0.28,
        "edge_case_min_aspect_ratio": 1.7,
        "edge_case_min_abs_nz": 0.88,
        "edge_case_max_abs_ny": 0.48,
        "fit_quad": True,
        "polygon_epsilon_ratio": 0.02,
        "reject_above_eye": True,
    },
    "obstacle_detection": {
        "edge_threshold": 0.18,
        "depth_threshold": 0.12,
        "min_component_area_ratio_of_top": 0.010,
        "close_px": 2,
        "open_px": 1,
        "subtract_panoptic_objects": True,
        "ignore_same_support_labels": [
            "table",
            "desk",
            "counter",
            "countertop",
            "shelf",
            "cabinet",
            "bench",
            "nightstand",
            "dresser",
            "floor",
            "window",
            "windowsill",
        ],
    },
    "free_space": {
        "min_free_area_ratio_of_top": 0.12,
        "erode_margin_ratio": 0.035,
        "min_component_area_ratio_of_top": 0.040,
        "prefer_centered_components": True,
    },
    "support_scoring": {
        "min_final_score": 0.58,
        "horizontal_weight": 0.32,
        "flatness_weight": 0.16,
        "free_area_weight": 0.30,
        "top_visibility_weight": 0.14,
        "occupancy_penalty_weight": 0.18,
        "border_penalty_weight": 0.10,
        "thin_penalty_weight": 0.08,
        "floor_penalty": 0.10,
        "cabinet_bonus_if_top_good": 0.04,
        "counter_bonus": 0.05,
    },
    "scene_priors": {
        "enabled": True,
        "camera_height_m": 1.5,
        "scene_type_hint": "",
    },
    "placement_context": {
        "enabled": True,
        "mask_dilate_px": 16,
        "crop_expand_ratio": 0.18,
        "max_crop_expand_px": 96,
    },
    "output": {
        "save_debug": True,
        "save_json": True,
    },
}


# -----------------------------------------------------------------------------
# Lazy globals
# -----------------------------------------------------------------------------

_DET_PROCESSOR = None
_DET_MODEL = None
_DEPTH_PIPE = None
_ONEFORMER_PROCESSOR = None
_ONEFORMER_MODEL = None
_NORMALS_PIPE = None


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

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

    def clamp(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x0=max(0.0, min(self.x0, width - 1)),
            y0=max(0.0, min(self.y0, height - 1)),
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            score=self.score,
            label=self.label,
        )

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x0)),
            int(round(self.y0)),
            int(round(self.x1)),
            int(round(self.y1)),
        )


@dataclass
class SupportInstance:
    id: str
    label: str
    source: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    mask_area_px: int
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupportSurface:
    id: str
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    support_mode: str
    region_area_px: int
    contact_band_xyxy: Tuple[int, int, int, int]
    visible_top_polygon_xy: List[List[int]]
    plane_quad_xy: Optional[List[List[float]]]
    front_edge_xy: Optional[List[List[float]]]
    back_edge_xy: Optional[List[List[float]]]
    homography_unit_to_img: Optional[List[List[float]]]
    homography_img_to_unit: Optional[List[List[float]]]
    depth_stats: Dict[str, float] = field(default_factory=dict)
    normal_stats: Dict[str, float] = field(default_factory=dict)
    geometry_scores: Dict[str, float] = field(default_factory=dict)
    vlm_verification: Dict[str, Any] = field(default_factory=dict)
    usable_area_ratio: float = 0.0
    occupied_area_ratio: float = 0.0
    border_touch_ratio: float = 0.0
    prior_dimensions_m: Dict[str, Any] = field(default_factory=dict)
    prior_scale: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneUnderstandingResult:
    image_size: Tuple[int, int]
    supports: List[SupportSurface]
    depth_path: Optional[str]
    normals_path: Optional[str]
    support_json_path: Optional[str]
    debug_dir: Optional[str]
    scene_priors: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_update(DEFAULT_SCENE_UNDERSTANDING_CFG, user_cfg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_map(arr: np.ndarray, percentile_hi: float = 99.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    hi = float(np.percentile(arr, percentile_hi))
    lo = float(np.percentile(arr, 1.0))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _binary_clean(mask: np.ndarray, open_px: int = 2, close_px: int = 3) -> np.ndarray:
    out = (mask > 0).astype(np.uint8) * 255
    if close_px > 0:
        k = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_px > 0:
        k = np.ones((open_px * 2 + 1, open_px * 2 + 1), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return (mask > 0).astype(np.uint8) * 255
    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best_idx, 255, 0).astype(np.uint8)


def _mask_area(mask_u8: np.ndarray) -> int:
    return int((mask_u8 > 0).sum())


def _mask_bbox(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _dilate_mask(mask_u8: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask_u8.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask_u8, k)


def _expand_box_xyxy(box_xyxy: Tuple[int, int, int, int], image_size: Tuple[int, int], pad_px: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_xyxy
    w, h = image_size
    return (
        max(0, x0 - pad_px),
        max(0, y0 - pad_px),
        min(w, x1 + pad_px),
        min(h, y1 + pad_px),
    )


def _build_refine_region(top_mask: np.ndarray, best_patch_mask: np.ndarray, image_size: Tuple[int, int], cfg: dict) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    ctx = cfg.get("placement_context", {}) or {}
    base = cv2.bitwise_or(top_mask, best_patch_mask)
    dilated = _dilate_mask(base, int(ctx.get("mask_dilate_px", 16)))
    bbox = _mask_bbox(dilated)
    if bbox is None:
        bbox = _mask_bbox(best_patch_mask) or _mask_bbox(top_mask) or (0, 0, image_size[0], image_size[1])
    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    pad = int(round(min(max(bw, bh) * float(ctx.get("crop_expand_ratio", 0.18)), float(ctx.get("max_crop_expand_px", 96)))))
    crop = _expand_box_xyxy(bbox, image_size, pad)
    return dilated, crop


def _mask_to_polygon(mask_u8: np.ndarray, epsilon_ratio: float) -> List[List[int]]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return [[int(p[0][0]), int(p[0][1])] for p in approx]


def _save_gray(arr: np.ndarray, out_path: Path) -> None:
    arr_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(out_path)


def _save_color_overlay(base_img: Image.Image, masks: List[np.ndarray], out_path: Path) -> None:
    rgb = _pil_to_np_rgb(base_img).copy()
    overlay = rgb.copy()
    colors = [
        np.array([255, 0, 0], dtype=np.uint8),
        np.array([0, 255, 0], dtype=np.uint8),
        np.array([0, 128, 255], dtype=np.uint8),
        np.array([255, 200, 0], dtype=np.uint8),
        np.array([180, 0, 255], dtype=np.uint8),
        np.array([0, 255, 255], dtype=np.uint8),
    ]
    for i, mask in enumerate(masks):
        m = mask > 0
        color = colors[i % len(colors)]
        overlay[m] = (0.55 * overlay[m] + 0.45 * color).astype(np.uint8)
    _np_rgb_to_pil(overlay).save(out_path)


def _crop_box_with_pad(
    box_xyxy: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_xyxy
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    px = int(round(w * pad_ratio))
    py = int(round(h * pad_ratio))
    W, H = image_size
    return (
        max(0, x0 - px),
        max(0, y0 - py),
        min(W, x1 + px),
        min(H, y1 + py),
    )


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    return iw * ih


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    inter = _intersection_area(a, b)
    union = a.area() + b.area() - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _nms_boxes(boxes: List[BoundingBox], iou_thr: float) -> List[BoundingBox]:
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    out: List[BoundingBox] = []
    for box in boxes:
        if all(_iou(box, prev) < iou_thr for prev in out):
            out.append(box)
    return out


def _maybe_resize_for_models(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale == 1.0:
        return image, 1.0
    new_w = max(32, int(round(w * scale / 8) * 8))
    new_h = max(32, int(round(h * scale / 8) * 8))
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def _scale_box(box: BoundingBox, inv_scale: float) -> BoundingBox:
    return BoundingBox(
        x0=box.x0 * inv_scale,
        y0=box.y0 * inv_scale,
        x1=box.x1 * inv_scale,
        y1=box.y1 * inv_scale,
        score=box.score,
        label=box.label,
    )


def _count_touched_borders(box_xyxy: Tuple[int, int, int, int], image_size: Tuple[int, int], tol: int = 2) -> int:
    x0, y0, x1, y1 = box_xyxy
    W, H = image_size
    count = 0
    if x0 <= tol:
        count += 1
    if y0 <= tol:
        count += 1
    if x1 >= W - tol:
        count += 1
    if y1 >= H - tol:
        count += 1
    return count


def _fit_support_quad_and_homography(mask_u8: np.ndarray) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 40:
        return None, None, None

    rows: List[Tuple[int, int, int]] = []
    for yy in range(mask_u8.shape[0]):
        row = np.where(mask_u8[yy] > 0)[0]
        if len(row) < 2:
            continue
        rows.append((yy, int(row.min()), int(row.max())))

    if len(rows) < 8:
        return None, None, None

    def fit_edge(rows_in: List[Tuple[int, int, int]], side: str) -> Tuple[float, float]:
        pts = []
        for y, xl, xr in rows_in:
            x = xl if side == "left" else xr
            pts.append((x, y))
        pts_np = np.array(pts, dtype=np.float32)
        line = cv2.fitLine(pts_np, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = [float(np.asarray(v).reshape(-1)[0]) for v in line]
        if abs(vy) < 1e-6:
            return 0.0, x0
        m = vx / vy
        b = x0 - m * y0
        return m, b

    n_band = max(3, len(rows) // 8)
    top_rows = rows[:n_band]
    bot_rows = rows[-n_band:]

    left_m, left_b = fit_edge(rows, "left")
    right_m, right_b = fit_edge(rows, "right")

    top_y = float(np.median([y for y, _, _ in top_rows]))
    bot_y = float(np.median([y for y, _, _ in bot_rows]))

    quad = np.array(
        [
            [left_m * top_y + left_b, top_y],
            [right_m * top_y + right_b, top_y],
            [right_m * bot_y + right_b, bot_y],
            [left_m * bot_y + left_b, bot_y],
        ],
        dtype=np.float32,
    )

    unit = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    H_unit_to_img = cv2.getPerspectiveTransform(unit, quad)
    H_img_to_unit = cv2.getPerspectiveTransform(quad, unit)
    return quad, H_unit_to_img, H_img_to_unit


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def _choose_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _get_detector(device: torch.device, cfg: dict):
    global _DET_PROCESSOR, _DET_MODEL
    if _DET_PROCESSOR is None or _DET_MODEL is None:
        model_id = cfg["models"]["detector_id"]
        token = os.getenv("HF_TOKEN")
        _DET_PROCESSOR = AutoProcessor.from_pretrained(model_id, token=token)
        _DET_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, token=token)
        _DET_MODEL.to(device)
        _DET_MODEL.eval()
    return _DET_PROCESSOR, _DET_MODEL


def _get_depth_pipe(device: torch.device, cfg: dict):
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        device_index = 0 if device.type == "cuda" else -1
        token = os.getenv("HF_TOKEN")
        _DEPTH_PIPE = pipeline(
            task="depth-estimation",
            model=cfg["models"]["depth_id"],
            device=device_index,
            token=token,
        )
    return _DEPTH_PIPE


def _get_oneformer(device: torch.device, cfg: dict):
    global _ONEFORMER_PROCESSOR, _ONEFORMER_MODEL
    if _ONEFORMER_PROCESSOR is None or _ONEFORMER_MODEL is None:
        model_id = cfg["models"]["oneformer_id"]
        token = os.getenv("HF_TOKEN")
        _ONEFORMER_PROCESSOR = OneFormerProcessor.from_pretrained(model_id, token=token)
        _ONEFORMER_MODEL = OneFormerForUniversalSegmentation.from_pretrained(model_id, token=token)
        _ONEFORMER_MODEL.to(device)
        _ONEFORMER_MODEL.eval()
    return _ONEFORMER_PROCESSOR, _ONEFORMER_MODEL


def _get_normals_pipe(device: torch.device, cfg: dict):
    global _NORMALS_PIPE
    if not _MARIGOLD_AVAILABLE:
        return None
    if _NORMALS_PIPE is None:
        token = os.getenv("HF_TOKEN")
        dtype = torch.float16 if (device.type == "cuda" and cfg["runtime"].get("use_fp16", True)) else torch.float32
        _NORMALS_PIPE = MarigoldNormalsPipeline.from_pretrained(
            cfg["models"]["normals_id"],
            torch_dtype=dtype,
            token=token,
        )
        _NORMALS_PIPE.to(device)
    return _NORMALS_PIPE


# -----------------------------------------------------------------------------
# Core inference
# -----------------------------------------------------------------------------

def _detect_support_boxes(image: Image.Image, device: torch.device, cfg: dict) -> List[BoundingBox]:
    inst_cfg = cfg["support_instances"]
    labels = [x.strip() for x in inst_cfg["detector_labels"] if str(x).strip()]
    if not labels:
        return []

    resized, scale = _maybe_resize_for_models(image, int(cfg["input"]["max_side"]))
    inv_scale = 1.0 / scale

    processor, model = _get_detector(device, cfg)
    text_labels = [labels]
    inputs = processor(images=resized, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=float(inst_cfg["box_threshold"]),
        text_threshold=float(inst_cfg["text_threshold"]),
        target_sizes=[resized.size[::-1]],
    )
    if not results:
        return []

    image_area = float(image.size[0] * image.size[1])
    detections: List[BoundingBox] = []

    result = results[0]
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels_out = result.get("labels", [])

    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        score = float(scores[i].item()) if i < len(scores) else 1.0
        label = str(labels_out[i]) if i < len(labels_out) else ""
        b = _scale_box(BoundingBox(x0, y0, x1, y1, score=score, label=label), inv_scale).clamp(*image.size)
        area_ratio = b.area() / image_area
        if area_ratio < float(inst_cfg["min_instance_area_ratio"]):
            continue
        if area_ratio > float(inst_cfg["max_instance_area_ratio"]):
            continue
        if b.width() < image.size[0] * float(inst_cfg["min_box_width_ratio"]):
            continue
        if b.height() < image.size[1] * float(inst_cfg["min_box_height_ratio"]):
            continue
        detections.append(b)

    detections = _nms_boxes(detections, float(inst_cfg["detector_nms_iou"]))
    return detections[: int(inst_cfg["max_detector_instances"])]


def _estimate_depth_map(image: Image.Image, device: torch.device, cfg: dict) -> np.ndarray:
    pipe = _get_depth_pipe(device, cfg)
    out = pipe(image)
    depth_img = out["depth"]
    if not isinstance(depth_img, Image.Image):
        depth_img = Image.fromarray(np.array(depth_img))
    depth_img = depth_img.resize(image.size, Image.BILINEAR)
    depth = np.array(depth_img).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = _normalize_map(depth, percentile_hi=99.0)

    h = depth.shape[0]
    top = float(np.median(depth[: max(1, h // 5), :]))
    bottom = float(np.median(depth[-max(1, h // 5):, :]))
    if bottom < top:
        depth = 1.0 - depth
    return depth.astype(np.float32)


def _estimate_normals_map(image: Image.Image, device: torch.device, cfg: dict) -> Optional[np.ndarray]:
    if not bool(cfg["normals"].get("enabled", True)):
        return None
    pipe = _get_normals_pipe(device, cfg)
    if pipe is None:
        return None

    try:
        out = pipe(image)
        pred = getattr(out, "prediction", None)
        if pred is None and isinstance(out, dict):
            pred = out.get("prediction")
        if pred is None:
            return None

        if torch.is_tensor(pred):
            normals = pred.detach().float().cpu().numpy()
        else:
            normals = np.array(pred, dtype=np.float32)

        if normals.ndim == 4:
            normals = normals[0]
        if normals.ndim == 3 and normals.shape[0] == 3:
            normals = np.transpose(normals, (1, 2, 0))

        normals = cv2.resize(normals, image.size, interpolation=cv2.INTER_LINEAR)
        if normals.min() >= 0.0 and normals.max() <= 1.0:
            normals = normals * 2.0 - 1.0

        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / np.clip(norm, 1e-6, None)
        return normals.astype(np.float32)
    except Exception:
        return None


def _run_oneformer_panoptic(image: Image.Image, device: torch.device, cfg: dict) -> Optional[Dict[str, Any]]:
    if not bool(cfg["segmentation"].get("enabled", True)):
        return None

    processor, model = _get_oneformer(device, cfg)
    task = str(cfg["segmentation"].get("task", "panoptic"))
    inputs = processor(images=image, task_inputs=[task], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    processed = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
    )
    if not processed:
        return None

    seg = processed[0]["segmentation"]
    if torch.is_tensor(seg):
        seg = seg.detach().cpu().numpy()

    segments_info = processed[0].get("segments_info", [])
    id2label = getattr(model.config, "id2label", {}) or {}
    for item in segments_info:
        lid = int(item.get("label_id", -1))
        item["label_name"] = str(id2label.get(lid, f"label_{lid}")).lower()

    return {
        "segmentation": seg.astype(np.int32),
        "segments_info": segments_info,
    }


# -----------------------------------------------------------------------------
# Structure maps
# -----------------------------------------------------------------------------

def _compute_structure_maps(rgb: np.ndarray, depth_map: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mag = cv2.GaussianBlur(grad_mag, (0, 0), 0.8)
    grad_mag = _normalize_map(grad_mag, percentile_hi=98.0)

    if depth_map is not None:
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = np.sqrt(dx * dx + dy * dy)
        depth_grad = cv2.GaussianBlur(depth_grad, (0, 0), 0.8)
        depth_grad = _normalize_map(depth_grad, percentile_hi=98.0)
    else:
        depth_grad = np.zeros_like(gray, dtype=np.float32)

    return {
        "gray": gray,
        "grad_mag": grad_mag,
        "depth_grad": depth_grad,
    }


def _normal_stats_for_mask(normals_map: Optional[np.ndarray], mask_u8: np.ndarray) -> Dict[str, float]:
    if normals_map is None:
        return {}
    m = mask_u8 > 0
    if not m.any():
        return {}
    vals = normals_map[m]
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    abs_mean = np.mean(np.abs(vals), axis=0)
    return {
        "mean_nx": float(mean[0]),
        "mean_ny": float(mean[1]),
        "mean_nz": float(mean[2]),
        "std_nx": float(std[0]),
        "std_ny": float(std[1]),
        "std_nz": float(std[2]),
        "abs_mean_nx": float(abs_mean[0]),
        "abs_mean_ny": float(abs_mean[1]),
        "abs_mean_nz": float(abs_mean[2]),
    }


def _depth_stats_for_mask(depth_map: Optional[np.ndarray], mask_u8: np.ndarray) -> Dict[str, float]:
    if depth_map is None:
        return {}
    m = mask_u8 > 0
    if not m.any():
        return {}
    vals = depth_map[m]
    return {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


# -----------------------------------------------------------------------------
# Support instance generation
# -----------------------------------------------------------------------------

def _canonicalize_label(raw_label: str, cfg: dict) -> str:
    label = str(raw_label or "").strip().lower()
    label_map = cfg["support_instances"].get("panoptic_label_map", {}) or {}
    if label in label_map:
        return str(label_map[label])
    if "kitchen counter" in label:
        return "countertop"
    if "counter" in label:
        return "countertop"
    if "bookshelf" in label:
        return "shelf"
    if "cabinet top" in label:
        return "cabinet"
    if "window" in label:
        return "windowsill"
    return label


def _extract_panoptic_support_instances(
    panoptic: Optional[Dict[str, Any]],
    image_size: Tuple[int, int],
    cfg: dict,
) -> List[Tuple[SupportInstance, np.ndarray]]:
    if panoptic is None:
        return []

    seg = panoptic["segmentation"]
    segments_info = panoptic.get("segments_info", [])
    inst_cfg = cfg["support_instances"]
    allow = {str(x).lower() for x in inst_cfg.get("panoptic_label_allowlist", [])}
    W, H = image_size
    image_area = float(W * H)

    out: List[Tuple[SupportInstance, np.ndarray]] = []
    for idx, info in enumerate(segments_info):
        sid = info.get("id")
        raw_name = str(info.get("label_name", "")).lower()
        canon = _canonicalize_label(raw_name, cfg)
        if raw_name not in allow and canon not in allow:
            continue

        mask = np.where(seg == sid, 255, 0).astype(np.uint8)
        mask = _binary_clean(
            mask,
            open_px=int(cfg["instance_refinement"]["open_px"]),
            close_px=int(cfg["instance_refinement"]["close_px"]),
        )
        mask = _largest_component(mask)
        area = _mask_area(mask)
        if area <= 0:
            continue

        area_ratio = area / image_area
        if area_ratio < float(inst_cfg["min_instance_area_ratio"]):
            continue
        if area_ratio > float(inst_cfg["max_instance_area_ratio"]):
            continue

        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < int(W * float(inst_cfg["min_box_width_ratio"])):
            continue
        if (y1 - y0) < int(H * float(inst_cfg["min_box_height_ratio"])):
            continue

        touch_count = _count_touched_borders(bbox, image_size)
        if touch_count > int(inst_cfg["max_border_touch_count"]):
            continue

        score = float(info.get("score", 0.65))
        inst = SupportInstance(
            id=f"inst_panoptic_{idx:02d}",
            label=canon,
            source="panoptic",
            confidence=score,
            box_xyxy=bbox,
            mask_area_px=area,
            debug={"raw_label": raw_name, "touch_count": touch_count},
        )
        out.append((inst, mask))

    out = sorted(out, key=lambda x: x[0].confidence, reverse=True)
    return out[: int(inst_cfg["max_panoptic_instances"])]


def _extract_detector_support_instances(
    image: Image.Image,
    device: torch.device,
    cfg: dict,
) -> List[Tuple[SupportInstance, np.ndarray]]:
    boxes = _detect_support_boxes(image, device, cfg)
    W, H = image.size
    inst_cfg = cfg["support_instances"]

    out: List[Tuple[SupportInstance, np.ndarray]] = []
    for idx, b in enumerate(boxes):
        bbox = b.to_int_tuple()
        touch_count = _count_touched_borders(bbox, image.size)
        if touch_count > int(inst_cfg["max_border_touch_count"]):
            continue

        label = _canonicalize_label(b.label, cfg)
        x0, y0, x1, y1 = bbox
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255

        inst = SupportInstance(
            id=f"inst_detector_{idx:02d}",
            label=label,
            source="detector",
            confidence=float(b.score),
            box_xyxy=bbox,
            mask_area_px=_mask_area(mask),
            debug={"touch_count": touch_count},
        )
        out.append((inst, mask))
    return out


def _instance_nms(
    items: List[Tuple[SupportInstance, np.ndarray]],
    iou_thr: float,
) -> List[Tuple[SupportInstance, np.ndarray]]:
    boxes = [
        BoundingBox(*inst.box_xyxy, score=inst.confidence, label=inst.label)
        for inst, _ in items
    ]
    kept: List[Tuple[SupportInstance, np.ndarray]] = []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].score, reverse=True)
    for idx in order:
        box = boxes[idx]
        if all(_iou(box, BoundingBox(*k[0].box_xyxy, score=k[0].confidence, label=k[0].label)) < iou_thr for k in kept):
            kept.append(items[idx])
    return kept


def _generate_support_instances(
    image: Image.Image,
    panoptic: Optional[Dict[str, Any]],
    device: torch.device,
    cfg: dict,
) -> List[Tuple[SupportInstance, np.ndarray]]:
    sources = set(cfg["support_instances"].get("enabled_sources", []))
    all_items: List[Tuple[SupportInstance, np.ndarray]] = []

    if "detector" in sources:
        all_items.extend(_extract_detector_support_instances(image, device, cfg))
    if "panoptic" in sources:
        all_items.extend(_extract_panoptic_support_instances(panoptic, image.size, cfg))

    all_items = _instance_nms(all_items, float(cfg["support_instances"]["detector_nms_iou"]))
    return all_items[: int(cfg["support_instances"]["max_instances_total"])]


# -----------------------------------------------------------------------------
# Top-surface extraction
# -----------------------------------------------------------------------------

def _support_view_profile(box_xyxy: Tuple[int, int, int, int], image_size: Tuple[int, int], cfg: dict) -> str:
    _, H = image_size
    _, y0, _, y1 = box_xyxy
    cy = 0.5 * (y0 + y1) / max(1.0, float(H))

    top_cfg = cfg["top_surface"]
    horizon = float(top_cfg["horizon_y_ratio"])
    band = float(top_cfg["eye_level_band_ratio"])

    if cy < (horizon - band):
        return "above_eye"
    if cy <= (horizon + band):
        return "eye_level"
    return "below_eye"


def _extract_visible_top_mask(
    instance_mask: np.ndarray,
    instance_box: Tuple[int, int, int, int],
    normals_map: Optional[np.ndarray],
    depth_map: Optional[np.ndarray],
    image_size: Tuple[int, int],
    cfg: dict,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if normals_map is None:
        return None, {"reject_reason": "missing_normals"}

    top_cfg = cfg["top_surface"]
    m = instance_mask > 0
    if not m.any():
        return None, {"reject_reason": "empty_instance"}

    abs_ny = np.abs(normals_map[:, :, 1])
    abs_nz = np.abs(normals_map[:, :, 2])

    top_facing = (
        (abs_nz >= float(top_cfg["min_abs_nz"])) &
        (abs_ny <= float(top_cfg["max_abs_ny"])) &
        m
    ).astype(np.uint8) * 255

    top_facing = _binary_clean(
        top_facing,
        open_px=int(top_cfg["open_px"]),
        close_px=int(top_cfg["close_px"]),
    )
    if _mask_area(top_facing) < int(top_cfg["min_pixels"]):
        return None, {"reject_reason": "not_enough_top_pixels"}

    x0, y0, x1, y1 = instance_box
    h = max(1, y1 - y0)
    w = max(1, x1 - x0)

    view = _support_view_profile(instance_box, image_size, cfg)
    if view == "above_eye" and bool(top_cfg.get("reject_above_eye", True)):
        return None, {"reject_reason": "above_eye"}

    band_h = max(3, int(round(h * float(top_cfg["top_band_ratio"]))))
    top_band_y1 = min(y1, y0 + band_h)

    band_mask = np.zeros_like(top_facing, dtype=np.uint8)
    if view == "below_eye":
        band_mask[y0:top_band_y1, x0:x1] = 255
    else:
        center_h = max(3, int(round(h * float(top_cfg["eye_level_band_ratio"]))))
        cy = int(round(0.5 * (y0 + y1)))
        yy0 = max(y0, cy - center_h)
        yy1 = min(y1, cy + center_h)
        band_mask[yy0:yy1, x0:x1] = 255

    top_in_band = np.logical_and(top_facing > 0, band_mask > 0)
    band_area = max(1, int((band_mask > 0).sum()))
    band_fill = float(top_in_band.sum()) / float(band_area)

    visible_ratio = float((top_facing > 0).sum()) / max(1.0, float((instance_mask > 0).sum()))
    aspect = w / max(1.0, float(h))
    abs_nz_mean = float(abs_nz[top_facing > 0].mean()) if (top_facing > 0).any() else 0.0
    abs_ny_mean = float(abs_ny[top_facing > 0].mean()) if (top_facing > 0).any() else 1.0

    edge_case = bool(
        view == "eye_level"
        and aspect >= float(top_cfg["edge_case_min_aspect_ratio"])
        and abs_nz_mean >= float(top_cfg["edge_case_min_abs_nz"])
        and abs_ny_mean <= float(top_cfg["edge_case_max_abs_ny"])
        and visible_ratio <= float(top_cfg["max_visible_ratio_eye_level"])
    )

    if view == "below_eye":
        if band_fill < float(top_cfg["min_top_band_fill_below_eye"]):
            return None, {"reject_reason": "below_eye_missing_top_band", "band_fill": band_fill}
        if visible_ratio < float(top_cfg["min_visible_ratio_below_eye"]):
            return None, {"reject_reason": "below_eye_not_enough_visible_top", "visible_ratio": visible_ratio}
    elif view == "eye_level":
        if band_fill < float(top_cfg["min_top_band_fill_eye_level"]) and not edge_case:
            return None, {"reject_reason": "eye_level_missing_edge_or_top", "band_fill": band_fill}

    top_mask = top_facing.copy()
    top_mask = _largest_component(top_mask)
    if _mask_area(top_mask) < int(top_cfg["min_pixels"]):
        return None, {"reject_reason": "top_component_too_small"}

    depth_stats = _depth_stats_for_mask(depth_map, top_mask)
    normal_stats = _normal_stats_for_mask(normals_map, top_mask)

    return top_mask, {
        "reject_reason": "accepted",
        "view_profile": view,
        "band_fill": band_fill,
        "visible_ratio": visible_ratio,
        "edge_case": edge_case,
        "depth_stats": depth_stats,
        "normal_stats": normal_stats,
    }


# -----------------------------------------------------------------------------
# Obstacle and free-space extraction
# -----------------------------------------------------------------------------

def _panoptic_overlap_obstacles(
    top_mask: np.ndarray,
    instance_label: str,
    panoptic: Optional[Dict[str, Any]],
    cfg: dict,
) -> np.ndarray:
    if panoptic is None or not bool(cfg["obstacle_detection"].get("subtract_panoptic_objects", True)):
        return np.zeros_like(top_mask, dtype=np.uint8)

    seg = panoptic["segmentation"]
    segments_info = panoptic.get("segments_info", [])
    ignore_labels = {str(x).lower() for x in cfg["obstacle_detection"].get("ignore_same_support_labels", [])}

    obstacle = np.zeros_like(top_mask, dtype=np.uint8)
    top_region = top_mask > 0

    for info in segments_info:
        sid = info.get("id")
        raw_label = str(info.get("label_name", "")).lower()
        canon = _canonicalize_label(raw_label, cfg)
        if canon == instance_label or raw_label in ignore_labels or canon in ignore_labels:
            continue
        seg_mask = seg == sid
        overlap = np.logical_and(top_region, seg_mask)
        if overlap.sum() <= 0:
            continue
        obstacle[overlap] = 255

    return obstacle


def _extract_obstacle_mask(
    top_mask: np.ndarray,
    structure_maps: Dict[str, np.ndarray],
    depth_map: Optional[np.ndarray],
    panoptic: Optional[Dict[str, Any]],
    instance_label: str,
    cfg: dict,
) -> np.ndarray:
    obs_cfg = cfg["obstacle_detection"]
    top_region = top_mask > 0
    if not top_region.any():
        return np.zeros_like(top_mask, dtype=np.uint8)

    grad = structure_maps["grad_mag"]
    depth_grad = structure_maps["depth_grad"]

    obstacle = np.zeros_like(top_mask, dtype=np.uint8)
    obstacle[top_region] = (
        (grad[top_region] > float(obs_cfg["edge_threshold"])) |
        (depth_grad[top_region] > float(obs_cfg["depth_threshold"]))
    ).astype(np.uint8) * 255

    if depth_map is not None:
        ref_depth = float(np.median(depth_map[top_region]))
        delta = np.abs(depth_map - ref_depth)
        delta_n = _normalize_map(delta, percentile_hi=96.0)
        depth_bin = np.zeros_like(top_mask, dtype=np.uint8)
        depth_bin[top_region] = (delta_n[top_region] > float(obs_cfg["depth_threshold"])).astype(np.uint8) * 255
        obstacle = np.maximum(obstacle, depth_bin)

    obstacle = np.maximum(obstacle, _panoptic_overlap_obstacles(top_mask, instance_label, panoptic, cfg))
    obstacle = _binary_clean(
        obstacle,
        open_px=int(obs_cfg["open_px"]),
        close_px=int(obs_cfg["close_px"]),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats((obstacle > 0).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(obstacle, dtype=np.uint8)
    top_area = max(1, _mask_area(top_mask))
    min_ratio = float(obs_cfg["min_component_area_ratio_of_top"])
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area / top_area < min_ratio:
            continue
        filtered[labels == idx] = 255

    filtered = np.where(np.logical_and(filtered > 0, top_region), 255, 0).astype(np.uint8)
    return filtered


def _extract_free_space_mask(top_mask: np.ndarray, obstacle_mask: np.ndarray, cfg: dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    free_cfg = cfg["free_space"]
    top_region = top_mask > 0
    obstacle = obstacle_mask > 0

    free_mask = np.where(np.logical_and(top_region, ~obstacle), 255, 0).astype(np.uint8)
    ys, xs = np.where(top_region)
    if len(xs) == 0:
        return free_mask, None

    bbox = _mask_bbox(top_mask)
    if bbox is None:
        return free_mask, None
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    erode_px = max(1, int(round(min(w, h) * float(free_cfg["erode_margin_ratio"]))))
    kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
    eroded = cv2.erode((free_mask > 0).astype(np.uint8) * 255, kernel, iterations=1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats((eroded > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return eroded, None

    top_area = max(1, _mask_area(top_mask))
    min_ratio = float(free_cfg["min_component_area_ratio_of_top"])

    best_idx = -1
    best_score = -1e9
    top_cy, top_cx = np.mean(ys), np.mean(xs)

    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area / top_area < min_ratio:
            continue
        cx, cy = centroids[idx]
        dist = math.hypot(float(cx) - float(top_cx), float(cy) - float(top_cy))
        score = float(area)
        if bool(free_cfg.get("prefer_centered_components", True)):
            score -= 0.35 * dist
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return eroded, None

    best = np.where(labels == best_idx, 255, 0).astype(np.uint8)
    return eroded, best


# -----------------------------------------------------------------------------
# Scoring and support build
# -----------------------------------------------------------------------------

def _compute_contact_band(box_xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_xyxy
    h = max(1, y1 - y0)
    band_h = max(4, int(round(h * 0.18)))
    return x0, max(y0, y1 - band_h), x1, y1


def _thin_strip_penalty(box_xyxy: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box_xyxy
    w = max(1.0, float(x1 - x0))
    h = max(1.0, float(y1 - y0))
    aspect = w / h
    if aspect >= 1.35:
        return 0.0
    if aspect <= 0.45:
        return 1.0
    return float(np.clip((1.35 - aspect) / 0.90, 0.0, 1.0))


def _build_support_surface(
    support_idx: int,
    instance: SupportInstance,
    instance_mask: np.ndarray,
    image: Image.Image,
    depth_map: Optional[np.ndarray],
    normals_map: Optional[np.ndarray],
    structure_maps: Dict[str, np.ndarray],
    panoptic: Optional[Dict[str, Any]],
    cfg: dict,
) -> Optional[Tuple[SupportSurface, Dict[str, np.ndarray]]]:
    top_mask, top_info = _extract_visible_top_mask(
        instance_mask=instance_mask,
        instance_box=instance.box_xyxy,
        normals_map=normals_map,
        depth_map=depth_map,
        image_size=image.size,
        cfg=cfg,
    )
    if top_mask is None:
        return None

    top_bbox = _mask_bbox(top_mask)
    if top_bbox is None:
        return None

    obstacle_mask = _extract_obstacle_mask(
        top_mask=top_mask,
        structure_maps=structure_maps,
        depth_map=depth_map,
        panoptic=panoptic,
        instance_label=instance.label,
        cfg=cfg,
    )
    free_mask, best_patch_mask = _extract_free_space_mask(top_mask, obstacle_mask, cfg)
    if best_patch_mask is None:
        best_patch_mask = free_mask.copy()

    top_area = max(1, _mask_area(top_mask))
    best_free_area = _mask_area(best_patch_mask)
    free_area_ratio = float(best_free_area) / float(top_area)
    occupied_area_ratio = float((_mask_area(obstacle_mask))) / float(top_area)

    if free_area_ratio < float(cfg["free_space"]["min_free_area_ratio_of_top"]):
        return None

    border = np.zeros_like(top_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_touch_ratio = float(np.logical_and(border, top_mask > 0).sum()) / float(top_area)

    depth_stats = top_info.get("depth_stats", _depth_stats_for_mask(depth_map, top_mask))
    normal_stats = top_info.get("normal_stats", _normal_stats_for_mask(normals_map, top_mask))

    thin_penalty = _thin_strip_penalty(top_bbox)
    flatness_score = float(np.clip(1.0 - (_safe_float(depth_stats.get("std"), 0.20) / 0.12), 0.0, 1.0))
    horizontal_score = float(np.clip(_safe_float(normal_stats.get("abs_mean_nz"), 0.0), 0.0, 1.0))
    visible_top_ratio = float(top_info.get("visible_ratio", 0.0))
    top_band_fill = float(top_info.get("band_fill", 0.0))

    score_cfg = cfg["support_scoring"]
    score = 0.0
    score += float(score_cfg["horizontal_weight"]) * horizontal_score
    score += float(score_cfg["flatness_weight"]) * flatness_score
    score += float(score_cfg["free_area_weight"]) * free_area_ratio
    score += float(score_cfg["top_visibility_weight"]) * max(visible_top_ratio, top_band_fill)
    score -= float(score_cfg["occupancy_penalty_weight"]) * occupied_area_ratio
    score -= float(score_cfg["border_penalty_weight"]) * border_touch_ratio
    score -= float(score_cfg["thin_penalty_weight"]) * thin_penalty

    if instance.label == "floor":
        score -= float(score_cfg["floor_penalty"])
    if instance.label == "countertop":
        score += float(score_cfg["counter_bonus"])
    if instance.label == "cabinet" and visible_top_ratio > 0.12:
        score += float(score_cfg["cabinet_bonus_if_top_good"])

    if score < float(score_cfg["min_final_score"]):
        return None

    polygon = _mask_to_polygon(best_patch_mask, epsilon_ratio=float(cfg["top_surface"]["polygon_epsilon_ratio"]))
    quad = None
    H_unit_to_img = None
    H_img_to_unit = None
    if bool(cfg["top_surface"].get("fit_quad", True)):
        quad, H_unit_to_img, H_img_to_unit = _fit_support_quad_and_homography(best_patch_mask)

    front_edge_xy = None
    back_edge_xy = None
    if quad is not None:
        front_edge_xy = [[float(quad[3, 0]), float(quad[3, 1])], [float(quad[2, 0]), float(quad[2, 1])]]
        back_edge_xy = [[float(quad[0, 0]), float(quad[0, 1])], [float(quad[1, 0]), float(quad[1, 1])]]

    refine_region_mask, refine_crop_box = _build_refine_region(top_mask, best_patch_mask, image.size, cfg)

    surface = SupportSurface(
        id=f"support_{support_idx:02d}",
        label=instance.label,
        confidence=float(score),
        box_xyxy=top_bbox,
        support_mode="plane" if quad is not None else "surface",
        region_area_px=top_area,
        contact_band_xyxy=_compute_contact_band(top_bbox),
        visible_top_polygon_xy=polygon,
        plane_quad_xy=quad.tolist() if quad is not None else None,
        front_edge_xy=front_edge_xy,
        back_edge_xy=back_edge_xy,
        homography_unit_to_img=H_unit_to_img.tolist() if H_unit_to_img is not None else None,
        homography_img_to_unit=H_img_to_unit.tolist() if H_img_to_unit is not None else None,
        depth_stats=depth_stats,
        normal_stats=normal_stats,
        geometry_scores={
            "horizontal_score": horizontal_score,
            "flatness_score": flatness_score,
            "top_band_fill_ratio": top_band_fill,
            "visible_top_surface_ratio": visible_top_ratio,
            "free_area_ratio": free_area_ratio,
            "occupied_area_ratio": occupied_area_ratio,
            "thin_strip_penalty": thin_penalty,
        },
        vlm_verification={},
        usable_area_ratio=free_area_ratio,
        occupied_area_ratio=occupied_area_ratio,
        border_touch_ratio=border_touch_ratio,
        prior_dimensions_m={},
        prior_scale={},
        debug={
            "instance_id": instance.id,
            "instance_label": instance.label,
            "instance_source": instance.source,
            "instance_confidence": instance.confidence,
            "view_profile": top_info.get("view_profile", "unknown"),
            "edge_case": bool(top_info.get("edge_case", False)),
            "support_score": score,
            "top_extract_reason": top_info.get("reject_reason", "accepted"),
            "refine_crop_xyxy": list(refine_crop_box),
        },
    )

    return surface, {
        "instance_mask": instance_mask,
        "top_mask": top_mask,
        "obstacle_mask": obstacle_mask,
        "free_mask": free_mask,
        "best_patch_mask": best_patch_mask,
        "refine_region_mask": refine_region_mask,
    }


# -----------------------------------------------------------------------------
# Scene priors
# -----------------------------------------------------------------------------

def _infer_scene_type_from_supports(surfaces: Sequence[SupportSurface], cfg: dict) -> str:
    hint = str(cfg["scene_priors"].get("scene_type_hint", "") or "").strip()
    if hint:
        return hint

    labels = " ".join(s.label.lower() for s in surfaces)
    if "countertop" in labels or "cabinet" in labels:
        return "kitchen"
    if "nightstand" in labels or "dresser" in labels:
        return "bedroom"
    if "desk" in labels:
        return "office"
    if "table" in labels or "bench" in labels:
        return "living room"
    return "generic indoor room"


def _heuristic_scene_dimensions(scene_type: str) -> Dict[str, Any]:
    st = scene_type.lower().strip()
    if "kitchen" in st:
        dims = (3.6, 4.4, 2.5)
    elif "bedroom" in st:
        dims = (3.4, 4.0, 2.5)
    elif "office" in st:
        dims = (3.2, 4.2, 2.5)
    elif "living" in st:
        dims = (4.2, 5.0, 2.6)
    else:
        dims = (3.5, 4.5, 2.5)
    return {
        "width_m": float(dims[0]),
        "depth_m": float(dims[1]),
        "height_m": float(dims[2]),
    }


def _heuristic_support_dimensions(label: str) -> Dict[str, float]:
    lab = label.lower().strip()
    presets = {
        "countertop": (2.2, 0.65, 0.91, 0.04),
        "table": (1.4, 0.75, 0.75, 0.04),
        "desk": (1.4, 0.70, 0.74, 0.04),
        "shelf": (1.0, 0.30, 1.30, 0.03),
        "bench": (1.2, 0.35, 0.46, 0.04),
        "nightstand": (0.50, 0.40, 0.60, 0.03),
        "dresser": (1.20, 0.50, 0.85, 0.03),
        "cabinet": (1.20, 0.55, 0.91, 0.04),
        "windowsill": (1.0, 0.18, 0.95, 0.03),
        "floor": (3.5, 4.0, 0.0, 0.0),
    }
    width_m, depth_m, height_m, thickness_m = presets.get(lab, (1.0, 0.5, 0.75, 0.04))
    return {
        "width_m": float(width_m),
        "depth_m": float(depth_m),
        "height_m": float(height_m),
        "top_surface_height_m": float(height_m),
        "thickness_m": float(thickness_m),
        "confidence": 0.60,
    }


def _apply_priors_to_surfaces(surfaces: List[SupportSurface], scene_priors: Dict[str, Any]) -> None:
    support_map = {str(item.get("id", "")): item for item in scene_priors.get("supports", []) or []}
    for s in surfaces:
        prior = support_map.get(s.id)
        if not prior:
            continue

        width_m = _safe_float(prior.get("width_m"), 0.0)
        depth_m = _safe_float(prior.get("depth_m"), 0.0)
        height_m = _safe_float(prior.get("height_m"), 0.0)
        top_surface_height_m = _safe_float(prior.get("top_surface_height_m"), height_m)
        thickness_m = _safe_float(prior.get("thickness_m"), 0.04)
        conf = float(np.clip(_safe_float(prior.get("confidence"), 0.5), 0.0, 1.0))

        s.prior_dimensions_m = {
            "width_m": width_m,
            "depth_m": depth_m,
            "height_m": height_m,
            "top_surface_height_m": top_surface_height_m,
            "thickness_m": thickness_m,
            "confidence": conf,
            "units": "meters",
        }

        x0, y0, x1, y1 = s.box_xyxy
        bw = max(1.0, float(x1 - x0))
        bh = max(1.0, float(y1 - y0))

        s.prior_scale = {
            "units": "meters_per_pixel",
            "width_m_per_px": width_m / bw if width_m > 0 else None,
            "height_m_per_px": height_m / bh if height_m > 0 else None,
            "depth_m_per_px_proxy": depth_m / bh if depth_m > 0 else None,
            "plane_unit_size_m": {
                "u_width_m": width_m if width_m > 0 else None,
                "v_depth_m": depth_m if depth_m > 0 else None,
            } if s.homography_unit_to_img is not None else None,
        }


def _estimate_scene_and_support_priors(surfaces: List[SupportSurface], cfg: dict) -> Dict[str, Any]:
    scene_type = _infer_scene_type_from_supports(surfaces, cfg)
    room = _heuristic_scene_dimensions(scene_type)

    supports_payload = []
    for s in surfaces:
        dims = _heuristic_support_dimensions(s.label)
        supports_payload.append({
            "id": s.id,
            **dims,
        })

    scene_priors = {
        "units": "meters",
        "source": "heuristic_label_based",
        "scene_type": scene_type,
        "room_dimensions_m": room,
        "camera_height_m": float(cfg["scene_priors"].get("camera_height_m", 1.5)),
        "confidence": 0.60,
        "supports": supports_payload,
        "debug": {},
    }

    _apply_priors_to_surfaces(surfaces, scene_priors)
    return scene_priors


# -----------------------------------------------------------------------------
# Debug rendering
# -----------------------------------------------------------------------------

def _draw_instances(image: Image.Image, items: List[Tuple[SupportInstance, np.ndarray]], out_path: Path) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for i, (inst, _) in enumerate(items):
        x0, y0, x1, y1 = inst.box_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=(255, 128, 0), width=3)
        draw.text((x0 + 4, max(0, y0 - 16)), f"{i}:{inst.label} {inst.source} {inst.confidence:.2f}", fill=(255, 255, 255))
    overlay.save(out_path)


def _draw_supports(image: Image.Image, surfaces: List[SupportSurface], out_path: Path) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for i, s in enumerate(surfaces):
        x0, y0, x1, y1 = s.box_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0 + 4, max(0, y0 - 18)), f"{i}:{s.label} score={s.confidence:.2f}", fill=(255, 255, 255))
        if s.visible_top_polygon_xy and len(s.visible_top_polygon_xy) >= 3:
            draw.polygon([tuple(p) for p in s.visible_top_polygon_xy], outline=(0, 255, 0), width=2)
        if s.plane_quad_xy is not None and len(s.plane_quad_xy) == 4:
            q = [tuple(map(int, p)) for p in s.plane_quad_xy]
            draw.line([q[0], q[1], q[2], q[3], q[0]], fill=(255, 255, 0), width=2)
    overlay.save(out_path)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

class SceneUnderstandingEngine:
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda") -> None:
        load_dotenv()
        self.cfg = cfg
        self.requested_device = device
        self.device = _choose_device(device)

    def analyze(self, scene_path: Path, output_dir: Path) -> SceneUnderstandingResult:
        _ensure_dir(output_dir)
        debug_dir = output_dir / "debug"
        if bool(self.cfg["output"].get("save_debug", True)):
            _ensure_dir(debug_dir)

        scene = _open_rgb(scene_path)
        scene_np = _pil_to_np_rgb(scene)

        depth_map = _estimate_depth_map(scene, self.device, self.cfg)
        normals_map = _estimate_normals_map(scene, self.device, self.cfg)
        panoptic = _run_oneformer_panoptic(scene, self.device, self.cfg)
        structure_maps = _compute_structure_maps(scene_np, depth_map)

        instances = _generate_support_instances(
            image=scene,
            panoptic=panoptic,
            device=self.device,
            cfg=self.cfg,
        )

        supports: List[SupportSurface] = []
        debug_masks: Dict[str, Dict[str, np.ndarray]] = {}

        for idx, (inst, inst_mask) in enumerate(instances):
            built = _build_support_surface(
                support_idx=idx,
                instance=inst,
                instance_mask=inst_mask,
                image=scene,
                depth_map=depth_map,
                normals_map=normals_map,
                structure_maps=structure_maps,
                panoptic=panoptic,
                cfg=self.cfg,
            )
            if built is None:
                continue
            surface, maps = built
            supports.append(surface)
            debug_masks[surface.id] = maps

        supports.sort(key=lambda s: s.confidence, reverse=True)

        scene_priors = _estimate_scene_and_support_priors(supports, self.cfg) if bool(self.cfg["scene_priors"].get("enabled", True)) else {}

        depth_path = None
        normals_path = None
        support_json_path = None

        if bool(self.cfg["output"].get("save_debug", True)):
            _save_gray(depth_map, debug_dir / "01_depth.png")
            depth_path = str(debug_dir / "01_depth.png")

            if normals_map is not None:
                normals_vis = ((normals_map + 1.0) * 0.5).clip(0.0, 1.0)
                _np_rgb_to_pil((normals_vis * 255.0).astype(np.uint8)).save(debug_dir / "02_normals.png")
                normals_path = str(debug_dir / "02_normals.png")

            _save_gray(structure_maps["grad_mag"], debug_dir / "03_grad_mag.png")
            _save_gray(structure_maps["depth_grad"], debug_dir / "04_depth_grad.png")
            _draw_instances(scene, instances, debug_dir / "05_support_instances.png")

            top_masks = []
            free_masks = []
            obstacle_masks = []
            for s in supports:
                maps = debug_masks.get(s.id)
                if maps is None:
                    continue
                Image.fromarray(maps["instance_mask"], mode="L").save(debug_dir / f"{s.id}_instance_mask.png")
                Image.fromarray(maps["top_mask"], mode="L").save(debug_dir / f"{s.id}_top_mask.png")
                Image.fromarray(maps["obstacle_mask"], mode="L").save(debug_dir / f"{s.id}_obstacle_mask.png")
                Image.fromarray(maps["free_mask"], mode="L").save(debug_dir / f"{s.id}_free_mask.png")
                Image.fromarray(maps["best_patch_mask"], mode="L").save(debug_dir / f"{s.id}_best_patch_mask.png")
                Image.fromarray(maps["refine_region_mask"], mode="L").save(debug_dir / f"{s.id}_refine_region_mask.png")
                top_masks.append(maps["top_mask"])
                free_masks.append(maps["best_patch_mask"])
                obstacle_masks.append(maps["obstacle_mask"])

            _save_color_overlay(scene, top_masks, debug_dir / "10_top_masks_overlay.png")
            _save_color_overlay(scene, obstacle_masks, debug_dir / "11_obstacle_masks_overlay.png")
            _save_color_overlay(scene, free_masks, debug_dir / "12_free_patches_overlay.png")
            _draw_supports(scene, supports, debug_dir / "13_support_overview.png")

        if bool(self.cfg["output"].get("save_json", True)):
            data = {
                "image_size": {"width": scene.size[0], "height": scene.size[1]},
                "scene_priors": scene_priors,
                "supports": [asdict(s) for s in supports],
            }
            support_json_path = str(output_dir / "scene_understanding.json")
            with (output_dir / "scene_understanding.json").open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return SceneUnderstandingResult(
            image_size=scene.size,
            supports=supports,
            depth_path=depth_path,
            normals_path=normals_path,
            support_json_path=support_json_path,
            debug_dir=str(debug_dir) if bool(self.cfg["output"].get("save_debug", True)) else None,
            scene_priors=scene_priors,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene understanding for support/surface inference and placement preparation."
    )
    parser.add_argument("--scene", required=True, type=str, help="Path to the scene image.")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device preference.")
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.config))
    engine = SceneUnderstandingEngine(cfg=cfg, device=args.device)
    result = engine.analyze(
        scene_path=Path(args.scene),
        output_dir=Path(args.output),
    )

    if result.support_json_path is not None:
        print(result.support_json_path)
    else:
        print("done")


if __name__ == "__main__":
    main()