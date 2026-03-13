"""
python .\extractor\object_extractor.py `
   --image ".\data\input\flowers.jpg" `
   --object-label "vase with flowers" `
   --object-understanding ".\data\output\object_understanding\object_understanding.json" `
   --output ".\data\output\extracted" `
   --config ".\config\extractor-config\extractor_config.yaml" `
   --device cuda
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageFilter

from transformers import (
    AutoModelForImageSegmentation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    pipeline,
)

try:
    from transformers import Sam2Model, Sam2Processor  # type: ignore
    _SAM2_AVAILABLE = True
except Exception:
    Sam2Model = None  # type: ignore
    Sam2Processor = None  # type: ignore
    _SAM2_AVAILABLE = False

try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None


# -----------------------------------------------------------------------------
# Default config
# -----------------------------------------------------------------------------

DEFAULT_EXTRACT_CFG: Dict[str, Any] = {
    "models": {
        "detector_id": "IDEA-Research/grounding-dino-tiny",
        "sam2_id": "facebook/sam2.1-hiera-large",
        "foreground_id": "ZhengPeng7/BiRefNet",
    },
    "runtime": {
        "device": "cuda",
        "dtype": "float16",
    },
    "input": {
        "max_side": 1536,
        "pad_to_crop": 6,
        "prefer_embedded_alpha": True,
        "trust_embedded_alpha_if_nontrivial": True,
    },
    "labels": {
        "split_pattern": r"\s*(?:,|and|\+|&|\/)\s*",
        "expand_basic_variants": True,
        "always_add_full_phrase": True,
        "drop_stopwords": True,
        "stopwords": ["a", "an", "the"],
    },
    "detection": {
        "box_threshold": 0.24,
        "text_threshold": 0.18,
        "max_boxes": 24,
        "min_box_area_ratio": 0.0008,
        "max_box_area_ratio": 0.95,
        "nms_iou": 0.45,
        "prefer_central_weight": 0.15,
        "prefer_large_weight": 0.20,
        "border_touch_penalty": 0.40,
    },
    "segmentation": {
        "use_sam2": True,
        "multimask_output": True,
        "mask_threshold": 0.0,
        "sam_box_expand_ratio": 0.03,
        "sam_point_prompts": True,
        "sam_center_point_weight": 1.0,
        "sam_negative_ring": True,
    },
    "foreground_prior": {
        "enabled": True,
        "weight": 0.55,
        "fallback_only": False,
    },
    "mask_scoring": {
        "detector_score_weight": 1.25,
        "foreground_iou_weight": 1.10,
        "interior_fill_weight": 0.75,
        "compactness_weight": 0.20,
        "edge_density_weight": 0.12,
        "box_tightness_weight": 0.30,
        "border_touch_penalty": 0.60,
        "tiny_island_penalty": 0.50,
    },
    "postprocess": {
        "binarize_threshold": 0.50,
        "close_px": 5,
        "open_px": 3,
        "fill_holes": True,
        "keep_mode": "label_aware",  # one_of: label_aware, largest, all
        "keep_min_area_ratio": 0.002,
        "merge_iou": 0.15,
        "feather_px": 1.2,
        "alpha_hard_zero_below": 8,
        "alpha_hard_full_above": 248,
    },
    "fallback": {
        "use_rembg_if_available": True,
        "use_dark_bg_heuristic": True,
    },
    "debug": {
        "save_debug": True,
    },
}


# -----------------------------------------------------------------------------
# Lazy globals
# -----------------------------------------------------------------------------

_DET_PROCESSOR = None
_DET_MODEL = None
_SAM2_PROCESSOR = None
_SAM2_MODEL = None
_FOREGROUND_PIPE = None


# -----------------------------------------------------------------------------
# Data classes
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

    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    def clamp(self, w: int, h: int) -> "BoundingBox":
        return BoundingBox(
            x0=max(0.0, min(float(w - 1), self.x0)),
            y0=max(0.0, min(float(h - 1), self.y0)),
            x1=max(0.0, min(float(w), self.x1)),
            y1=max(0.0, min(float(h), self.y1)),
            score=self.score,
            label=self.label,
        )

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x0)),
            int(round(self.y0)),
            int(round(self.x1)),
            int(round(self.y1)),
        )


@dataclass
class MaskCandidate:
    mask: np.ndarray          # float32 in [0,1]
    binary: np.ndarray        # uint8 0/255
    score: float
    label: str
    det_score: float
    box: BoundingBox
    reason: str = ""


@dataclass
class ExtractedObject:
    rgba: Image.Image
    alpha: Image.Image
    mask_binary: Image.Image
    bbox_xyxy: Tuple[int, int, int, int]
    method: str
    label: str
    debug: Dict[str, Any]


# -----------------------------------------------------------------------------
# Utils
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
    return _deep_update(DEFAULT_EXTRACT_CFG, user_cfg)


def _load_object_understanding(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_object_extraction_priors(object_understanding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not object_understanding:
        return {}
    return object_understanding.get("object_priors", {}).get("extraction_priors", {}) or {}


def _get_object_physical_attributes(object_understanding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not object_understanding:
        return {}
    return object_understanding.get("object_priors", {}).get("physical_attributes", {}) or {}


def _build_extraction_cfg(
    cfg: dict,
    object_understanding: Optional[Dict[str, Any]] = None,
) -> dict:
    if not object_understanding:
        return cfg

    extract_priors = _get_object_extraction_priors(object_understanding)
    physical_attrs = _get_object_physical_attributes(object_understanding)

    post_cfg = dict(cfg["postprocess"])

    hole_policy = str(extract_priors.get("mask_hole_policy", "fill_small_holes_only"))
    island_policy = str(extract_priors.get("mask_island_policy", "largest_plus_attached"))
    foreground_mode = str(extract_priors.get("foreground_mode", "generic"))
    pocket_removal = str(extract_priors.get("background_pocket_removal", "medium"))
    container_bias = str(extract_priors.get("container_region_bias", "none"))

    if foreground_mode in {"structure_aware", "sparse_top_dense_bottom"}:
        post_cfg["close_px"] = min(int(post_cfg.get("close_px", 5)), 3)
        post_cfg["open_px"] = max(int(post_cfg.get("open_px", 3)), 2)
        post_cfg["feather_px"] = max(float(post_cfg.get("feather_px", 1.2)), 1.4)

    preserve_holes = (
        hole_policy in {"preserve_true_holes", "preserve_structural_holes_only"}
        or bool(physical_attrs.get("has_true_holes", False))
    )
    if preserve_holes:
        post_cfg["fill_holes"] = False

    if island_policy == "keep_relevant_parts":
        post_cfg["keep_mode"] = "all"
        post_cfg["keep_min_area_ratio"] = min(float(post_cfg.get("keep_min_area_ratio", 0.002)), 0.0006)
    elif island_policy == "largest_plus_attached":
        post_cfg["keep_mode"] = "label_aware"

    if pocket_removal == "high":
        post_cfg["keep_min_area_ratio"] = min(float(post_cfg.get("keep_min_area_ratio", 0.002)), 0.0012)
    elif pocket_removal == "low":
        post_cfg["keep_min_area_ratio"] = min(float(post_cfg.get("keep_min_area_ratio", 0.002)), 0.0004)

    sam_cfg = dict(cfg["segmentation"])
    if (
        foreground_mode in {"structure_aware", "sparse_top_dense_bottom"}
        or bool(physical_attrs.get("has_thin_structures", False))
        or bool(physical_attrs.get("is_multipart", False))
        or bool(physical_attrs.get("upper_region_sparse", False))
    ):
        sam_cfg["multimask_output"] = True
        sam_cfg["sam_negative_ring"] = True
        sam_cfg["sam_box_expand_ratio"] = max(float(sam_cfg.get("sam_box_expand_ratio", 0.03)), 0.05)

    tmp_cfg = dict(cfg)
    tmp_cfg["postprocess"] = post_cfg
    tmp_cfg["segmentation"] = sam_cfg
    tmp_cfg["_object_understanding_debug"] = {
        "extraction_priors": extract_priors,
        "physical_attributes": physical_attrs,
        "hole_policy": hole_policy,
        "island_policy": island_policy,
        "foreground_mode": foreground_mode,
        "background_pocket_removal": pocket_removal,
        "container_region_bias": container_bias,
    }
    return tmp_cfg


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _pil_to_np_rgba(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA"))


def _np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _np_to_pil_rgba(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def _np_to_pil_l(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), "L")


def _maybe_resize(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    w, h = image.size
    scale = min(float(max_side) / float(max(w, h)), 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_w = max(32, int(round(w * scale / 8) * 8))
    new_h = max(32, int(round(h * scale / 8) * 8))
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def _expand_box(box: BoundingBox, ratio: float, w: int, h: int) -> BoundingBox:
    bw = box.width()
    bh = box.height()
    ex = bw * ratio
    ey = bh * ratio
    return BoundingBox(
        x0=box.x0 - ex,
        y0=box.y0 - ey,
        x1=box.x1 + ex,
        y1=box.y1 + ey,
        score=box.score,
        label=box.label,
    ).clamp(w, h)


def _alpha_bbox(alpha_u8: np.ndarray, thr: int = 10) -> Tuple[int, int, int, int]:
    ys, xs = np.where(alpha_u8 > thr)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _crop_rgba_to_alpha(rgba: Image.Image, pad: int = 4) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    arr = _pil_to_np_rgba(rgba)
    alpha = arr[:, :, 3]
    x0, y0, x1, y1 = _alpha_bbox(alpha, thr=10)
    h, w = alpha.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    cropped = arr[y0:y1 + 1, x0:x1 + 1, :]
    return _np_to_pil_rgba(cropped), (x0, y0, x1, y1)


def _largest_component(binary_u8: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary_u8 > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary_u8.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _fill_holes(binary_u8: np.ndarray) -> np.ndarray:
    h, w = binary_u8.shape
    flood = binary_u8.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary_u8, inv)


def _morph(binary_u8: np.ndarray, open_px: int = 0, close_px: int = 0) -> np.ndarray:
    out = binary_u8.copy()
    if close_px > 0:
        k = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_px > 0:
        k = np.ones((open_px * 2 + 1, open_px * 2 + 1), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    return out


def _box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.area() + b.area() - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def _mask_bbox(binary_u8: np.ndarray) -> BoundingBox:
    ys, xs = np.where(binary_u8 > 0)
    if len(xs) == 0:
        return BoundingBox(0, 0, 0, 0, 0.0, "")
    return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _mask_area_ratio(binary_u8: np.ndarray) -> float:
    return float((binary_u8 > 0).sum()) / float(binary_u8.shape[0] * binary_u8.shape[1])


def _border_touch_ratio(binary_u8: np.ndarray) -> float:
    h, w = binary_u8.shape
    border = np.zeros_like(binary_u8, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    fg = binary_u8 > 0
    denom = max(1, int(fg.sum()))
    return float(np.logical_and(border, fg).sum()) / float(denom)


def _edge_density(rgb: np.ndarray, binary_u8: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    mask = binary_u8 > 0
    if mask.sum() == 0:
        return 0.0
    return float((edges[mask] > 0).mean())


def _compactness(binary_u8: np.ndarray) -> float:
    cnts, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = max(1.0, float(cv2.contourArea(cnt)))
    peri = max(1.0, float(cv2.arcLength(cnt, True)))
    return float(4.0 * math.pi * area / (peri * peri))


def _nonmax_suppress_boxes(boxes: List[BoundingBox], iou_thr: float) -> List[BoundingBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: List[BoundingBox] = []
    for box in boxes:
        if all(_box_iou(box, prev) < iou_thr for prev in kept):
            kept.append(box)
    return kept


def _save_mask_overlay(image: Image.Image, binary_u8: np.ndarray, out_path: Path) -> None:
    rgb = _pil_to_np_rgb(image).copy()
    overlay = rgb.copy()
    mask = binary_u8 > 0
    overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    _np_to_pil_rgb(overlay).save(out_path)


def _save_boxes_overlay(image: Image.Image, boxes: Sequence[BoundingBox], out_path: Path) -> None:
    arr = _pil_to_np_rgb(image).copy()
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    for b in boxes:
        x0, y0, x1, y1 = b.to_xyxy()
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        txt = f"{b.label} {b.score:.2f}"
        cv2.putText(bgr, txt, (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(out_path)


# -----------------------------------------------------------------------------
# Label parsing
# -----------------------------------------------------------------------------

def _normalize_label_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _singularize(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _pluralize(word: str) -> str:
    if word.endswith("y") and len(word) > 2:
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _parse_target_groups(label_text: str, cfg: dict) -> List[str]:
    label_text = _normalize_label_text(label_text)
    pattern = cfg["labels"]["split_pattern"]
    parts = [p.strip() for p in re.split(pattern, label_text) if p.strip()]
    if not parts:
        return [label_text]
    return parts


def _label_phrases(label_text: str, cfg: dict) -> List[str]:
    full = _normalize_label_text(label_text)
    groups = _parse_target_groups(full, cfg)
    phrases: List[str] = []

    if cfg["labels"].get("always_add_full_phrase", True):
        phrases.append(full)

    for g in groups:
        phrases.append(g)
        if cfg["labels"].get("expand_basic_variants", True):
            toks = [t for t in g.split() if t]
            if cfg["labels"].get("drop_stopwords", True):
                stop = set(cfg["labels"].get("stopwords", []))
                toks = [t for t in toks if t not in stop]
            if toks:
                singular = " ".join(_singularize(t) for t in toks)
                plural = " ".join(_pluralize(_singularize(t)) for t in toks)
                phrases.extend([singular, plural])

    out: List[str] = []
    seen = set()
    for p in phrases:
        p = _normalize_label_text(p)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def _choose_device(device_pref: str) -> torch.device:
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
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


def _get_sam2(device: torch.device, cfg: dict):
    global _SAM2_PROCESSOR, _SAM2_MODEL
    if not _SAM2_AVAILABLE:
        return None, None
    if _SAM2_PROCESSOR is None or _SAM2_MODEL is None:
        model_id = cfg["models"]["sam2_id"]
        token = os.getenv("HF_TOKEN")
        _SAM2_PROCESSOR = Sam2Processor.from_pretrained(model_id, token=token)
        _SAM2_MODEL = Sam2Model.from_pretrained(model_id, token=token)
        _SAM2_MODEL.to(device)
        _SAM2_MODEL.eval()
    return _SAM2_PROCESSOR, _SAM2_MODEL


def _get_foreground_pipe(device: torch.device, cfg: dict):
    global _FOREGROUND_PIPE
    if _FOREGROUND_PIPE is None:
        model_id = cfg["models"]["foreground_id"]
        device_index = 0 if device.type == "cuda" else -1
        token = os.getenv("HF_TOKEN")
        _FOREGROUND_PIPE = pipeline(
            task="image-segmentation",
            model=model_id,
            device=device_index,
            token=token,
        )
    return _FOREGROUND_PIPE


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def _detect_label_boxes(
    image: Image.Image,
    label_text: str,
    device: torch.device,
    cfg: dict,
) -> List[BoundingBox]:
    phrases = _label_phrases(label_text, cfg)
    resized, scale = _maybe_resize(image, int(cfg["input"]["max_side"]))
    inv_scale = 1.0 / scale

    processor, model = _get_detector(device, cfg)
    target_sizes = [resized.size[::-1]]

    all_boxes: List[BoundingBox] = []

    for phrase in phrases:
        text = [[phrase]]
        inputs = processor(images=resized, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(cfg["detection"]["box_threshold"]),
            text_threshold=float(cfg["detection"]["text_threshold"]),
            target_sizes=target_sizes,
        )

        if not results:
            continue

        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            score = float(scores[i].item()) if i < len(scores) else 1.0
            lbl = str(labels[i]) if i < len(labels) else phrase

            scaled = BoundingBox(
                x0=x0 * inv_scale,
                y0=y0 * inv_scale,
                x1=x1 * inv_scale,
                y1=y1 * inv_scale,
                score=score,
                label=lbl,
            ).clamp(*image.size)

            area_ratio = scaled.area() / float(image.size[0] * image.size[1])
            if area_ratio < float(cfg["detection"]["min_box_area_ratio"]):
                continue
            if area_ratio > float(cfg["detection"]["max_box_area_ratio"]):
                continue

            cx, cy = scaled.center()
            center_penalty = abs(cx / image.size[0] - 0.5) + abs(cy / image.size[1] - 0.5)
            border_penalty = 1.0 if _box_touches_border(scaled, image.size[0], image.size[1]) else 0.0
            size_bonus = math.sqrt(max(1e-8, area_ratio))

            adj = (
                score
                + size_bonus * float(cfg["detection"]["prefer_large_weight"])
                - center_penalty * float(cfg["detection"]["prefer_central_weight"])
                - border_penalty * float(cfg["detection"]["border_touch_penalty"])
            )
            scaled.score = float(adj)
            all_boxes.append(scaled)

    all_boxes = _nonmax_suppress_boxes(all_boxes, float(cfg["detection"]["nms_iou"]))
    all_boxes = sorted(all_boxes, key=lambda b: b.score, reverse=True)
    return all_boxes[: int(cfg["detection"]["max_boxes"])]


def _box_touches_border(box: BoundingBox, w: int, h: int, margin: int = 2) -> bool:
    return (
        box.x0 <= margin or box.y0 <= margin or box.x1 >= w - margin or box.y1 >= h - margin
    )


# -----------------------------------------------------------------------------
# Generic foreground prior
# -----------------------------------------------------------------------------

def _foreground_prior_mask(image: Image.Image, device: torch.device, cfg: dict) -> Optional[np.ndarray]:
    if not bool(cfg["foreground_prior"]["enabled"]):
        return None
    try:
        pipe = _get_foreground_pipe(device, cfg)
        out = pipe(image)
        if isinstance(out, list) and out:
            best = None
            best_area = -1
            for item in out:
                m = item.get("mask")
                if m is None:
                    continue
                arr = np.array(m.convert("L"), dtype=np.uint8)
                area = int((arr > 0).sum())
                if area > best_area:
                    best_area = area
                    best = arr
            if best is not None:
                return (best.astype(np.float32) / 255.0).clip(0.0, 1.0)
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# SAM2 segmentation
# -----------------------------------------------------------------------------

def _sam2_masks_from_box(
    image: Image.Image,
    box: BoundingBox,
    device: torch.device,
    cfg: dict,
) -> List[np.ndarray]:
    if not bool(cfg["segmentation"]["use_sam2"]):
        return []
    processor, model = _get_sam2(device, cfg)
    if processor is None or model is None:
        return []

    w, h = image.size
    expanded = _expand_box(box, float(cfg["segmentation"]["sam_box_expand_ratio"]), w, h)

    input_boxes = [[[expanded.x0, expanded.y0, expanded.x1, expanded.y1]]]
    kwargs = {
        "images": image,
        "input_boxes": input_boxes,
        "return_tensors": "pt",
    }

    if bool(cfg["segmentation"].get("sam_point_prompts", True)):
        cx, cy = expanded.center()
        pos_points = [[[
            [cx, cy],
        ]]]
        pos_labels = [[[1]]]

        if bool(cfg["segmentation"].get("sam_negative_ring", True)):
            bw = expanded.width()
            bh = expanded.height()
            neg = [
                [max(0.0, cx - 0.45 * bw), cy],
                [min(float(w - 1), cx + 0.45 * bw), cy],
                [cx, max(0.0, cy - 0.45 * bh)],
                [cx, min(float(h - 1), cy + 0.45 * bh)],
            ]
            input_points = [[[ [cx, cy], *neg ]]]
            input_labels = [[[1, 0, 0, 0, 0]]]
            kwargs["input_points"] = input_points
            kwargs["input_labels"] = input_labels
        else:
            kwargs["input_points"] = pos_points
            kwargs["input_labels"] = pos_labels

    inputs = processor(**kwargs).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks: List[np.ndarray] = []

    pred_masks = getattr(outputs, "pred_masks", None)
    if pred_masks is None:
        return masks

    pm = pred_masks
    while pm.ndim > 4:
        pm = pm[:, 0]

    if pm.ndim == 4:
        pm = pm[0]
    elif pm.ndim == 3:
        pass
    else:
        return masks

    multimask_output = bool(cfg["segmentation"].get("multimask_output", True))
    num_masks_to_keep = pm.shape[0] if multimask_output else min(1, pm.shape[0])

    for i in range(num_masks_to_keep):
        mask = pm[i].detach().float().cpu().numpy()
        mask = 1.0 / (1.0 + np.exp(-mask))
        mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_LINEAR)
        masks.append(mask.astype(np.float32))

    return masks


# -----------------------------------------------------------------------------
# Fallbacks
# -----------------------------------------------------------------------------

def _mask_from_rembg(image: Image.Image) -> Optional[np.ndarray]:
    if rembg_remove is None:
        return None
    try:
        buf = io.BytesIO()
        image.convert("RGBA").save(buf, format="PNG")
        out_bytes = rembg_remove(buf.getvalue())
        out_img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        alpha = np.array(out_img.getchannel("A"), dtype=np.uint8)
        if alpha.max() == 0:
            return None
        return alpha.astype(np.float32) / 255.0
    except Exception:
        return None


def _mask_from_dark_bg_heuristic(image: Image.Image) -> Optional[np.ndarray]:
    rgb = _pil_to_np_rgb(image).astype(np.int16)
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    low_sat = (channel_max - channel_min) < 18
    bg = ((rgb[:, :, 0] < 40) & (rgb[:, :, 1] < 40) & (rgb[:, :, 2] < 40)) | ((channel_max < 55) & low_sat)
    fg = (~bg).astype(np.uint8) * 255
    if fg.max() == 0:
        return None
    fg = _morph(fg, open_px=1, close_px=2)
    fg = _largest_component(fg)
    return fg.astype(np.float32) / 255.0


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def _score_mask_candidate(
    image_rgb: np.ndarray,
    box: BoundingBox,
    label: str,
    det_score: float,
    mask_prob: np.ndarray,
    fg_prior: Optional[np.ndarray],
    cfg: dict,
) -> MaskCandidate:
    thr = float(cfg["postprocess"]["binarize_threshold"])
    binary = (mask_prob >= thr).astype(np.uint8) * 255

    if binary.max() == 0:
        return MaskCandidate(
            mask=mask_prob.astype(np.float32),
            binary=binary,
            score=-1e9,
            label=label,
            det_score=det_score,
            box=box,
            reason="empty",
        )

    mask_box = _mask_bbox(binary)
    mask_area = max(1.0, float((binary > 0).sum()))
    img_area = float(binary.shape[0] * binary.shape[1])

    box_mask = np.zeros_like(binary, dtype=np.uint8)
    x0, y0, x1, y1 = box.to_xyxy()
    x0 = max(0, min(binary.shape[1] - 1, x0))
    y0 = max(0, min(binary.shape[0] - 1, y0))
    x1 = max(x0 + 1, min(binary.shape[1], x1))
    y1 = max(y0 + 1, min(binary.shape[0], y1))
    box_mask[y0:y1, x0:x1] = 255

    inside_box = float(np.logical_and(binary > 0, box_mask > 0).sum()) / mask_area
    box_tightness = _box_iou(mask_box, box)
    compact = _compactness(binary)
    edge_den = _edge_density(image_rgb, binary)
    border_touch = _border_touch_ratio(binary)

    fg_iou = 0.0
    if fg_prior is not None:
        fg_bin = (fg_prior >= 0.5).astype(np.uint8) * 255
        fg_iou = _mask_iou(binary, fg_bin)

    n, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    islands = 0
    if n > 1:
        min_island = max(8, int(0.0008 * img_area))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_island:
                islands += 1

    score = (
        det_score * float(cfg["mask_scoring"]["detector_score_weight"])
        + fg_iou * float(cfg["mask_scoring"]["foreground_iou_weight"])
        + inside_box * float(cfg["mask_scoring"]["interior_fill_weight"])
        + compact * float(cfg["mask_scoring"]["compactness_weight"])
        + edge_den * float(cfg["mask_scoring"]["edge_density_weight"])
        + box_tightness * float(cfg["mask_scoring"]["box_tightness_weight"])
        - border_touch * float(cfg["mask_scoring"]["border_touch_penalty"])
        - float(islands) * float(cfg["mask_scoring"]["tiny_island_penalty"])
    )

    reason = (
        f"det={det_score:.3f} "
        f"fg_iou={fg_iou:.3f} "
        f"inside={inside_box:.3f} "
        f"tight={box_tightness:.3f} "
        f"compact={compact:.3f} "
        f"border={border_touch:.3f}"
    )

    return MaskCandidate(
        mask=mask_prob.astype(np.float32),
        binary=binary,
        score=float(score),
        label=label,
        det_score=float(det_score),
        box=box,
        reason=reason,
    )


# -----------------------------------------------------------------------------
# Merge / prune
# -----------------------------------------------------------------------------

def _merge_candidates(
    candidates: List[MaskCandidate],
    image_shape: Tuple[int, int],
    cfg: dict,
) -> np.ndarray:
    if not candidates:
        return np.zeros(image_shape, dtype=np.uint8)

    keep_mode = str(cfg["postprocess"]["keep_mode"]).strip().lower()
    merge_iou = float(cfg["postprocess"]["merge_iou"])

    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

    if keep_mode == "largest":
        best = max(candidates, key=lambda c: (c.binary > 0).sum())
        return best.binary.copy()

    if keep_mode == "all":
        out = np.zeros(image_shape, dtype=np.uint8)
        for c in candidates:
            out = np.maximum(out, c.binary)
        return out

    selected: List[MaskCandidate] = []
    for cand in candidates:
        should_add = True
        for prev in selected:
            if _mask_iou(cand.binary, prev.binary) >= merge_iou:
                should_add = False
                break
        if should_add:
            selected.append(cand)

    out = np.zeros(image_shape, dtype=np.uint8)
    for c in selected:
        out = np.maximum(out, c.binary)
    return out


def _postprocess_mask(binary_u8: np.ndarray, cfg: dict) -> np.ndarray:
    out = binary_u8.copy()

    if bool(cfg["postprocess"].get("fill_holes", True)):
        out = _fill_holes(out)

    out = _morph(
        out,
        open_px=int(cfg["postprocess"].get("open_px", 0)),
        close_px=int(cfg["postprocess"].get("close_px", 0)),
    )

    keep_min_area_ratio = float(cfg["postprocess"].get("keep_min_area_ratio", 0.002))
    min_area = max(8, int(round(keep_min_area_ratio * out.shape[0] * out.shape[1])))

    n, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(out, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def _binary_to_soft_alpha(binary_u8: np.ndarray, cfg: dict) -> np.ndarray:
    feather = float(cfg["postprocess"].get("feather_px", 1.2))
    alpha = binary_u8.astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    alpha = np.clip(alpha, 0.0, 255.0)
    low = int(cfg["postprocess"].get("alpha_hard_zero_below", 8))
    high = int(cfg["postprocess"].get("alpha_hard_full_above", 248))
    alpha[alpha <= low] = 0.0
    alpha[alpha >= high] = 255.0
    return alpha.astype(np.uint8)


# -----------------------------------------------------------------------------
# Embedded alpha path
# -----------------------------------------------------------------------------

def _use_embedded_alpha_if_good(original: Image.Image, cfg: dict) -> Optional[ExtractedObject]:
    if not bool(cfg["input"].get("prefer_embedded_alpha", True)):
        return None

    rgba = original.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)

    if alpha.max() == 0:
        return None

    if not bool(cfg["input"].get("trust_embedded_alpha_if_nontrivial", True)):
        return None

    unique = np.unique(alpha)
    area_ratio = float((alpha > 0).sum()) / float(alpha.shape[0] * alpha.shape[1])

    if len(unique) <= 2 and area_ratio > 0.995 and int(alpha.min()) >= 250:
        return None

    cropped, bbox = _crop_rgba_to_alpha(rgba, pad=int(cfg["input"]["pad_to_crop"]))
    cropped_alpha = np.array(cropped.getchannel("A"), dtype=np.uint8)
    return ExtractedObject(
        rgba=cropped,
        alpha=_np_to_pil_l(cropped_alpha),
        mask_binary=_np_to_pil_l(np.where(cropped_alpha > 0, 255, 0).astype(np.uint8)),
        bbox_xyxy=bbox,
        method="embedded_alpha",
        label="",
        debug={"used_embedded_alpha": True},
    )


# -----------------------------------------------------------------------------
# Main extraction
# -----------------------------------------------------------------------------

def extract_object(
    image: Image.Image,
    object_label: str,
    cfg: dict,
    device: str = "cuda",
    debug_dir: Optional[Path] = None,
    object_understanding: Optional[Dict[str, Any]] = None,
) -> ExtractedObject:
    load_dotenv()
    torch_device = _choose_device(device)
    rgb_image = image.convert("RGB")
    cfg = _build_extraction_cfg(cfg, object_understanding)

    if debug_dir is not None:
        _ensure_dir(debug_dir)

    embedded = _use_embedded_alpha_if_good(image, cfg)
    if embedded is not None:
        embedded.label = object_label
        if object_understanding is not None:
            embedded.debug["used_object_understanding"] = True
            embedded.debug["object_understanding_debug"] = cfg.get("_object_understanding_debug", {})
        return embedded

    work_img, scale = _maybe_resize(rgb_image, int(cfg["input"]["max_side"]))
    rgb_np = _pil_to_np_rgb(work_img)

    boxes = _detect_label_boxes(work_img, object_label, torch_device, cfg)

    fg_prior = _foreground_prior_mask(work_img, torch_device, cfg)
    if fg_prior is None and bool(cfg["fallback"].get("use_rembg_if_available", True)):
        fg_prior = _mask_from_rembg(work_img)
    if fg_prior is None and bool(cfg["fallback"].get("use_dark_bg_heuristic", True)):
        fg_prior = _mask_from_dark_bg_heuristic(work_img)

    candidates: List[MaskCandidate] = []

    for box in boxes:
        sam_masks = _sam2_masks_from_box(work_img, box, torch_device, cfg)
        if not sam_masks:
            continue
        for sm in sam_masks:
            cand = _score_mask_candidate(
                image_rgb=rgb_np,
                box=box,
                label=object_label,
                det_score=box.score,
                mask_prob=sm,
                fg_prior=fg_prior,
                cfg=cfg,
            )
            if cand.score > -1e8:
                candidates.append(cand)

    method = "grounding_dino+sam2"

    if not candidates:
        if fg_prior is not None:
            binary = (fg_prior >= float(cfg["postprocess"]["binarize_threshold"])).astype(np.uint8) * 255
            candidates = [
                MaskCandidate(
                    mask=fg_prior.astype(np.float32),
                    binary=binary,
                    score=0.0,
                    label=object_label,
                    det_score=0.0,
                    box=_mask_bbox(binary),
                    reason="foreground_fallback",
                )
            ]
            method = "foreground_fallback"
        else:
            raise RuntimeError(
                "Extraction failed: no label-conditioned masks and no fallback foreground mask available."
            )

    merged = _merge_candidates(candidates, image_shape=rgb_np.shape[:2], cfg=cfg)
    final_binary = _postprocess_mask(merged, cfg)

    if final_binary.max() == 0:
        raise RuntimeError("Extraction failed: final mask is empty after postprocessing.")

    final_alpha = _binary_to_soft_alpha(final_binary, cfg)

    rgba_np = np.dstack([rgb_np, final_alpha]).astype(np.uint8)
    rgba_np[final_alpha == 0, :3] = 0
    rgba = _np_to_pil_rgba(rgba_np)

    cropped_rgba, bbox = _crop_rgba_to_alpha(rgba, pad=int(cfg["input"]["pad_to_crop"]))
    cropped_alpha = np.array(cropped_rgba.getchannel("A"), dtype=np.uint8)
    cropped_binary = np.where(cropped_alpha > 0, 255, 0).astype(np.uint8)

    if scale < 1.0:
        inv_scale = 1.0 / scale
        x0 = int(round(bbox[0] * inv_scale))
        y0 = int(round(bbox[1] * inv_scale))
        x1 = int(round(bbox[2] * inv_scale))
        y1 = int(round(bbox[3] * inv_scale))
        x0 = max(0, min(image.size[0] - 1, x0))
        y0 = max(0, min(image.size[1] - 1, y0))
        x1 = max(x0 + 1, min(image.size[0], x1))
        y1 = max(y0 + 1, min(image.size[1], y1))

        full_alpha = cv2.resize(final_alpha, image.size, interpolation=cv2.INTER_LINEAR)
        full_rgb = _pil_to_np_rgb(image)
        full_rgba = np.dstack([full_rgb, full_alpha]).astype(np.uint8)
        full_rgba[full_alpha == 0, :3] = 0
        rgba = _np_to_pil_rgba(full_rgba)
        cropped_rgba, bbox = _crop_rgba_to_alpha(rgba, pad=int(cfg["input"]["pad_to_crop"]))
        cropped_alpha = np.array(cropped_rgba.getchannel("A"), dtype=np.uint8)
        cropped_binary = np.where(cropped_alpha > 0, 255, 0).astype(np.uint8)

    if debug_dir is not None and bool(cfg["debug"].get("save_debug", True)):
        if boxes:
            _save_boxes_overlay(work_img, boxes, debug_dir / "01_detected_boxes.png")
        if fg_prior is not None:
            _np_to_pil_l((fg_prior * 255.0).astype(np.uint8)).save(debug_dir / "02_foreground_prior.png")
        _save_mask_overlay(work_img, final_binary, debug_dir / "03_final_mask_overlay.png")
        _np_to_pil_l(final_binary).save(debug_dir / "04_final_mask_binary.png")
        _np_to_pil_l(final_alpha).save(debug_dir / "05_final_alpha.png")
        cropped_rgba.save(debug_dir / "06_cropped_rgba.png")

        with (debug_dir / "07_candidates.txt").open("w", encoding="utf-8") as f:
            for i, c in enumerate(sorted(candidates, key=lambda x: x.score, reverse=True)):
                f.write(
                    f"[{i}] score={c.score:.4f} "
                    f"det={c.det_score:.4f} "
                    f"label={c.label} "
                    f"box={c.box.to_xyxy()} "
                    f"{c.reason}\n"
                )

    return ExtractedObject(
        rgba=cropped_rgba,
        alpha=_np_to_pil_l(cropped_alpha),
        mask_binary=_np_to_pil_l(cropped_binary),
        bbox_xyxy=bbox,
        method=method,
        label=object_label,
        debug={
            "num_boxes": len(boxes),
            "num_candidates": len(candidates),
            "used_foreground_prior": fg_prior is not None,
            "scale_for_processing": scale,
            "used_object_understanding": object_understanding is not None,
            "object_understanding_debug": cfg.get("_object_understanding_debug", {}),
        },
    )


def extract_object_from_path(
    image_path: str | Path,
    object_label: str,
    cfg: dict,
    device: str = "cuda",
    debug_dir: Optional[Path] = None,
    object_understanding: Optional[Dict[str, Any]] = None,
) -> ExtractedObject:
    image = Image.open(Path(image_path))
    return extract_object(
        image=image,
        object_label=object_label,
        cfg=cfg,
        device=device,
        debug_dir=debug_dir,
        object_understanding=object_understanding,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label-aware object extraction using Grounding DINO + SAM2 + BiRefNet fallback."
    )
    parser.add_argument("--image", required=True, type=str, help="Path to the object/source image.")
    parser.add_argument("--object-label", required=True, type=str, help="Label text, e.g. 'lemon' or 'lemon and mint'.")
    parser.add_argument("--object-understanding", required=False, type=str, help="Path to object_understanding.json")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], type=str)
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.config))
    object_understanding = (
        _load_object_understanding(Path(args.object_understanding))
        if args.object_understanding
        else None
    )

    output_dir = Path(args.output)
    debug_dir = output_dir / "debug"
    _ensure_dir(output_dir)
    _ensure_dir(debug_dir)

    result = extract_object_from_path(
        image_path=args.image,
        object_label=args.object_label,
        cfg=cfg,
        device=args.device,
        debug_dir=debug_dir,
        object_understanding=object_understanding,
    )

    result.rgba.save(output_dir / "object_rgba.png")
    result.alpha.save(output_dir / "object_alpha.png")
    result.mask_binary.save(output_dir / "object_mask.png")

    meta = {
        "method": result.method,
        "label": result.label,
        "bbox_xyxy": result.bbox_xyxy,
        "debug": result.debug,
    }
    with (output_dir / "meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print(str(output_dir / "object_rgba.png"))


if __name__ == "__main__":
    main()