from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageFilter
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, pipeline
from PIL import Image, ImageDraw, ImageFilter

try:
    from rembg import remove
except ImportError:
    remove = None

# ----------------------------
# Global lazy-loaded caches
# ----------------------------
_DET_PROCESSOR = None
_DET_MODEL = None
_DEPTH_PIPE = None

SURFACE_LABELS = [
    "countertop",
    "kitchen counter",
    "table",
    "desk",
    "shelf",
    "island",
]

DEFAULT_SCENE_CFG: Dict[str, Any] = {
    "models": {
        "detector_id": "IDEA-Research/grounding-dino-tiny",
        "depth_id": "depth-anything/Depth-Anything-V2-Small-hf",
    },
    "detection": {
        "object_threshold": 0.28,
        "object_text_threshold": 0.20,
        "support_threshold": 0.26,
        "support_text_threshold": 0.20,
        "context_threshold": 0.25,
        "context_text_threshold": 0.20,
        "max_side": 1024,
    },
    "placement": {
        "attempt_index": 0,
        "top_k_to_keep": 12,
        "max_supports_to_try": 6,
        "candidate_step_x_divisor": 6,
        "edge_margin_ratio": 0.06,
        "min_scale_ratio": 0.78,
        "max_scale_ratio": 1.28,
        "avoid_overlap_weight": 6.0,
        "total_overlap_weight": 2.0,
        "depth_std_weight": 1.8,
        "center_offset_weight": 0.35,
        "support_band_weight": 1.8,
        "perspective_weight": 2.2,
        "favor_empty_space_weight": 1.2,
        "size_consistency_weight": 2.4,
    },
    "support_geometry": {
        "thin_height_ratio": 0.035,
        "plane_min_height_ratio": 0.055,
        "edge_depth_slope_max": 0.035,
        "plane_depth_slope_min": 0.045,
        "plane_depth_variance_min": 0.015,
        "plane_back_start_ratio": 0.10,
        "plane_front_end_ratio": 0.62,
        "edge_contact_offset_px": 1,
        "plane_candidate_step_y_divisor": 6,
    },
    "support_preferences": {
        "preferred_labels": ["table", "desk", "island", "countertop", "kitchen counter"],
        "disfavored_labels": ["shelf"],
        "prefer_mode": "plane",
        "label_match_bonus": 2.5,
        "mode_match_bonus": 1.5,
        "disfavored_label_penalty": 2.0,
    },
    "shadow": {
        "enabled": True,
        "opacity": 0.40,
        "softness_px": 30,
        "squash_y": 0.20,
        "shear_x": 0.10,
        "offset_x": 0,
        "offset_y": 8,
    },
    "output": {
        "save_debug_overlay": False,
    },
}


def _save_detection_overlay(
        image: Image.Image,
        boxes: List["BoundingBox"],
        out_path: Path,
        title: Optional[str] = None,
) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    for box in boxes:
        x0, y0, x1, y1 = box.to_int_tuple()
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)

        label_text = box.label.strip() if box.label else "object"
        text = f"{label_text} {box.score:.2f}"

        tx0 = x0
        ty0 = max(0, y0 - 18)
        tx1 = min(overlay.width, tx0 + max(60, 7 * len(text)))
        ty1 = min(overlay.height, ty0 + 16)

        draw.rectangle([tx0, ty0, tx1, ty1], fill=(255, 0, 0))
        draw.text((tx0 + 3, ty0 + 1), text, fill=(255, 255, 255))

    if title:
        banner_h = 22
        draw.rectangle([0, 0, overlay.width, banner_h], fill=(0, 0, 0))
        draw.text((6, 3), title, fill=(255, 255, 255))

    overlay.save(out_path)


def _save_support_perspective_overlay(
        image: Image.Image,
        persp: "SupportPerspective",
        out_path: Path,
) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    pts = [tuple(map(int, p)) for p in persp.polygon.tolist()]
    if len(pts) >= 3:
        draw.polygon(pts, outline=(0, 255, 0), width=3)

    rows = persp.x_bounds_by_y
    step = max(1, len(rows) // 30)
    for y, xl, xr in rows[::step]:
        draw.line([(xl, y), (xr, y)], fill=(255, 255, 0), width=1)

    overlay.save(out_path)


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
    return _deep_update(DEFAULT_SCENE_CFG, user_cfg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _pil_rgba_to_np_rgba(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA"))


def _np_rgba_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _alpha_bbox(alpha_u8: np.ndarray, thr: int = 10) -> Tuple[int, int, int, int]:
    ys, xs = np.where(alpha_u8 > thr)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _crop_to_alpha(img_rgba: Image.Image, pad: int = 4) -> Image.Image:
    arr = _pil_rgba_to_np_rgba(img_rgba)
    alpha = arr[:, :, 3]
    x1, y1, x2, y2 = _alpha_bbox(alpha, thr=10)
    h, w = alpha.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return _np_rgba_to_pil(arr[y1: y2 + 1, x1: x2 + 1, :])


def _largest_alpha_component(alpha: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    binary = alpha > 0
    visited = np.zeros((h, w), dtype=bool)
    best_count = 0
    best_coords = None

    for yy in range(h):
        for xx in range(w):
            if not binary[yy, xx] or visited[yy, xx]:
                continue
            stack = [(yy, xx)]
            visited[yy, xx] = True
            coords = []

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(coords) > best_count:
                best_count = len(coords)
                best_coords = coords

    out = np.zeros_like(alpha, dtype=np.uint8)
    if best_coords is not None:
        for yy, xx in best_coords:
            out[yy, xx] = 255
    return out


def _apply_morph(mask: np.ndarray, erode_px: int = 0, dilate_px: int = 0) -> np.ndarray:
    out = mask.copy()
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        out = cv2.erode(out, k, iterations=1)
    if dilate_px > 0:
        k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        out = cv2.dilate(out, k, iterations=1)
    return out


def _mask_from_rembg(image: Image.Image, alpha_threshold: int = 20) -> Optional[Image.Image]:
    if remove is None:
        return None

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


def _extract_from_dark_bg_fallback(image: Image.Image) -> Optional[Image.Image]:
    rgba = image.convert("RGBA")
    arr = np.array(rgba).copy()
    rgb = arr[:, :, :3].astype(np.int16)

    bg = (rgb[:, :, 0] < 40) & (rgb[:, :, 1] < 40) & (rgb[:, :, 2] < 40)
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    low_sat = (channel_max - channel_min) < 18
    bg |= ((channel_max < 55) & low_sat)

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    if alpha.max() == 0:
        return None

    alpha_img = Image.fromarray(alpha, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.MaxFilter(3))
    alpha_img = alpha_img.filter(ImageFilter.MinFilter(3))
    alpha = np.array(alpha_img, dtype=np.uint8)

    arr[:, :, 3] = alpha
    arr[alpha == 0, :3] = 0

    out = Image.fromarray(arr, mode="RGBA")
    out = _crop_to_alpha(out, pad=4)
    return out


def _clean_extracted_rgba(rgba: Image.Image) -> Image.Image:
    arr = np.array(rgba.convert("RGBA")).copy()
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    alpha[alpha < 20] = 0
    alpha = _largest_alpha_component(alpha)
    alpha = _apply_morph(alpha, erode_px=1, dilate_px=1)

    alpha_img = Image.fromarray(alpha, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.MaxFilter(3))
    alpha_img = alpha_img.filter(ImageFilter.MinFilter(3))
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.7))

    alpha = np.array(alpha_img, dtype=np.uint8)
    alpha[alpha < 18] = 0
    alpha[alpha > 245] = 255

    low_alpha = alpha < 64
    rgb[low_alpha] = 0

    dark_pixels = (rgb[:, :, 0] < 12) & (rgb[:, :, 1] < 12) & (rgb[:, :, 2] < 12)
    alpha[dark_pixels & (alpha < 180)] = 0

    arr[:, :, :3] = rgb
    arr[:, :, 3] = alpha

    out = Image.fromarray(arr, mode="RGBA")
    out = _crop_to_alpha(out, pad=4)

    softened_alpha = out.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.45))
    out.putalpha(softened_alpha)
    return out


def _extract_object_rgba(image: Image.Image) -> Optional[Image.Image]:
    rembg_mask = _mask_from_rembg(image, alpha_threshold=20)

    if rembg_mask is not None:
        mask_np = np.array(rembg_mask, dtype=np.uint8)
        if mask_np.max() > 0:
            rgba_np = np.array(image.convert("RGBA")).copy()
            rgba_np[:, :, 3] = mask_np
            rgba_np[mask_np == 0, :3] = 0
            out = Image.fromarray(rgba_np, mode="RGBA")
            out = _clean_extracted_rgba(out)

            alpha = np.array(out.getchannel("A"), dtype=np.uint8)
            if alpha.max() > 0:
                return out

    fallback = _extract_from_dark_bg_fallback(image)
    if fallback is not None:
        fallback = _clean_extracted_rgba(fallback)
        alpha = np.array(fallback.getchannel("A"), dtype=np.uint8)
        if alpha.max() > 0:
            return fallback

    return None


def _resize_rgba(arr_rgba: np.ndarray, scale: float) -> np.ndarray:
    new_w = max(1, int(round(arr_rgba.shape[1] * scale)))
    new_h = max(1, int(round(arr_rgba.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(arr_rgba, (new_w, new_h), interpolation=interp)


def _place_rgba_over_rgb(scene_rgb: np.ndarray, obj_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    out = scene_rgb.copy()
    H, W = scene_rgb.shape[:2]
    h, w = obj_rgba.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return out

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    patch = out[y1:y2, x1:x2, :].astype(np.float32)
    obj_patch = obj_rgba[oy1:oy2, ox1:ox2, :].astype(np.float32)
    alpha = obj_patch[:, :, 3:4] / 255.0

    blended = obj_patch[:, :, :3] * alpha + patch * (1.0 - alpha)
    out[y1:y2, x1:x2, :] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def _render_contact_shadow(
        scene_rgb: np.ndarray,
        obj_alpha_u8: np.ndarray,
        x: int,
        y: int,
        softness_px: int,
        opacity: float,
        squash_y: float,
        shear_x: float,
        offset_x: int,
        offset_y: int,
) -> np.ndarray:
    h, w = obj_alpha_u8.shape[:2]

    yy = np.linspace(0, 1, h).reshape(h, 1)
    weight = np.clip((yy - 0.65) / 0.35, 0, 1)
    base = (obj_alpha_u8.astype(np.float32) / 255.0) * weight
    base = (base * 255).astype(np.uint8)

    M = np.array([[1.0, shear_x, 0.0], [0.0, squash_y, 0.0]], dtype=np.float32)
    shadow = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, softness_px / 6.0))

    out = scene_rgb.copy()
    H, W = out.shape[:2]
    sx = x + offset_x
    sy = y + int(h * 0.55) + offset_y

    x1 = max(0, sx)
    y1 = max(0, sy)
    x2 = min(W, sx + w)
    y2 = min(H, sy + h)
    if x1 >= x2 or y1 >= y2:
        return out

    rx1 = x1 - sx
    ry1 = y1 - sy
    rx2 = rx1 + (x2 - x1)
    ry2 = ry1 + (y2 - y1)

    alpha = shadow[ry1:ry2, rx1:rx2].astype(np.float32)[:, :, None] / 255.0
    patch = out[y1:y2, x1:x2, :].astype(np.float32)
    darkened = patch * (1.0 - opacity * alpha)
    out[y1:y2, x1:x2, :] = np.clip(darkened, 0, 255).astype(np.uint8)
    return out


def _open_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def _maybe_resize_for_detection(image: Image.Image, max_side: int = 1024) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale == 1.0:
        return image, 1.0
    new_w = max(32, int(round(w * scale / 8) * 8))
    new_h = max(32, int(round(h * scale / 8) * 8))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale


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
        return (
            int(round(self.x0)),
            int(round(self.y0)),
            int(round(self.x1)),
            int(round(self.y1)),
        )


@dataclass
class Placement:
    x: int
    y: int
    scale: float


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


@dataclass
class SupportPerspective:
    support: SupportGeometry
    polygon: np.ndarray
    front_line: Tuple[float, float]
    back_line: Tuple[float, float]
    x_bounds_by_y: List[Tuple[int, int, int]]
    corners: np.ndarray
    H_unit_to_img: np.ndarray
    H_img_to_unit: np.ndarray


def _scale_box(box: BoundingBox, inv_scale: float) -> BoundingBox:
    return BoundingBox(
        x0=box.x0 * inv_scale,
        y0=box.y0 * inv_scale,
        x1=box.x1 * inv_scale,
        y1=box.y1 * inv_scale,
        score=box.score,
        label=box.label,
    )


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


def _detect_objects(
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

    resized, scale = _maybe_resize_for_detection(image, max_side=max_side or cfg["detection"]["max_side"])
    inv_scale = 1.0 / scale

    processor, model = _get_detector(device, cfg)
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
        detections.append(
            _scale_box(BoundingBox(x0, y0, x1, y1, score=score, label=label), inv_scale).clamp(*image.size))

    return detections


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

    dmin = float(depth.min())
    dmax = float(depth.max())
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    depth = (depth - dmin) / (dmax - dmin)

    h = depth.shape[0]
    top = np.median(depth[: max(1, h // 5), :])
    bottom = np.median(depth[-max(1, h // 5):, :])
    if bottom < top:
        depth = 1.0 - depth
    return depth


def _depth_at_box_base(depth_map: np.ndarray, box: BoundingBox) -> float:
    h, w = depth_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0 + box.width() * 0.15))))
    x1 = max(x0 + 1, min(w, int(round(box.x1 - box.width() * 0.15))))
    y0 = max(0, min(h - 1, int(round(box.y1 - max(2.0, box.height() * 0.10)))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.5
    return float(np.median(patch))


def _local_depth_stats(depth_map: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[float, float]:
    H, W = depth_map.shape
    x0 = max(0, min(W - 1, x))
    y0 = max(0, min(H - 1, y))
    x1 = max(x0 + 1, min(W, x + w))
    y1 = max(y0 + 1, min(H, y + h))
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.5, 1.0
    return float(np.median(patch)), float(np.std(patch))


def _estimate_reference_scale_from_neighbors(existing_boxes: List[BoundingBox],
                                             candidate_center: Tuple[float, float]) -> Optional[float]:
    if not existing_boxes:
        return None
    cx, cy = candidate_center
    scored = []
    for b in existing_boxes:
        bx, by = b.centre()
        dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        scored.append((dist, b.height()))
    scored.sort(key=lambda t: t[0])
    top = [h for _, h in scored[:3]]
    if not top:
        return None
    return float(np.median(top))


def _choose_target_size(
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

    min_scale_ratio = float(cfg["placement"]["min_scale_ratio"]) if cfg else 0.78
    max_scale_ratio = float(cfg["placement"]["max_scale_ratio"]) if cfg else 1.28

    if candidate_center is not None:
        neighbor_h = _estimate_reference_scale_from_neighbors(existing_same_objects, candidate_center)
        if neighbor_h is not None:
            target_h = neighbor_h
            if depth_map is not None and candidate_depth is not None and existing_same_objects:
                ref_depth = np.median([_depth_at_box_base(depth_map, b) for b in
                                       existing_same_objects[: min(3, len(existing_same_objects))]])
                ratio = float(candidate_depth / max(0.12, ref_depth))
                scale = float(np.clip(ratio ** 0.55, min_scale_ratio, max_scale_ratio))
                target_h *= scale
            target_h = int(round(np.clip(target_h, scene_h * 0.04, scene_h * 0.22)))
            target_w = int(round(target_h * aspect))
            return max(20, target_w), max(20, target_h)

    if existing_same_objects:
        base_box = sorted(existing_same_objects, key=lambda b: b.area(), reverse=True)[0]
        target_h = float(base_box.height())
        if depth_map is not None and candidate_depth is not None:
            ref_depth = _depth_at_box_base(depth_map, base_box)
            ratio = float(candidate_depth / max(0.12, ref_depth))
            scale = float(np.clip(ratio ** 0.55, min_scale_ratio, max_scale_ratio))
            target_h *= scale
        target_h = int(round(np.clip(target_h, scene_h * 0.04, scene_h * 0.22)))
        target_w = int(round(target_h * aspect))
        return max(20, target_w), max(20, target_h)

    if support_box is not None:
        support_factor = float(np.clip(support_box.width() / max(1.0, scene_w), 0.18, 0.75))
        target_w = int(round(scene_w * (0.07 + 0.11 * support_factor)))
        if candidate_depth is not None:
            depth_scale = float(np.interp(candidate_depth, [0.0, 1.0], [0.78, 1.16]))
            target_w = int(round(target_w * depth_scale))
        target_h = int(round(target_w / max(0.1, aspect)))
        target_h = int(round(np.clip(target_h, scene_h * 0.04, scene_h * 0.24)))
        target_w = int(round(target_h * aspect))
        return max(20, target_w), max(20, target_h)

    target_w = int(round(scene_w * 0.10))
    target_h = int(round(target_w / max(0.1, aspect)))
    target_h = int(round(np.clip(target_h, scene_h * 0.04, scene_h * 0.20)))
    target_w = int(round(target_h * aspect))
    return max(20, target_w), max(20, target_h)


def _filter_support_boxes(boxes: List[BoundingBox], image_size: Tuple[int, int]) -> List[BoundingBox]:
    w, h = image_size
    min_area = w * h * 0.02
    min_width = w * 0.22
    min_height = h * 0.025
    max_height = h * 0.22

    filtered: List[BoundingBox] = []
    for box in boxes:
        if box.area() < min_area:
            continue
        if box.width() < min_width:
            continue
        if box.height() < min_height:
            continue
        if box.height() > max_height:
            continue
        if box.y0 > h * 0.78:
            continue
        filtered.append(box)
    return filtered


def _support_depth_profile(depth_map: np.ndarray, box: BoundingBox) -> tuple[float, float, float]:
    h, w = depth_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0))))
    x1 = max(x0 + 1, min(w, int(round(box.x1))))
    y0 = max(0, min(h - 1, int(round(box.y0))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))

    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0 or patch.shape[0] < 3:
        return 0.0, 0.0, 0.0

    row_medians = np.median(patch, axis=1)
    depth_slope = float(row_medians[-1] - row_medians[0])
    depth_variance = float(np.std(row_medians))
    overall_std = float(np.std(patch))
    return depth_slope, depth_variance, overall_std


def _classify_support_geometry(box: BoundingBox, image_size: Tuple[int, int], depth_map: Optional[np.ndarray],
                               cfg: dict) -> SupportGeometry:
    scene_w, scene_h = image_size
    height_ratio = box.height() / max(1.0, scene_h)
    width_ratio = box.width() / max(1.0, scene_w)
    cy_ratio = box.centre()[1] / max(1.0, scene_h)

    if depth_map is not None:
        depth_slope, depth_variance, overall_std = _support_depth_profile(depth_map, box)
    else:
        depth_slope, depth_variance, overall_std = 0.0, 0.0, 0.0

    geom_cfg = cfg["support_geometry"]

    thin_height_ratio = float(geom_cfg["thin_height_ratio"])
    plane_min_height_ratio = float(geom_cfg["plane_min_height_ratio"])
    edge_depth_slope_max = float(geom_cfg["edge_depth_slope_max"])
    plane_depth_slope_min = float(geom_cfg["plane_depth_slope_min"])
    plane_depth_variance_min = float(geom_cfg["plane_depth_variance_min"])

    is_thin = height_ratio <= thin_height_ratio
    is_low_in_image = cy_ratio >= 0.48
    has_plane_depth = (depth_slope >= plane_depth_slope_min) or (depth_variance >= plane_depth_variance_min)

    if is_thin and depth_slope <= edge_depth_slope_max:
        mode = "edge"
    elif height_ratio >= plane_min_height_ratio and is_low_in_image and has_plane_depth:
        mode = "plane"
    elif height_ratio >= plane_min_height_ratio and has_plane_depth:
        mode = "plane"
    else:
        mode = "edge"

    if mode == "plane":
        y_min = int(round(box.y0 + box.height() * float(geom_cfg["plane_back_start_ratio"])))
        y_max = int(round(box.y0 + box.height() * float(geom_cfg["plane_front_end_ratio"])))
        y_min = max(int(round(box.y0)), min(y_min, int(round(box.y1)) - 1))
        y_max = max(y_min + 1, min(int(round(box.y1)), y_max))
        score = (
                width_ratio * 2.0
                + height_ratio * 3.0
                + max(0.0, depth_slope) * 4.0
                + depth_variance * 3.0
                + cy_ratio
        )
    else:
        edge_y = int(round(box.y0 + float(geom_cfg["edge_contact_offset_px"])))
        y_min = edge_y
        y_max = edge_y
        score = (
                width_ratio * 2.5
                + (1.0 - min(0.2, height_ratio)) * 1.5
                + (1.0 - min(0.2, max(0.0, depth_slope))) * 1.0
                + (1.0 - min(0.2, depth_variance)) * 0.8
        )

    _ = overall_std

    return SupportGeometry(
        box=box,
        mode=mode,
        plane_y_min=y_min,
        plane_y_max=y_max,
        depth_slope=float(depth_slope),
        depth_variance=float(depth_variance),
        score=float(score),
    )


def _build_support_geometries(boxes: List[BoundingBox], image_size: Tuple[int, int], depth_map: Optional[np.ndarray],
                              cfg: dict) -> List[SupportGeometry]:
    geoms = [_classify_support_geometry(box, image_size, depth_map, cfg) for box in boxes]
    geoms.sort(key=lambda g: g.score, reverse=True)
    return geoms


def _support_preference_adjustment(support: SupportGeometry, cfg: dict) -> float:
    prefs = cfg.get("support_preferences", {})
    preferred_labels = [str(x).strip().lower() for x in prefs.get("preferred_labels", [])]
    disfavored_labels = [str(x).strip().lower() for x in prefs.get("disfavored_labels", [])]
    prefer_mode = str(prefs.get("prefer_mode", "any")).strip().lower()

    label_match_bonus = float(prefs.get("label_match_bonus", 2.5))
    mode_match_bonus = float(prefs.get("mode_match_bonus", 1.5))
    disfavored_label_penalty = float(prefs.get("disfavored_label_penalty", 2.0))

    label = (support.box.label or "").strip().lower()

    adjustment = 0.0
    if label in preferred_labels:
        adjustment -= label_match_bonus
    if label in disfavored_labels:
        adjustment += disfavored_label_penalty
    if prefer_mode in {"plane", "edge"} and support.mode == prefer_mode:
        adjustment -= mode_match_bonus
    return adjustment


def _candidate_positions_on_support(
        support: SupportGeometry,
        target_w: int,
        target_h: int,
        scene_w: int,
        scene_h: int,
        cfg: dict,
) -> List[tuple[int, int]]:
    box = support.box
    edge_margin = box.width() * float(cfg["placement"].get("edge_margin_ratio", 0.06))
    usable_left = int(round(box.x0 + edge_margin))
    usable_right = int(round(box.x1 - edge_margin))

    xs = []
    step_x = max(8, target_w // max(2, int(cfg["placement"].get("candidate_step_x_divisor", 6))))
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
        step_y = max(
            5,
            target_h // max(2, int(cfg["support_geometry"].get("plane_candidate_step_y_divisor", 6))),
        )
        y = support.plane_y_min
        while y <= support.plane_y_max:
            ys.append(y)
            y += step_y
        ys.append(support.plane_y_max)
        ys = list(dict.fromkeys(ys))

    out = []
    for foot_y in ys:
        for xx in xs:
            out.append(
                (
                    max(0, min(scene_w - target_w, xx)),
                    max(0, min(scene_h - 1, foot_y)),
                )
            )
    return out


def _project_unit_point(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([u, v, 1.0], dtype=np.float32)
    q = H @ p
    q /= max(1e-6, q[2])
    return float(q[0]), float(q[1])


def _candidate_positions_on_perspective_support(
        persp: SupportPerspective,
        target_w: int,
        target_h: int,
        scene_w: int,
        scene_h: int,
        cfg: dict,
) -> List[tuple[int, int]]:
    out: List[tuple[int, int]] = []

    v_count = max(6, int(cfg["support_geometry"].get("plane_candidate_step_y_divisor", 6)) + 2)
    u_count = max(10, int(cfg["placement"].get("candidate_step_x_divisor", 6)) * 2 + 2)

    v_samples = np.linspace(0.12, 0.82, v_count)
    u_samples = np.linspace(0.06, 0.94, u_count)

    edge_margin_ratio = float(cfg["placement"].get("edge_margin_ratio", 0.06))

    for v in v_samples:
        left_x, _ = _project_unit_point(persp.H_unit_to_img, 0.0, float(v))
        right_x, _ = _project_unit_point(persp.H_unit_to_img, 1.0, float(v))
        usable_w = right_x - left_x
        if usable_w < max(20.0, target_w * 0.75):
            continue

        local_margin = edge_margin_ratio + (1.0 - min(1.0, usable_w / max(1.0, target_w * 3.0))) * 0.06

        for u in u_samples:
            if u < local_margin or u > 1.0 - local_margin:
                continue

            foot_x, foot_y = _project_unit_point(persp.H_unit_to_img, float(u), float(v))
            x = int(round(foot_x - target_w * 0.5))
            y = int(round(foot_y))

            out.append((
                max(0, min(scene_w - target_w, x)),
                max(0, min(scene_h - 1, y)),
            ))

    return list(dict.fromkeys(out))


def _normalize_map(arr: np.ndarray, high_percentile: float = 95.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    hi = float(np.percentile(arr, high_percentile))
    if hi <= 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / hi, 0.0, 1.0)


def _compute_scene_structure_maps(scene_image: Image.Image, depth_map: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    rgb = _pil_to_np_rgb(scene_image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mag = cv2.GaussianBlur(grad_mag, (0, 0), 0.8)
    grad_mag = _normalize_map(grad_mag, high_percentile=94.0)

    if depth_map is not None:
        dx = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = np.sqrt(dx * dx + dy * dy)
        depth_grad = cv2.GaussianBlur(depth_grad, (0, 0), 0.8)
        depth_grad = _normalize_map(depth_grad, high_percentile=94.0)
    else:
        depth_grad = np.zeros_like(gray, dtype=np.float32)

    return {
        "gray": gray,
        "grad_mag": grad_mag,
        "depth_grad": depth_grad,
    }


def _estimate_support_plane_mask(
        structure_maps: Dict[str, np.ndarray],
        support: SupportGeometry,
        depth_map: Optional[np.ndarray],
) -> np.ndarray:
    gray = structure_maps["gray"]
    grad_mag = structure_maps["grad_mag"]
    depth_grad = structure_maps["depth_grad"]

    H, W = gray.shape
    x0 = max(0, min(W - 1, int(round(support.box.x0))))
    x1 = max(x0 + 1, min(W, int(round(support.box.x1))))
    y0 = max(0, min(H - 1, int(round(support.box.y0))))
    y1 = max(y0 + 1, min(H, int(round(support.box.y1))))

    grad_patch = grad_mag[y0:y1, x0:x1]
    depth_patch = depth_grad[y0:y1, x0:x1]

    signal = grad_patch * 1.4 + depth_patch * 1.8

    if depth_map is not None:
        raw_depth = depth_map[y0:y1, x0:x1].astype(np.float32)
        row_median = np.median(raw_depth, axis=1, keepdims=True)
        depth_dev = np.abs(raw_depth - row_median)

        hi = float(np.percentile(depth_dev, 92))
        if hi > 1e-6:
            depth_dev = np.clip(depth_dev / hi, 0.0, 1.0)
        else:
            depth_dev = np.zeros_like(depth_dev)

        signal += depth_dev * 0.9

    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 1.2)

    thresh = max(0.18, float(np.percentile(signal, 68)))
    mask = (signal > thresh).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)

    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best_idx, 255, 0).astype(np.uint8)


def _fit_support_perspective(
        support: SupportGeometry,
        structure_maps: Dict[str, np.ndarray],
        depth_map: Optional[np.ndarray],
) -> Optional[SupportPerspective]:
    if support.mode != "plane":
        return None

    mask = _estimate_support_plane_mask(structure_maps, support, depth_map)
    if mask.max() == 0:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return None

    box_x0 = int(round(support.box.x0))
    box_y0 = int(round(support.box.y0))

    H_local, _ = mask.shape
    x_bounds_by_y: List[Tuple[int, int, int]] = []
    local_rows: List[Tuple[int, int, int]] = []
    for yy in range(H_local):
        row = np.where(mask[yy] > 0)[0]
        if len(row) < 2:
            continue
        x_left = int(row.min())
        x_right = int(row.max())
        local_rows.append((yy, x_left, x_right))
        x_bounds_by_y.append((yy + box_y0, x_left + box_x0, x_right + box_x0))

    if len(x_bounds_by_y) < 8:
        return None

    top_rows = x_bounds_by_y[: max(3, len(x_bounds_by_y) // 8)]
    bot_rows = x_bounds_by_y[-max(3, len(x_bounds_by_y) // 8):]

    def fit_line(rows: List[Tuple[int, int, int]]) -> Tuple[float, float]:
        pts = []
        for y, xl, xr in rows:
            pts.append((xl, y))
            pts.append((xr, y))
        pts = np.array(pts, dtype=np.float32)
        if len(pts) < 2:
            return 0.0, float(rows[0][0] if rows else 0.0)

        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = [float(np.asarray(v).reshape(-1)[0]) for v in line]

        if abs(vx) < 1e-6:
            return 0.0, y0

        m = vy / vx
        b = y0 - m * x0
        return float(m), float(b)

    back_line = fit_line(top_rows)
    front_line = fit_line(bot_rows)

    def robust_edge(rows: List[Tuple[int, int, int]], side: str) -> Tuple[float, float]:
        pts = []
        for y, xl, xr in rows:
            x = xl if side == "left" else xr
            pts.append((x, y))
        pts = np.array(pts, dtype=np.float32)
        if len(pts) < 2:
            return 0.0, float(pts[0, 0] if len(pts) else 0.0)

        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = [float(np.asarray(v).reshape(-1)[0]) for v in line]
        if abs(vy) < 1e-6:
            return 0.0, x0

        m = vx / vy
        b = x0 - m * y0
        return float(m), float(b)

    n_band = max(3, len(local_rows) // 8)
    top_local = local_rows[:n_band]
    bot_local = local_rows[-n_band:]

    l_m, l_b = robust_edge(local_rows, "left")
    r_m, r_b = robust_edge(local_rows, "right")

    top_y = float(np.median([y for y, _, _ in top_local]))
    bot_y = float(np.median([y for y, _, _ in bot_local]))

    corners = np.array(
        [
            [l_m * top_y + l_b + box_x0, top_y + box_y0],
            [r_m * top_y + r_b + box_x0, top_y + box_y0],
            [r_m * bot_y + r_b + box_x0, bot_y + box_y0],
            [l_m * bot_y + l_b + box_x0, bot_y + box_y0],
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
    H_unit_to_img = cv2.getPerspectiveTransform(unit, corners)
    H_img_to_unit = cv2.getPerspectiveTransform(corners, unit)

    return SupportPerspective(
        support=support,
        polygon=corners.astype(np.int32),
        front_line=front_line,
        back_line=back_line,
        x_bounds_by_y=x_bounds_by_y,
        corners=corners,
        H_unit_to_img=H_unit_to_img,
        H_img_to_unit=H_img_to_unit,
    )


def _perspective_depth_ratio(persp: SupportPerspective, x: float, y: float) -> float:
    fm, fb = persp.front_line
    bm, bb = persp.back_line

    front_y = fm * x + fb
    back_y = bm * x + bb

    denom = max(1.0, front_y - back_y)
    t = (y - back_y) / denom
    return float(np.clip(t, 0.0, 1.0))


def _point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0


def _candidate_surface_occupancy_score(
        structure_maps: Dict[str, np.ndarray],
        support: SupportGeometry,
        candidate_box: BoundingBox,
) -> float:
    gray = structure_maps["gray"]
    grad_mag = structure_maps["grad_mag"]
    depth_grad = structure_maps["depth_grad"]

    H, W = gray.shape
    x0 = max(0, min(W - 1, int(round(candidate_box.x0))))
    x1 = max(x0 + 1, min(W, int(round(candidate_box.x1))))
    foot_y = int(round(candidate_box.y1))

    contact_h = max(8, int(round(candidate_box.height() * 0.18)))
    below_h = max(2, int(round(candidate_box.height() * 0.03)))

    if support.mode == "plane":
        y0 = max(int(round(support.box.y0)), foot_y - contact_h)
        y1 = min(int(round(support.box.y1)), foot_y + below_h)
    else:
        y0 = max(0, foot_y - contact_h)
        y1 = min(H, foot_y + 2)

    if y1 <= y0 or x1 <= x0:
        return 0.0

    gray_patch = gray[y0:y1, x0:x1]
    grad_patch = grad_mag[y0:y1, x0:x1]
    depth_patch = depth_grad[y0:y1, x0:x1]

    edge_density = float(np.mean(grad_patch > 0.28))
    depth_density = float(np.mean(depth_patch > 0.25))
    texture_std = float(np.std(gray_patch))

    above_y0 = max(0, foot_y - contact_h)
    above_y1 = max(above_y0 + 1, min(H, foot_y))
    below_y0 = max(0, min(H - 1, foot_y))
    below_y1 = max(below_y0 + 1, min(H, foot_y + below_h))

    above = gray[above_y0:above_y1, x0:x1]
    below = gray[below_y0:below_y1, x0:x1]

    seam_diff = 0.0
    if above.size > 0 and below.size > 0:
        seam_diff = float(abs(np.mean(above) - np.mean(below)))

    support_x0 = max(0, min(W - 1, int(round(support.box.x0))))
    support_x1 = max(support_x0 + 1, min(W, int(round(support.box.x1))))

    baseline_mask = np.ones((y1 - y0, support_x1 - support_x0), dtype=bool)
    rel_x0 = max(0, x0 - support_x0)
    rel_x1 = min(support_x1 - support_x0, x1 - support_x0)
    baseline_mask[:, rel_x0:rel_x1] = False

    support_gray_patch = gray[y0:y1, support_x0:support_x1]
    support_grad_patch = grad_mag[y0:y1, support_x0:support_x1]
    support_depth_patch = depth_grad[y0:y1, support_x0:support_x1]

    baseline_edge = 0.0
    baseline_depth = 0.0
    baseline_texture = 0.0

    if baseline_mask.any():
        baseline_edge = float(np.mean(support_grad_patch[baseline_mask] > 0.28))
        baseline_depth = float(np.mean(support_depth_patch[baseline_mask] > 0.25))
        baseline_texture = float(np.std(support_gray_patch[baseline_mask]))

    rel_edge = max(0.0, edge_density - baseline_edge)
    rel_depth = max(0.0, depth_density - baseline_depth)
    rel_texture = max(0.0, texture_std - baseline_texture)

    col_signal = np.mean(grad_patch + 0.75 * depth_patch, axis=0)
    occupied_columns = float(np.mean(col_signal > 0.42)) if col_signal.size else 0.0

    return (
            rel_edge * 2.2
            + rel_depth * 1.8
            + rel_texture * 1.4
            + seam_diff * 0.8
            + occupied_columns * 1.2
    )


def _support_contact_band_bounds(
        support: SupportGeometry,
        candidate_h: int,
        image_h: int,
) -> tuple[int, int]:
    if support.mode == "plane":
        y0 = max(int(round(support.box.y0)), support.plane_y_min)
        y1 = min(int(round(support.box.y1)), support.plane_y_max + max(3, candidate_h // 18))
    else:
        edge_y = support.plane_y_min
        y0 = max(0, edge_y - max(6, candidate_h // 14))
        y1 = min(image_h, edge_y + max(4, candidate_h // 22))
    return y0, max(y0 + 1, y1)


def _build_support_occupancy_profile(
        structure_maps: Dict[str, np.ndarray],
        support: SupportGeometry,
        candidate_h: int,
) -> np.ndarray:
    gray = structure_maps["gray"]
    grad_mag = structure_maps["grad_mag"]
    depth_grad = structure_maps["depth_grad"]

    H, W = gray.shape
    sx0 = max(0, min(W - 1, int(round(support.box.x0))))
    sx1 = max(sx0 + 1, min(W, int(round(support.box.x1))))
    sy0, sy1 = _support_contact_band_bounds(support, candidate_h=candidate_h, image_h=H)

    gray_patch = gray[sy0:sy1, sx0:sx1]
    grad_patch = grad_mag[sy0:sy1, sx0:sx1]
    depth_patch = depth_grad[sy0:sy1, sx0:sx1]

    if gray_patch.size == 0:
        return np.zeros((sx1 - sx0,), dtype=np.float32)

    local_texture = np.std(gray_patch, axis=0)
    local_edges = np.mean(grad_patch > 0.26, axis=0).astype(np.float32)
    local_depth_edges = np.mean(depth_patch > 0.22, axis=0).astype(np.float32)

    raw = local_edges * 1.9 + local_depth_edges * 1.7 + local_texture * 2.1

    if raw.size == 0:
        return raw.astype(np.float32)

    raw = cv2.GaussianBlur(raw.reshape(1, -1), (0, 0), sigmaX=max(1.2, candidate_h / 18.0)).reshape(-1)
    baseline = float(np.percentile(raw, 35))
    raw = np.clip(raw - baseline, 0.0, None)

    hi = float(np.percentile(raw, 92)) if raw.size else 0.0
    if hi > 1e-6:
        raw = np.clip(raw / hi, 0.0, 1.0)
    else:
        raw = np.zeros_like(raw, dtype=np.float32)

    return raw.astype(np.float32)


def _build_support_obstacle_map(
        structure_maps: Dict[str, np.ndarray],
        support: SupportGeometry,
        candidate_h: int,
        depth_map: Optional[np.ndarray],
) -> Dict[str, Any]:
    gray = structure_maps["gray"]
    grad_mag = structure_maps["grad_mag"]
    depth_grad = structure_maps["depth_grad"]

    H, W = gray.shape
    sx0 = max(0, min(W - 1, int(round(support.box.x0))))
    sx1 = max(sx0 + 1, min(W, int(round(support.box.x1))))

    scan_up = max(candidate_h, int(round(support.box.height() * 2.0)))

    if support.mode == "plane":
        anchor_y = support.plane_y_max
        sy0 = max(0, anchor_y - scan_up)
        sy1 = min(H, anchor_y + max(4, candidate_h // 16))
    else:
        anchor_y = support.plane_y_min
        sy0 = max(0, anchor_y - int(round(candidate_h * 0.95)))
        sy1 = min(H, anchor_y + max(3, candidate_h // 20))

    if sy1 <= sy0 or sx1 <= sx0:
        return {"map": np.zeros((1, 1), dtype=np.float32), "x0": sx0, "y0": sy0}

    grad_patch = grad_mag[sy0:sy1, sx0:sx1]
    depth_grad_patch = depth_grad[sy0:sy1, sx0:sx1]

    signal = grad_patch * 1.8 + depth_grad_patch * 1.5

    if depth_map is not None:
        depth_patch = depth_map[sy0:sy1, sx0:sx1].astype(np.float32)
        row_ref = np.percentile(depth_patch, 45, axis=1, keepdims=True)
        depth_resid = np.abs(depth_patch - row_ref)

        hi = float(np.percentile(depth_resid, 92))
        if hi > 1e-6:
            depth_resid = np.clip(depth_resid / hi, 0.0, 1.0)
        else:
            depth_resid = np.zeros_like(depth_resid, dtype=np.float32)

        signal = signal + depth_resid * 1.2

    row_base = np.percentile(signal, 35, axis=1, keepdims=True)
    signal = np.clip(signal - row_base, 0.0, None)

    signal = cv2.GaussianBlur(
        signal.astype(np.float32),
        (0, 0),
        sigmaX=max(1.2, candidate_h / 20.0),
        sigmaY=max(1.2, candidate_h / 14.0),
    )

    hi = float(np.percentile(signal, 92))
    if hi > 1e-6:
        signal = np.clip(signal / hi, 0.0, 1.0)
    else:
        signal = np.zeros_like(signal, dtype=np.float32)

    mask = (signal > 0.34).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(signal, dtype=np.float32)

    min_area = max(18, int(round(candidate_h * 0.10)))
    min_height = max(6, int(round(candidate_h * 0.14)))

    for idx in range(1, num_labels):
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]

        if area < min_area:
            continue
        if h < min_height:
            continue

        component = labels == idx
        cleaned[component] = np.maximum(cleaned[component], signal[component])

    cleaned = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=max(1.0, candidate_h / 24.0))
    cleaned = np.clip(cleaned, 0.0, 1.0).astype(np.float32)

    return {
        "map": cleaned,
        "x0": sx0,
        "y0": sy0,
    }


def _candidate_obstacle_metrics(
        obstacle_map_info: Dict[str, Any],
        candidate_box: BoundingBox,
) -> tuple[float, float]:
    obstacle_map = obstacle_map_info["map"]
    ox0 = int(obstacle_map_info["x0"])
    oy0 = int(obstacle_map_info["y0"])

    if obstacle_map.size == 0:
        return 0.0, 0.0

    rel_x0 = max(0, int(round(candidate_box.x0)) - ox0)
    rel_y0 = max(0, int(round(candidate_box.y0)) - oy0)
    rel_x1 = min(obstacle_map.shape[1], int(round(candidate_box.x1)) - ox0)
    rel_y1 = min(obstacle_map.shape[0], int(round(candidate_box.y1)) - oy0)

    if rel_x1 <= rel_x0 or rel_y1 <= rel_y0:
        return 0.0, 0.0

    window = obstacle_map[rel_y0:rel_y1, rel_x0:rel_x1]
    if window.size == 0:
        return 0.0, 0.0

    mean_block = float(np.mean(window))
    strong_block = float(np.mean(window > 0.52))
    return mean_block, strong_block


def _candidate_support_occupancy_metrics(
        occupancy_profile: np.ndarray,
        support: SupportGeometry,
        candidate_box: BoundingBox,
) -> tuple[float, float]:
    if occupancy_profile.size == 0:
        return 0.0, 0.0

    sx0 = int(round(support.box.x0))
    rel_x0 = max(0, int(round(candidate_box.x0)) - sx0)
    rel_x1 = min(len(occupancy_profile), int(round(candidate_box.x1)) - sx0)

    if rel_x1 <= rel_x0:
        return 0.0, 0.0

    window = occupancy_profile[rel_x0:rel_x1]
    if window.size == 0:
        return 0.0, 0.0

    mean_occ = float(np.mean(window))
    heavy_occ = float(np.mean(window > 0.55))
    return mean_occ, heavy_occ


def _rank_placements(
        scene_image: Image.Image,
        object_rgba: Image.Image,
        support_boxes: List[BoundingBox],
        avoid_boxes: List[BoundingBox],
        existing_same_objects: List[BoundingBox],
        depth_map: Optional[np.ndarray],
        cfg: dict,
) -> Tuple[List[PlacementCandidate], Optional[SupportPerspective]]:
    scene_w, scene_h = scene_image.size
    support_geometries = _build_support_geometries(support_boxes, scene_image.size, depth_map, cfg)
    structure_maps = _compute_scene_structure_maps(scene_image, depth_map)

    if not support_geometries:
        return [], None

    support_candidates: List[Tuple[SupportGeometry, float]] = []
    for support in support_geometries:
        preference_adjustment = _support_preference_adjustment(support, cfg)
        support_priority = support.score - preference_adjustment
        support_candidates.append((support, support_priority))

    support_candidates.sort(key=lambda x: x[1], reverse=True)

    best_support, _ = support_candidates[0]
    persp_model = _fit_support_perspective(best_support, structure_maps, depth_map) if best_support.mode == "plane" else None

    seed_depth = _depth_at_box_base(depth_map, best_support.box) if depth_map is not None else 0.5

    base_w, base_h = _choose_target_size(
        scene_size=scene_image.size,
        obj_size=object_rgba.size,
        support_box=best_support.box,
        existing_same_objects=existing_same_objects,
        depth_map=depth_map,
        candidate_depth=seed_depth,
        candidate_center=best_support.box.centre(),
        cfg=cfg,
    )

    occupancy_profile = _build_support_occupancy_profile(
        structure_maps=structure_maps,
        support=best_support,
        candidate_h=base_h,
    )
    obstacle_map_info = _build_support_obstacle_map(
        structure_maps=structure_maps,
        support=best_support,
        candidate_h=base_h,
        depth_map=depth_map,
    )

    if persp_model is not None:
        candidate_positions = _candidate_positions_on_perspective_support(
            persp=persp_model,
            target_w=base_w,
            target_h=base_h,
            scene_w=scene_w,
            scene_h=scene_h,
            cfg=cfg,
        )
    else:
        candidate_positions = _candidate_positions_on_support(best_support, base_w, base_h, scene_w, scene_h, cfg)

    valid_candidates: List[PlacementCandidate] = []
    fallback_candidates: List[PlacementCandidate] = []

    for x, foot_y in candidate_positions:
        sample_y = foot_y - int(base_h * 0.20) if best_support.mode == "edge" else foot_y - int(base_h * 0.08)

        depth_median, depth_std = (0.5, 0.25)
        if depth_map is not None:
            depth_median, depth_std = _local_depth_stats(
                depth_map,
                x=x + int(base_w * 0.1),
                y=max(0, sample_y),
                w=max(6, int(base_w * 0.8)),
                h=max(4, int(base_h * 0.22)),
            )

        center_guess = (x + base_w * 0.5, foot_y - base_h * 0.5)
        target_w, target_h = _choose_target_size(
            scene_size=scene_image.size,
            obj_size=object_rgba.size,
            support_box=best_support.box,
            existing_same_objects=existing_same_objects,
            depth_map=depth_map,
            candidate_depth=depth_median,
            candidate_center=center_guess,
            cfg=cfg,
        )

        if persp_model is not None:
            t = _perspective_depth_ratio(
                persp_model,
                x + target_w * 0.5,
                foot_y,
            )
            persp = float(np.interp(t, [0.0, 1.0], [0.82, 1.22]))
        else:
            if best_support.mode == "plane":
                plane_rel = np.clip(
                    (foot_y - best_support.plane_y_min) / max(1.0,
                                                              best_support.plane_y_max - best_support.plane_y_min),
                    0.0,
                    1.0,
                )
                persp = float(np.interp(plane_rel, [0.0, 1.0], [0.90, 1.18]))
            else:
                persp = 1.0

        target_w = int(round(target_w * persp))
        target_h = int(round(target_h * persp))

        obj_y = int(round(foot_y - target_h))
        candidate_box = BoundingBox(x, obj_y, x + target_w, obj_y + target_h)

        if candidate_box.y0 < 0 or candidate_box.x0 < 0 or candidate_box.x1 > scene_w or candidate_box.y1 > scene_h:
            continue

        if persp_model is not None:
            base_left = (candidate_box.x0 + candidate_box.width() * 0.15, candidate_box.y1)
            base_mid = (candidate_box.x0 + candidate_box.width() * 0.50, candidate_box.y1)
            base_right = (candidate_box.x0 + candidate_box.width() * 0.85, candidate_box.y1)

            inside_count = sum(
                _point_in_polygon(persp_model.polygon, px, py)
                for px, py in (base_left, base_mid, base_right)
            )
            if inside_count < 2:
                continue

        overlaps = [_iou(candidate_box, other) for other in avoid_boxes]
        max_overlap = max(overlaps, default=0.0)
        sum_overlap = sum(overlaps)

        center_offset = abs(candidate_box.centre()[0] - best_support.box.centre()[0]) / max(1.0,
                                                                                            best_support.box.width())

        if best_support.mode == "plane":
            support_band_pref = abs(
                (foot_y - best_support.plane_y_min) / max(1.0,
                                                          best_support.plane_y_max - best_support.plane_y_min) - 0.35
            )
        else:
            support_band_pref = 0.0

        predicted_w, predicted_h = _choose_target_size(
            scene_size=scene_image.size,
            obj_size=object_rgba.size,
            support_box=best_support.box,
            existing_same_objects=existing_same_objects,
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

        mode_penalty = 0.0
        if best_support.mode == "edge":
            mode_penalty = max(0.0, best_support.depth_slope - 0.05) * 3.0
        else:
            mode_penalty = max(0.0, 0.035 - best_support.depth_slope) * 4.0

        mean_occ, heavy_occ = _candidate_support_occupancy_metrics(
            occupancy_profile=occupancy_profile,
            support=best_support,
            candidate_box=candidate_box,
        )
        mean_block, strong_block = _candidate_obstacle_metrics(
            obstacle_map_info=obstacle_map_info,
            candidate_box=candidate_box,
        )

        score = (
                max_overlap * 1.25
                + sum_overlap * 0.35
                + mean_occ * 1.2
                + heavy_occ * 2.2
                + mean_block * 3.6
                + strong_block * 7.5
                + depth_std * cfg["placement"]["depth_std_weight"]
                + center_offset * cfg["placement"]["center_offset_weight"]
                + support_band_pref * cfg["placement"]["support_band_weight"]
                + (1.0 - min(1.0, persp)) * cfg["placement"]["perspective_weight"]
                + empty_space_score * 0.20
                + size_consistency * cfg["placement"]["size_consistency_weight"]
                + mode_penalty
        )

        cand = PlacementCandidate(
            placement=Placement(
                x=int(candidate_box.x0),
                y=int(candidate_box.y0),
                scale=target_h / max(1.0, object_rgba.size[1]),
            ),
            score=float(score),
            debug=(
                f"label={best_support.box.label} "
                f"mode={best_support.mode} "
                f"support_score={best_support.score:.3f} "
                f"overlap={max_overlap:.3f} "
                f"sum_overlap={sum_overlap:.3f} "
                f"mean_occ={mean_occ:.3f} "
                f"heavy_occ={heavy_occ:.3f} "
                f"mean_block={mean_block:.3f} "
                f"strong_block={strong_block:.3f} "
                f"depth_std={depth_std:.3f} "
                f"depth_slope={best_support.depth_slope:.3f} "
                f"perspective={persp:.3f}"
            ),
        )

        if strong_block <= 0.20 and mean_block <= 0.33 and heavy_occ <= 0.45:
            valid_candidates.append(cand)
        fallback_candidates.append(cand)

    ranked = valid_candidates if valid_candidates else fallback_candidates
    ranked.sort(key=lambda c: c.score)
    return ranked[: int(cfg["placement"].get("top_k_to_keep", 12))], persp_model


def _choose_placement_from_ranked(candidates: List[PlacementCandidate], attempt_index: int) -> Placement:
    if not candidates:
        raise RuntimeError("No plausible placement candidates found.")
    idx = max(0, min(attempt_index, len(candidates) - 1))
    return candidates[idx].placement


class ContextualImageEditor:
    def __init__(self, cfg: Dict[str, Any], device: str) -> None:
        load_dotenv()
        self.cfg = cfg
        self.requested_device = device
        self.device = self._choose_device(device)

    @staticmethod
    def _choose_device(device: str) -> torch.device:
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda was requested but CUDA is not available.")
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_object_rgba(self, object_path: Path) -> Image.Image:
        original = Image.open(object_path)
        obj = original.convert("RGBA")

        alpha = np.array(obj.getchannel("A"), dtype=np.uint8)
        bbox = obj.getchannel("A").getbbox()

        has_real_transparency = int(alpha.min()) < 250 and int(alpha.max()) > 0
        alpha_covers_entire_image = bbox is not None and bbox == (0, 0, obj.width, obj.height)

        if has_real_transparency and not alpha_covers_entire_image:
            return _crop_to_alpha(obj, pad=4)

        extracted = _extract_object_rgba(original.convert("RGB"))
        if extracted is not None:
            return extracted

        if int(alpha.max()) == 0:
            raise RuntimeError(
                "Failed to extract the object from --object-image. Install rembg or provide a transparent PNG cutout."
            )

        if alpha_covers_entire_image:
            raise RuntimeError(
                "The object image appears to be a normal photo with background, but extraction failed. "
                "Install rembg or provide a cleaner object image / transparent PNG."
            )

        return _crop_to_alpha(obj, pad=4)

    def run(self, scene_path: Path, object_path: Path, object_label: str, output_dir: Path) -> Path:
        _ensure_dir(output_dir)
        debug_dir = output_dir / "debug"
        _ensure_dir(debug_dir)

        scene = _open_rgb(scene_path)
        obj_rgba = self._load_object_rgba(object_path)
        obj_rgba.save(debug_dir / "object_rgba.png")

        depth_map = _estimate_depth_map(scene, self.device, self.cfg)

        raw_support_boxes = _detect_objects(
            scene,
            labels=SURFACE_LABELS,
            device=self.device,
            cfg=self.cfg,
            threshold=self.cfg["detection"]["support_threshold"],
            text_threshold=self.cfg["detection"]["support_text_threshold"],
        )
        _save_detection_overlay(
            scene,
            raw_support_boxes,
            debug_dir / "support_boxes_raw.png",
            title="Raw support detections",
        )

        support_boxes = _filter_support_boxes(raw_support_boxes, scene.size)
        _save_detection_overlay(
            scene,
            support_boxes,
            debug_dir / "support_boxes_filtered.png",
            title="Filtered support detections",
        )

        existing_same_objects = _detect_objects(
            scene,
            labels=[object_label],
            device=self.device,
            cfg=self.cfg,
            threshold=self.cfg["detection"]["object_threshold"],
            text_threshold=self.cfg["detection"]["object_text_threshold"],
        )
        _save_detection_overlay(
            scene,
            existing_same_objects,
            debug_dir / "existing_same_objects.png",
            title=f"Detected: {object_label}",
        )

        avoid_boxes = existing_same_objects

        ranked_candidates, best_support_perspective = _rank_placements(
            scene_image=scene,
            object_rgba=obj_rgba,
            support_boxes=support_boxes,
            avoid_boxes=avoid_boxes,
            existing_same_objects=existing_same_objects,
            depth_map=depth_map,
            cfg=self.cfg,
        )

        if best_support_perspective is not None:
            _save_support_perspective_overlay(
                scene,
                best_support_perspective,
                debug_dir / "best_support_perspective.png",
            )

        placement = _choose_placement_from_ranked(
            ranked_candidates,
            int(self.cfg["placement"].get("attempt_index", 0)),
        )

        scene_np = _pil_to_np_rgb(scene)
        obj_arr = _pil_rgba_to_np_rgba(obj_rgba)
        obj_scaled = _resize_rgba(obj_arr, placement.scale)
        alpha_s = obj_scaled[:, :, 3]

        shadow_cfg = self.cfg.get("shadow", {})
        if bool(shadow_cfg.get("enabled", True)):
            scene_np = _render_contact_shadow(
                scene_rgb=scene_np,
                obj_alpha_u8=alpha_s,
                x=placement.x,
                y=placement.y,
                softness_px=int(shadow_cfg.get("softness_px", shadow_cfg.get("blur_px", 30))),
                opacity=float(shadow_cfg.get("opacity", 0.40)),
                squash_y=float(shadow_cfg.get("squash_y", 0.20)),
                shear_x=float(shadow_cfg.get("shear_x", 0.10)),
                offset_x=int(shadow_cfg.get("offset_x", 0)),
                offset_y=int(shadow_cfg.get("offset_y", 8)),
            )

        composite_np = _place_rgba_over_rgb(scene_np, obj_scaled, placement.x, placement.y)
        composite = _np_rgb_to_pil(composite_np)

        out_path = output_dir / "composite_raw.png"
        composite.save(out_path)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene-aware local object insertion using grounding + depth + direct compositing.")
    parser.add_argument("--scene", required=True, type=str, help="Path to the scene image.")
    parser.add_argument("--object-image", required=True, type=str, help="Path to the object image.")
    parser.add_argument("--object-label", required=True, type=str, help="Object label, e.g. lemon or dutch oven.")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device preference.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.config))
    editor = ContextualImageEditor(cfg=cfg, device=args.device)
    out_path = editor.run(
        scene_path=Path(args.scene),
        object_path=Path(args.object_image),
        object_label=args.object_label,
        output_dir=Path(args.output),
    )
    print(str(out_path))


if __name__ == "__main__":
    main()