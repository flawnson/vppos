from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter
from rembg import remove
from transformers import (
    AutoModel,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

try:
    from diffusers import FluxFillPipeline  # type: ignore

    _FLUX_FILL_AVAILABLE = True
except ImportError:
    FluxFillPipeline = None  # type: ignore
    _FLUX_FILL_AVAILABLE = False

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

try:
    from transformers import Sam2Model, Sam2Processor  # type: ignore

    _SAM_AVAILABLE = True
except ImportError:
    Sam2Model = None  # type: ignore
    Sam2Processor = None  # type: ignore
    _SAM_AVAILABLE = False


# ----------------------------
# Global lazy-loaded caches
# ----------------------------
_DET_PROCESSOR = None
_DET_MODEL = None
_SAM_PROCESSOR = None
_SAM_MODEL = None
_DEPTH_PIPE = None
_FILL_PIPE = None
_CLIP_PROCESSOR = None
_CLIP_MODEL = None


DEFAULT_CONFIG = {
    "models": {
        "detector_id": "IDEA-Research/grounding-dino-tiny",
        "sam_id": "facebook/sam2.1-hiera-tiny",
        "depth_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "semantic_clip_id": "openai/clip-vit-base-patch32",
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
    "extraction": {
        "use_rembg": True,
        "rembg_alpha_threshold": 20,
        "sam_box_padding": 6,
        "matte_edge_blur": 0.7,
        "keep_largest_component": True,
        "grabcut_refine": True,
        "grabcut_iters": 2,
        "combine_mode": "hybrid",
        "dark_bg_fallback": True,
        "erode_px": 1,
        "dilate_px": 1,
    },
    "placement": {
        "max_supports_to_try": 6,
        "candidate_step_x_divisor": 6,
        "attempt_index": 0,
        "top_k_to_keep": 12,
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
    "semantic_scoring": {
        "enabled": True,
        "surface_prompt_templates": [
            "a stable flat top surface for placing a {label}",
            "a realistic place to put a {label}",
            "an upward facing support surface for a {label}",
        ],
        "scene_prompt_templates": [
            "a {label} naturally placed on this surface",
            "a {label} resting on this surface",
        ],
        "support_label_bonus": 0.22,
        "scene_patch_bonus": 0.16,
    },
    "placement_search": {
        "scale_multipliers": [0.82, 0.92, 1.0, 1.1, 1.22],
        "surface_semantic_weight": 2.2,
        "scene_semantic_weight": 1.4,
        "flatness_weight": 1.6,
        "physics_weight": 2.6,
        "occupancy_hard_threshold": 0.33,
        "surface_semantic_min": 0.18,
        "scene_semantic_min": 0.17,
        "min_support_coverage_ratio": 0.58,
        "base_contact_ratio": 0.72,
        "allow_edge_contacts": False,
    },
    "shadow": {
        "enabled": True,
        "contact_opacity": 0.30,
        "contact_blur_px": 2.0,
        "cast_opacity": 0.17,
        "cast_blur_px": 7.0,
        "cast_length_scale": 0.70,
        "squash_ratio": 0.35,
        "shear_strength": 0.22,
        "shadow_color_mode": "surface_tinted",
        "ambient_occlusion_band_ratio": 0.08,
        "local_contrast_influence": 0.4,
        "shadow_softness_influence": 0.35,
    },
    "collision": {
        "enabled": True,
        "max_iou": 0.01,
        "max_intersection_ratio_of_candidate": 0.02,
        "use_occupancy_map": True,
        "occupancy_blur_px": 9,
        "occupancy_threshold": 0.20,
        "occupancy_penalty_weight": 10.0,
    },
    "refinement": {
        "enabled": True,
        "model_id": "black-forest-labs/FLUX.1-Fill-dev",
        "steps": 28,
        "guidance_scale": 18.0,
        "max_sequence_length": 256,
        "mask_expand_px": 18,
        "mask_blur_px": 5,
        "core_erode_px": 10,
        "contact_band_ratio": 0.12,
        "contact_expand_px": 12,
        "crop_context_ratio": 0.55,
        "crop_min_px": 384,
        "crop_max_px": 1280,
        "padding_mask_crop": 32,
        "blend_strength": 1.0,
        "prompt_template": "Blend the placed {label} naturally into the scene. Preserve the object's exact shape, color, surface texture, branding, and fine details. Match local lighting, contact, shadows, reflections, and occlusion only around the object boundary.",
        "seed": 0,
        "cpu_offload": True,
        "skip_if_unavailable": True,
        "object_harmonize_strength": 0.18,
    },
    "output": {
        "timestamp_outputs": True,
        "save_candidate_overlay": False,
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
        return (
            int(round(self.x0)),
            int(round(self.y0)),
            int(round(self.x1)),
            int(round(self.y1)),
        )


@dataclass
class ExtractedObject:
    rgba: Image.Image
    mask: Image.Image
    bbox: BoundingBox
    label: str


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
    mode: str  # "edge" or "plane"
    plane_y_min: int
    plane_y_max: int
    depth_slope: float
    depth_variance: float
    score: float


SURFACE_LABELS = [
    "countertop",
    "kitchen counter",
    "table",
    "desk",
    "shelf",
    "island",
]
CONTEXT_OBJECT_LABELS: List[str] = [
    "fruit bowl",
    "bowl",
    "plate",
    "cup",
    "mug",
    "glass",
    "bottle",
    "jar",
    "vase",
    "utensil",
    "fork",
    "knife",
    "spoon",
    "napkin",
    "tray",
    "cutting board",
    "board",
    "sink",
    "stove",
    "cabinet",
    "appliance",
    "toaster",
    "kettle",
    "coffee maker",
    "microwave",
    "container",
    "basket",
    "book",
    "box",
    "plant",
    "flower pot",
    "lamp",
    "phone",
    "tablet",
    "laptop",
    "bag",
    "food",
    "fruit",
    "bread",
    "banana",
    "apple",
    "orange",
    "lemon",
]


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> dict:
    if not path:
        return DEFAULT_CONFIG
    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


# ----------------------------
# Model loaders
# ----------------------------
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


def get_depth_pipe(device: torch.device, cfg: dict):
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        device_index = 0 if device.type == "cuda" else -1
        _DEPTH_PIPE = pipeline(
            task="depth-estimation",
            model=cfg["models"]["depth_id"],
            device=device_index,
            token=HF_TOKEN,
        )
    return _DEPTH_PIPE


def get_fill_pipe(device: torch.device, cfg: dict):
    global _FILL_PIPE
    if not _FLUX_FILL_AVAILABLE:
        return None
    if _FILL_PIPE is None:
        refine_cfg = cfg.get("refinement", {})
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        _FILL_PIPE = FluxFillPipeline.from_pretrained(
            refine_cfg.get("model_id", "black-forest-labs/FLUX.1-Fill-dev"),
            torch_dtype=dtype,
            token=HF_TOKEN,
        )
        if device.type == "cuda":
            if refine_cfg.get("cpu_offload", True) and hasattr(_FILL_PIPE, "enable_model_cpu_offload"):
                _FILL_PIPE.enable_model_cpu_offload()
            else:
                _FILL_PIPE.to(device)
        if hasattr(_FILL_PIPE, "vae") and hasattr(_FILL_PIPE.vae, "enable_slicing"):
            _FILL_PIPE.vae.enable_slicing()
        if hasattr(_FILL_PIPE, "vae") and hasattr(_FILL_PIPE.vae, "enable_tiling"):
            _FILL_PIPE.vae.enable_tiling()
    return _FILL_PIPE


def get_clip_model(device: torch.device, cfg: dict):
    global _CLIP_PROCESSOR, _CLIP_MODEL
    if _CLIP_PROCESSOR is None or _CLIP_MODEL is None:
        model_id = cfg["models"].get("semantic_clip_id", "openai/clip-vit-base-patch32")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id, token=HF_TOKEN)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_id, token=HF_TOKEN)
        _CLIP_MODEL.to(device)
        _CLIP_MODEL.eval()
    return _CLIP_PROCESSOR, _CLIP_MODEL


# ----------------------------
# Utility helpers
# ----------------------------
def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def maybe_resize_for_detection(image: Image.Image, max_side: int = 1024) -> tuple[Image.Image, float]:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale == 1.0:
        return image, 1.0
    new_w = max(32, int(round(w * scale / 8) * 8))
    new_h = max(32, int(round(h * scale / 8) * 8))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale


def scale_box(box: BoundingBox, inv_scale: float) -> BoundingBox:
    return BoundingBox(
        x0=box.x0 * inv_scale,
        y0=box.y0 * inv_scale,
        x1=box.x1 * inv_scale,
        y1=box.y1 * inv_scale,
        score=box.score,
        label=box.label,
    )


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


def choose_best_detection(boxes: List[BoundingBox]) -> Optional[BoundingBox]:
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b.score, b.area()))


def crop_to_alpha_bbox(rgba: Image.Image) -> tuple[Image.Image, Image.Image, BoundingBox]:
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        full_box = BoundingBox(0, 0, rgba.width, rgba.height)
        return rgba, alpha, full_box

    x0, y0, x1, y1 = bbox
    pad = 2
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgba.width, x1 + pad)
    y1 = min(rgba.height, y1 + pad)

    cropped_rgba = rgba.crop((x0, y0, x1, y1))
    cropped_alpha = alpha.crop((x0, y0, x1, y1))
    return cropped_rgba, cropped_alpha, BoundingBox(float(x0), float(y0), float(x1), float(y1))


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

    alpha[alpha < 20] = 0
    if cfg["extraction"].get("keep_largest_component", True):
        alpha = _largest_alpha_component(alpha)

    alpha = apply_morph(
        alpha,
        erode_px=int(cfg["extraction"].get("erode_px", 0)),
        dilate_px=int(cfg["extraction"].get("dilate_px", 0)),
    )

    alpha_img = Image.fromarray(alpha, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.MaxFilter(3))
    alpha_img = alpha_img.filter(ImageFilter.MinFilter(3))
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=float(cfg["extraction"].get("matte_edge_blur", 0.6))))

    alpha = np.array(alpha_img, dtype=np.uint8)
    hard = alpha.copy()
    hard[hard < 18] = 0
    hard[hard > 245] = 255

    low_alpha = hard < 64
    rgb[low_alpha] = 0

    dark_pixels = (rgb[:, :, 0] < 12) & (rgb[:, :, 1] < 12) & (rgb[:, :, 2] < 12)
    hard[dark_pixels & (hard < 180)] = 0

    arr[:, :, :3] = rgb
    arr[:, :, 3] = hard
    cleaned = Image.fromarray(arr, mode="RGBA")
    cropped_rgba, cropped_alpha, _ = crop_to_alpha_bbox(cleaned)
    cropped_alpha = cropped_alpha.filter(ImageFilter.GaussianBlur(radius=0.45))
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


def refine_mask_with_sam(image: Image.Image, box: BoundingBox, device: torch.device, cfg: dict) -> Optional[Image.Image]:
    sam_processor, sam_model = get_sam(device, cfg)
    if sam_processor is None or sam_model is None:
        return None

    pad = int(cfg["extraction"].get("sam_box_padding", 6))
    try:
        input_boxes = [[[max(0.0, box.x0 - pad), max(0.0, box.y0 - pad), min(image.width, box.x1 + pad), min(image.height, box.y1 + pad)]]]
        sam_inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = sam_model(**sam_inputs)

        masks = sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"],
            sam_inputs["reshaped_input_sizes"],
        )
        if not masks or len(masks[0]) == 0:
            return None

        best_mask = None
        best_score = -1.0
        candidates = masks[0][0]
        for m in candidates:
            mask_np = (m.numpy() > 0).astype(np.uint8) * 255
            ys, xs = np.where(mask_np > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            mask_area = float((mask_np > 0).sum())
            bbox_area = float(max(1, (x1 - x0 + 1) * (y1 - y0 + 1)))
            fill_ratio = mask_area / bbox_area
            border_touch = float(
                ((mask_np[0, :] > 0).sum() + (mask_np[-1, :] > 0).sum() + (mask_np[:, 0] > 0).sum() + (mask_np[:, -1] > 0).sum())
            ) / max(1.0, (2 * mask_np.shape[0] + 2 * mask_np.shape[1]))
            score = mask_area * fill_ratio * (1.0 - 0.6 * border_touch)
            if 0.05 <= fill_ratio <= 0.97 and score > best_score:
                best_score = score
                best_mask = mask_np

        if best_mask is None:
            best_mask = (candidates[0].numpy() > 0).astype(np.uint8) * 255
        return Image.fromarray(best_mask, mode="L")
    except Exception:
        return None


def refine_mask_grabcut(image: Image.Image, init_mask: Image.Image, iters: int = 2) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    mask = np.array(init_mask, dtype=np.uint8)

    gc_mask = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[mask > 220] = cv2.GC_PR_FGD

    sure_fg = cv2.erode(((mask > 220).astype(np.uint8) * 255), np.ones((5, 5), np.uint8), iterations=1)
    sure_region = cv2.dilate(((mask > 0).astype(np.uint8) * 255), np.ones((7, 7), np.uint8), iterations=1)
    sure_bg = sure_region == 0

    gc_mask[sure_fg > 0] = cv2.GC_FGD
    gc_mask[sure_bg] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, gc_mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def extract_dark_bg_foreground(image: Image.Image, box: BoundingBox) -> Image.Image:
    x0, y0, x1, y1 = box.to_int_tuple()
    crop = image.crop((x0, y0, x1, y1)).convert("RGBA")
    arr = np.array(crop).copy()
    rgb = arr[:, :, :3].astype(np.int16)

    bg = (rgb[:, :, 0] < 40) & (rgb[:, :, 1] < 40) & (rgb[:, :, 2] < 40)
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    low_sat = (channel_max - channel_min) < 18
    bg |= ((channel_max < 55) & low_sat)

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.MaxFilter(3))
    alpha_img = alpha_img.filter(ImageFilter.MinFilter(3))
    alpha = np.array(alpha_img, dtype=np.uint8)

    arr[:, :, 3] = alpha
    arr[alpha == 0, 0] = 0
    arr[alpha == 0, 1] = 0
    arr[alpha == 0, 2] = 0

    out = Image.fromarray(arr, mode="RGBA")
    out, _, _ = crop_to_alpha_bbox(out)
    return out

def intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    return iw * ih


def candidate_intersection_ratio(candidate: BoundingBox, other: BoundingBox) -> float:
    inter = intersection_area(candidate, other)
    denom = max(1.0, candidate.area())
    return inter / denom

def build_occupancy_map(
    image_size: Tuple[int, int],
    occupied_boxes: List[BoundingBox],
    blur_px: int = 9,
) -> np.ndarray:
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

    occ = np.clip(occ, 0.0, 1.0)
    return occ


def occupancy_score_for_box(occupancy_map: np.ndarray, box: BoundingBox) -> float:
    h, w = occupancy_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0))))
    y0 = max(0, min(h - 1, int(round(box.y0))))
    x1 = max(x0 + 1, min(w, int(round(box.x1))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))

    patch = occupancy_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))

def combine_candidate_masks(rembg_mask: Optional[Image.Image], sam_mask: Optional[Image.Image], mode: str = "hybrid") -> Optional[Image.Image]:
    if rembg_mask is None and sam_mask is None:
        return None
    if rembg_mask is None:
        return sam_mask
    if sam_mask is None:
        return rembg_mask

    a = np.array(rembg_mask, dtype=np.uint8) > 0
    b = np.array(sam_mask, dtype=np.uint8) > 0

    if mode == "intersection":
        out = a & b
    elif mode == "union":
        out = a | b
    else:
        inter = a & b
        union = a | b
        out = inter if inter.sum() > 0.55 * min(max(1, a.sum()), max(1, b.sum())) else union

    return Image.fromarray((out.astype(np.uint8) * 255), mode="L")


def extract_reference_object(reference_image: Image.Image, object_label: str, device: torch.device, cfg: dict) -> ExtractedObject:
    detections = detect_objects(
        reference_image,
        labels=[object_label],
        device=device,
        cfg=cfg,
        threshold=cfg["detection"]["object_threshold"],
        text_threshold=cfg["detection"]["object_text_threshold"],
        max_side=min(960, cfg["detection"]["max_side"]),
    )
    best = choose_best_detection(detections)
    if best is None:
        raise RuntimeError(f"Could not detect '{object_label}' in the reference image.")

    rembg_mask = None
    if cfg["extraction"].get("use_rembg", True):
        rembg_mask = mask_from_rembg(reference_image, int(cfg["extraction"].get("rembg_alpha_threshold", 20)))

    sam_mask = refine_mask_with_sam(reference_image, best, device, cfg)
    merged = combine_candidate_masks(rembg_mask=rembg_mask, sam_mask=sam_mask, mode=cfg["extraction"].get("combine_mode", "hybrid"))

    if merged is None and cfg["extraction"].get("dark_bg_fallback", True):
        cleaned_rgba = clean_extracted_object(extract_dark_bg_foreground(reference_image, best), cfg)
        cleaned_mask = cleaned_rgba.getchannel("A")
        return ExtractedObject(
            rgba=cleaned_rgba,
            mask=cleaned_mask,
            bbox=BoundingBox(0, 0, cleaned_rgba.width, cleaned_rgba.height),
            label=object_label,
        )

    if merged is None:
        raise RuntimeError(f"Failed to extract '{object_label}' from the reference image.")

    if cfg["extraction"].get("grabcut_refine", True):
        try:
            merged = refine_mask_grabcut(reference_image, merged, int(cfg["extraction"].get("grabcut_iters", 2)))
        except Exception:
            pass

    mask_np = np.array(merged, dtype=np.uint8)
    rgba_np = np.array(reference_image.convert("RGBA")).copy()
    rgba_np[:, :, 3] = mask_np
    rgba_np[mask_np == 0, :3] = 0

    out = Image.fromarray(rgba_np, mode="RGBA")
    out = clean_extracted_object(out, cfg)

    cropped_rgba, cropped_mask, _ = crop_to_alpha_bbox(out)
    return ExtractedObject(
        rgba=cropped_rgba,
        mask=cropped_mask,
        bbox=BoundingBox(0, 0, cropped_rgba.width, cropped_rgba.height),
        label=object_label,
    )


def iou(a: BoundingBox, b: BoundingBox) -> float:
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


def estimate_depth_map(image: Image.Image, device: torch.device, cfg: dict) -> np.ndarray:
    pipe = get_depth_pipe(device, cfg)
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
    bottom = np.median(depth[-max(1, h // 5) :, :])
    if bottom < top:
        depth = 1.0 - depth
    return depth


def depth_at_box_base(depth_map: np.ndarray, box: BoundingBox) -> float:
    h, w = depth_map.shape
    x0 = max(0, min(w - 1, int(round(box.x0 + box.width() * 0.15))))
    x1 = max(x0 + 1, min(w, int(round(box.x1 - box.width() * 0.15))))
    y0 = max(0, min(h - 1, int(round(box.y1 - max(2.0, box.height() * 0.10)))))
    y1 = max(y0 + 1, min(h, int(round(box.y1))))
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.5
    return float(np.median(patch))


def local_depth_stats(depth_map: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[float, float]:
    H, W = depth_map.shape
    x0 = max(0, min(W - 1, x))
    y0 = max(0, min(H - 1, y))
    x1 = max(x0 + 1, min(W, x + w))
    y1 = max(y0 + 1, min(H, y + h))
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.5, 1.0
    return float(np.median(patch)), float(np.std(patch))


def estimate_surface_scale_factor(support_box: BoundingBox, candidate_y_bottom: float) -> float:
    rel = np.clip((candidate_y_bottom - support_box.y0) / max(1.0, support_box.height()), 0.0, 1.0)
    return float(np.interp(rel, [0.0, 1.0], [0.84, 1.18]))


def estimate_reference_scale_from_neighbors(existing_boxes: List[BoundingBox], candidate_center: Tuple[float, float]) -> Optional[float]:
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


def cosine_to_unit_interval(value: float) -> float:
    return float(np.clip((value + 1.0) * 0.5, 0.0, 1.0))


def clip_image_text_score(image: Image.Image, prompts: List[str], device: torch.device, cfg: dict) -> float:
    if not prompts:
        return 0.0
    processor, model = get_clip_model(device, cfg)
    inputs = processor(text=prompts, images=[image.convert("RGB")], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        sims = (image_embeds @ text_embeds.T).squeeze(0).detach().float().cpu().numpy()
    return float(np.max([cosine_to_unit_interval(float(v)) for v in np.atleast_1d(sims)]))


def score_surface_semantics(scene_image: Image.Image, support_box: BoundingBox, object_label: str, device: torch.device, cfg: dict) -> tuple[float, float]:
    sem_cfg = cfg.get("semantic_scoring", {})
    if not sem_cfg.get("enabled", True):
        return 0.5, 0.5

    pad_x = int(round(support_box.width() * 0.05))
    pad_y = int(round(support_box.height() * 0.08))
    x0 = max(0, int(round(support_box.x0)) - pad_x)
    y0 = max(0, int(round(support_box.y0)) - pad_y)
    x1 = min(scene_image.width, int(round(support_box.x1)) + pad_x)
    y1 = min(scene_image.height, int(round(support_box.y1)) + pad_y)
    crop = scene_image.crop((x0, y0, x1, y1)).convert("RGB")

    label = (support_box.label or "surface").strip()
    surface_prompts = [p.format(label=object_label) for p in sem_cfg.get("surface_prompt_templates", [])]
    if label:
        surface_prompts += [
            f"a {label} that can support a {object_label}",
            f"a realistic {label} for placing a {object_label}",
        ]

    support_score = clip_image_text_score(crop, surface_prompts, device, cfg)

    scene_prompts = [p.format(label=object_label) for p in sem_cfg.get("scene_prompt_templates", [])]
    if label:
        scene_prompts += [f"a {object_label} naturally placed on a {label}"]
    scene_score = clip_image_text_score(crop, scene_prompts, device, cfg)

    support_score = float(np.clip(support_score + (sem_cfg.get("support_label_bonus", 0.22) if label in {"table", "desk", "countertop", "kitchen counter", "island"} else 0.0), 0.0, 1.0))
    scene_score = float(np.clip(scene_score + (sem_cfg.get("scene_patch_bonus", 0.16) if label in {"table", "desk", "countertop", "kitchen counter", "island"} else 0.0), 0.0, 1.0))
    return support_score, scene_score


def bottom_support_coverage(candidate_box: BoundingBox, support_box: BoundingBox) -> float:
    overlap_left = max(candidate_box.x0, support_box.x0)
    overlap_right = min(candidate_box.x1, support_box.x1)
    overlap_w = max(0.0, overlap_right - overlap_left)
    return float(overlap_w / max(1.0, candidate_box.width()))


def physics_plausibility_score(candidate_box: BoundingBox, support: SupportGeometry, scene_h: int, cfg: dict) -> float:
    search_cfg = cfg.get("placement_search", {})
    support_cov = bottom_support_coverage(candidate_box, support.box)
    min_cov = float(search_cfg.get("min_support_coverage_ratio", 0.58))
    base_contact_ratio = float(search_cfg.get("base_contact_ratio", 0.72))

    base_y = candidate_box.y1
    contact_y = support.plane_y_min if support.mode == "edge" else float(np.clip(base_y, support.plane_y_min, support.plane_y_max))
    y_gap = abs(base_y - contact_y) / max(2.0, candidate_box.height() * 0.08)
    y_term = float(np.exp(-y_gap))

    coverage_term = float(np.clip((support_cov - min_cov) / max(1e-6, 1.0 - min_cov), 0.0, 1.0))
    if support_cov < min_cov:
        coverage_term *= 0.2

    if support.mode == "plane":
        base_ratio = np.clip((base_y - support.plane_y_min) / max(1.0, support.plane_y_max - support.plane_y_min), 0.0, 1.0)
        contact_pref = 1.0 - abs(base_ratio - base_contact_ratio)
        contact_pref = float(np.clip(contact_pref, 0.0, 1.0))
    else:
        contact_pref = 1.0
        if not search_cfg.get("allow_edge_contacts", False):
            contact_pref *= 0.7

    elevation_penalty = float(np.clip((candidate_box.y0 / max(1.0, scene_h) - 0.05) / 0.95, 0.0, 1.0))
    return float(np.clip(0.45 * coverage_term + 0.35 * y_term + 0.15 * contact_pref + 0.05 * elevation_penalty, 0.0, 1.0))


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

    if candidate_center is not None:
        neighbor_h = estimate_reference_scale_from_neighbors(existing_same_objects, candidate_center)
        if neighbor_h is not None:
            target_h = neighbor_h
            if depth_map is not None and candidate_depth is not None and existing_same_objects:
                ref_depth = np.median([depth_at_box_base(depth_map, b) for b in existing_same_objects[: min(3, len(existing_same_objects))]])
                ratio = float(candidate_depth / max(0.12, ref_depth))
                scale = (
                    float(np.clip(ratio**0.55, cfg["placement"]["min_scale_ratio"], cfg["placement"]["max_scale_ratio"]))
                    if cfg
                    else float(np.clip(ratio**0.55, 0.78, 1.28))
                )
                target_h *= scale
            target_h = int(round(np.clip(target_h, scene_h * 0.05, scene_h * 0.32)))
            target_w = int(round(target_h * aspect))
            return max(24, target_w), max(24, target_h)

    if existing_same_objects:
        base_box = sorted(existing_same_objects, key=lambda b: b.area(), reverse=True)[0]
        target_h = float(base_box.height())
        if depth_map is not None and candidate_depth is not None:
            ref_depth = depth_at_box_base(depth_map, base_box)
            ratio = float(candidate_depth / max(0.12, ref_depth))
            scale = (
                float(np.clip(ratio**0.55, cfg["placement"]["min_scale_ratio"], cfg["placement"]["max_scale_ratio"]))
                if cfg
                else float(np.clip(ratio**0.55, 0.78, 1.28))
            )
            target_h *= scale
        target_h = int(round(np.clip(target_h, scene_h * 0.05, scene_h * 0.30)))
        target_w = int(round(target_h * aspect))
        return max(24, target_w), max(24, target_h)

    if support_box is not None:
        support_factor = float(np.clip(support_box.width() / max(1.0, scene_w), 0.18, 0.75))
        target_w = int(round(scene_w * (0.08 + 0.12 * support_factor)))
        if candidate_depth is not None:
            depth_scale = float(np.interp(candidate_depth, [0.0, 1.0], [0.78, 1.18]))
            target_w = int(round(target_w * depth_scale))
        target_h = int(round(target_w / max(0.1, aspect)))
        return max(24, target_w), max(24, target_h)

    target_w = int(round(scene_w * 0.12))
    target_h = int(round(target_w / max(0.1, aspect)))
    return max(24, target_w), max(24, target_h)

def support_preference_adjustment(support: SupportGeometry, cfg: dict) -> float:
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

def filter_support_boxes(boxes: List[BoundingBox], image_size: Tuple[int, int]) -> List[BoundingBox]:
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
    depth_slope = float(row_medians[-1] - row_medians[0])
    depth_variance = float(np.std(row_medians))
    overall_std = float(np.std(patch))
    return depth_slope, depth_variance, overall_std


def classify_support_geometry(box: BoundingBox, image_size: Tuple[int, int], depth_map: Optional[np.ndarray], cfg: dict) -> SupportGeometry:
    scene_w, scene_h = image_size
    height_ratio = box.height() / max(1.0, scene_h)
    width_ratio = box.width() / max(1.0, scene_w)
    cy_ratio = box.centre()[1] / max(1.0, scene_h)

    if depth_map is not None:
        depth_slope, depth_variance, overall_std = support_depth_profile(depth_map, box)
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

    _ = overall_std  # reserved for future use / debugging

    return SupportGeometry(
        box=box,
        mode=mode,
        plane_y_min=y_min,
        plane_y_max=y_max,
        depth_slope=float(depth_slope),
        depth_variance=float(depth_variance),
        score=float(score),
    )


def build_support_geometries(boxes: List[BoundingBox], image_size: Tuple[int, int], depth_map: Optional[np.ndarray], cfg: dict) -> List[SupportGeometry]:
    geoms = [classify_support_geometry(box, image_size, depth_map, cfg) for box in boxes]
    geoms.sort(key=lambda g: g.score, reverse=True)
    return geoms


def candidate_positions_on_support(
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


def rank_placements(
    scene_image: Image.Image,
    extracted_object: ExtractedObject,
    support_boxes: List[BoundingBox],
    avoid_boxes: List[BoundingBox],
    existing_same_objects: List[BoundingBox],
    depth_map: Optional[np.ndarray],
    cfg: dict,
    device: torch.device,
) -> List[PlacementCandidate]:
    scene_w, scene_h = scene_image.size
    support_geometries = build_support_geometries(support_boxes, scene_image.size, depth_map, cfg)
    out: List[PlacementCandidate] = []

    collision_cfg = cfg.get("collision", {})
    search_cfg = cfg.get("placement_search", {})
    occupancy_map = None
    if collision_cfg.get("enabled", True) and collision_cfg.get("use_occupancy_map", True):
        occupancy_map = build_occupancy_map(
            scene_image.size,
            avoid_boxes,
            blur_px=int(collision_cfg.get("occupancy_blur_px", 9)),
        )

    support_semantic_cache: dict[tuple[int, int, int, int, str], tuple[float, float]] = {}
    scale_multipliers = [float(v) for v in search_cfg.get("scale_multipliers", [0.82, 0.92, 1.0, 1.1, 1.22])]

    for support in support_geometries[: int(cfg["placement"].get("max_supports_to_try", 6))]:
        support_key = (*support.box.to_int_tuple(), support.box.label or "")
        if support_key not in support_semantic_cache:
            support_semantic_cache[support_key] = score_surface_semantics(
                scene_image=scene_image,
                support_box=support.box,
                object_label=extracted_object.label,
                device=device,
                cfg=cfg,
            )
        support_semantic, scene_semantic = support_semantic_cache[support_key]

        if support_semantic < float(search_cfg.get("surface_semantic_min", 0.18)):
            continue
        if scene_semantic < float(search_cfg.get("scene_semantic_min", 0.17)):
            continue

        seed_depth = depth_at_box_base(depth_map, support.box) if depth_map is not None else 0.5
        base_w, base_h = choose_target_size(
            scene_size=scene_image.size,
            obj_size=extracted_object.rgba.size,
            support_box=support.box,
            existing_same_objects=existing_same_objects,
            depth_map=depth_map,
            candidate_depth=seed_depth,
            candidate_center=support.box.centre(),
            cfg=cfg,
        )

        for scale_mul in scale_multipliers:
            scaled_w = max(24, int(round(base_w * scale_mul)))
            scaled_h = max(24, int(round(base_h * scale_mul)))

            for x, foot_y in candidate_positions_on_support(support, scaled_w, scaled_h, scene_w, scene_h, cfg):
                sample_y = foot_y - int(scaled_h * 0.20) if support.mode == "edge" else foot_y - int(scaled_h * 0.08)

                depth_median, depth_std = (0.5, 0.25)
                if depth_map is not None:
                    depth_median, depth_std = local_depth_stats(
                        depth_map,
                        x=x + int(scaled_w * 0.1),
                        y=max(0, sample_y),
                        w=max(6, int(scaled_w * 0.8)),
                        h=max(4, int(scaled_h * 0.22)),
                    )

                center_guess = (x + scaled_w * 0.5, foot_y - scaled_h * 0.5)
                target_w, target_h = choose_target_size(
                    scene_size=scene_image.size,
                    obj_size=extracted_object.rgba.size,
                    support_box=support.box,
                    existing_same_objects=existing_same_objects,
                    depth_map=depth_map,
                    candidate_depth=depth_median,
                    candidate_center=center_guess,
                    cfg=cfg,
                )
                target_w = max(24, int(round(target_w * scale_mul)))
                target_h = max(24, int(round(target_h * scale_mul)))

                if support.mode == "plane":
                    plane_rel = np.clip(
                        (foot_y - support.plane_y_min) / max(1.0, support.plane_y_max - support.plane_y_min),
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

                overlaps = [iou(candidate_box, other) for other in avoid_boxes]
                max_overlap = max(overlaps, default=0.0)
                sum_overlap = sum(overlaps)
                if collision_cfg.get("enabled", True) and max_overlap > float(collision_cfg.get("max_iou", 0.01)):
                    continue

                inter_ratios = [candidate_intersection_ratio(candidate_box, other) for other in avoid_boxes]
                if inter_ratios and max(inter_ratios) > float(collision_cfg.get("max_intersection_ratio_of_candidate", 0.02)):
                    continue

                occupancy_penalty = 0.0
                if occupancy_map is not None:
                    occ_score = occupancy_score_for_box(occupancy_map, candidate_box)
                    if occ_score > float(search_cfg.get("occupancy_hard_threshold", 0.33)):
                        continue
                    occupancy_penalty = occ_score * float(collision_cfg.get("occupancy_penalty_weight", 10.0))

                center_offset = abs(candidate_box.centre()[0] - support.box.centre()[0]) / max(1.0, support.box.width())
                if support.mode == "plane":
                    support_band_pref = abs(
                        (foot_y - support.plane_y_min) / max(1.0, support.plane_y_max - support.plane_y_min) - 0.35
                    )
                else:
                    support_band_pref = 0.0

                predicted_w, predicted_h = choose_target_size(
                    scene_size=scene_image.size,
                    obj_size=extracted_object.rgba.size,
                    support_box=support.box,
                    existing_same_objects=existing_same_objects,
                    depth_map=depth_map,
                    candidate_depth=depth_median,
                    candidate_center=candidate_box.centre(),
                    cfg=cfg,
                )
                size_consistency = (abs(predicted_w - target_w) / max(1.0, predicted_w)) + (abs(predicted_h - target_h) / max(1.0, predicted_h))

                empty_space_score = 0.0
                ccx, ccy = candidate_box.centre()
                for other in avoid_boxes:
                    ocx, ocy = other.centre()
                    dist = ((ocx - ccx) ** 2 + (ocy - ccy) ** 2) ** 0.5
                    empty_space_score += 1.0 / max(30.0, dist)

                mode_penalty = 0.0
                if support.mode == "edge":
                    mode_penalty = max(0.0, support.depth_slope - 0.05) * 3.0
                else:
                    mode_penalty = max(0.0, 0.035 - support.depth_slope) * 4.0

                preference_adjustment = support_preference_adjustment(support, cfg)
                flatness_score = float(np.clip(1.0 - min(1.0, depth_std / 0.08), 0.0, 1.0))
                physics_score = physics_plausibility_score(candidate_box, support, scene_h, cfg)

                score = (
                    max_overlap * cfg["placement"]["avoid_overlap_weight"]
                    + sum_overlap * cfg["placement"]["total_overlap_weight"]
                    + depth_std * cfg["placement"]["depth_std_weight"]
                    + center_offset * cfg["placement"]["center_offset_weight"]
                    + support_band_pref * cfg["placement"]["support_band_weight"]
                    + (1.0 - min(1.0, persp)) * cfg["placement"]["perspective_weight"]
                    + empty_space_score * cfg["placement"]["favor_empty_space_weight"]
                    + size_consistency * cfg["placement"]["size_consistency_weight"]
                    + mode_penalty
                    + preference_adjustment
                    + occupancy_penalty
                    - support_semantic * float(search_cfg.get("surface_semantic_weight", 2.2))
                    - scene_semantic * float(search_cfg.get("scene_semantic_weight", 1.4))
                    - flatness_score * float(search_cfg.get("flatness_weight", 1.6))
                    - physics_score * float(search_cfg.get("physics_weight", 2.6))
                )

                out.append(
                    PlacementCandidate(
                        placement=Placement(
                            x=int(candidate_box.x0),
                            y=int(candidate_box.y0),
                            width=target_w,
                            height=target_h,
                            support_box=support.box,
                        ),
                        score=float(score),
                        debug=(
                            f"label={support.box.label} mode={support.mode} sem={support_semantic:.3f}/{scene_semantic:.3f} "
                            f"flat={flatness_score:.3f} phys={physics_score:.3f} overlap={max_overlap:.3f} "
                            f"depth_std={depth_std:.3f} persp={persp:.3f} scale={scale_mul:.2f}"
                        ),
                    )
                )

    out.sort(key=lambda c: c.score)
    return out[: int(cfg["placement"].get("top_k_to_keep", 12))]


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
    vx /= norm
    vy /= norm

    sx = -vx
    sy = max(0.12, -vy)
    norm2 = max(1e-6, (sx * sx + sy * sy) ** 0.5)
    return sx / norm2, sy / norm2


def surface_shadow_color(scene_patch: Image.Image) -> Tuple[int, int, int]:
    arr = np.array(scene_patch, dtype=np.uint8).reshape(-1, 3)
    med = np.median(arr, axis=0)
    tinted = np.clip(med * 0.34, 0, 255).astype(np.uint8)
    return int(tinted[0]), int(tinted[1]), int(tinted[2])


def surface_patch_stats(scene_patch: Image.Image) -> tuple[float, float]:
    gray = np.array(scene_patch.convert("L"), dtype=np.float32) / 255.0
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return brightness, contrast


def build_projected_shadow(alpha: Image.Image, placement: Placement, light_dir: Tuple[float, float], cfg: dict, scene_patch: Image.Image) -> Image.Image:
    a = np.array(alpha, dtype=np.uint8)
    brightness, contrast = surface_patch_stats(scene_patch)

    squash_ratio = float(cfg["shadow"]["squash_ratio"])
    shear_strength = float(cfg["shadow"]["shear_strength"])
    cast_length_scale = float(cfg["shadow"]["cast_length_scale"])
    softness_influence = float(cfg["shadow"].get("shadow_softness_influence", 0.35))

    adaptive_squash = float(np.clip(squash_ratio + (0.05 * (1.0 - brightness)), 0.24, 0.52))
    adaptive_shear = float(np.clip(shear_strength * (0.85 + contrast), 0.10, 0.42))
    adaptive_length = float(np.clip(cast_length_scale * (0.85 + 0.5 * brightness), 0.35, 1.15))

    new_h = max(1, int(round(a.shape[0] * adaptive_squash)))
    squashed = cv2.resize(a, (a.shape[1], new_h), interpolation=cv2.INTER_LINEAR)

    dx = int(round(light_dir[0] * placement.width * adaptive_shear))
    dy = int(round(light_dir[1] * placement.height * adaptive_length))

    canvas_h = max(new_h + abs(dy) + 6, placement.height)
    canvas_w = max(a.shape[1] + abs(dx) + 6, placement.width)

    src = np.float32([[0, 0], [squashed.shape[1] - 1, 0], [0, squashed.shape[0] - 1]])
    dst = np.float32(
        [
            [max(0, dx), 0],
            [max(0, dx) + squashed.shape[1] - 1, 0],
            [0, squashed.shape[0] - 1 + max(0, dy)],
        ]
    )
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(squashed, M, (canvas_w, canvas_h))

    blur_sigma = float(cfg["shadow"]["cast_blur_px"]) * (0.8 + softness_influence * (1.0 - contrast))
    warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=blur_sigma, sigmaY=max(0.8, blur_sigma * 0.7))

    opacity = float(cfg["shadow"]["cast_opacity"])
    opacity *= float(np.clip(0.82 + contrast + 0.18 * brightness, 0.75, 1.25))
    warped = np.clip(warped.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
    return Image.fromarray(warped, mode="L")


def harmonize_object_to_scene(obj_rgba: Image.Image, scene_patch: Image.Image, strength: float) -> Image.Image:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return obj_rgba

    rgba = np.array(obj_rgba.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    mask = alpha > 0
    if not np.any(mask):
        return obj_rgba

    obj_rgb = rgba[..., :3].astype(np.float32)
    patch_gray = np.array(scene_patch.convert("L"), dtype=np.float32) / 255.0
    target_mean = float(np.mean(patch_gray))

    obj_gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    obj_mean = float(np.mean(obj_gray[mask]))
    gain = target_mean / max(1e-4, obj_mean)
    gain = float(np.clip(gain, 0.82, 1.18))
    gain = 1.0 + (gain - 1.0) * strength

    obj_rgb[mask] *= gain
    rgba[..., :3] = np.clip(obj_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def render_initial_composite(
    scene_image: Image.Image,
    extracted_object: ExtractedObject,
    placement: Placement,
    cfg: dict,
) -> tuple[Image.Image, Image.Image]:
    canvas = scene_image.convert("RGBA")
    obj = extracted_object.rgba.resize((placement.width, placement.height), Image.LANCZOS)

    harmonize_strength = float(cfg.get("refinement", {}).get("object_harmonize_strength", 0.18))
    if harmonize_strength > 0.0:
        scene_patch = sample_surface_patch(scene_image, placement)
        obj = harmonize_object_to_scene(obj, scene_patch, harmonize_strength)

    alpha = obj.getchannel("A")

    if cfg["shadow"].get("enabled", True):
        surface_patch = sample_surface_patch(scene_image, placement)
        light_dir = estimate_light_direction(surface_patch)
        brightness, contrast = surface_patch_stats(surface_patch)

        shadow_rgb = (0, 0, 0)
        if cfg["shadow"].get("shadow_color_mode", "surface_tinted") == "surface_tinted":
            shadow_rgb = surface_shadow_color(surface_patch)

        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))

        contact = Image.new("L", obj.size, 0)
        draw = ImageDraw.Draw(contact)
        band = max(2, int(round(placement.height * float(cfg["shadow"].get("ambient_occlusion_band_ratio", 0.08)))))
        y0 = placement.height - band - max(1, int(round(placement.height * 0.03)))
        y1 = placement.height - 1
        draw.ellipse(
            [int(placement.width * 0.16), y0, int(placement.width * 0.84), y1],
            fill=int(255 * float(cfg["shadow"]["contact_opacity"]) * np.clip(0.9 + contrast, 0.85, 1.25)),
        )
        contact_blur = float(cfg["shadow"]["contact_blur_px"]) * (0.9 + 0.2 * (1.0 - brightness))
        contact = contact.filter(ImageFilter.GaussianBlur(radius=contact_blur))

        contact_rgba = Image.new("RGBA", obj.size, shadow_rgb + (0,))
        contact_rgba.putalpha(contact)

        projected = build_projected_shadow(alpha, placement, light_dir, cfg, surface_patch)
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

    full_mask = Image.new("L", scene_image.size, 0)
    full_mask.paste(alpha, (placement.x, placement.y))
    return canvas.convert("RGB"), full_mask


def build_refinement_mask(full_object_mask: Image.Image, placement: Placement, cfg: dict) -> Image.Image:
    refine_cfg = cfg.get("refinement", {})
    mask = np.array(full_object_mask, dtype=np.uint8)
    expand_px = max(1, int(refine_cfg.get("mask_expand_px", 18)))
    core_erode_px = max(1, int(refine_cfg.get("core_erode_px", 10)))
    contact_expand_px = max(1, int(refine_cfg.get("contact_expand_px", 12)))

    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_erode_px * 2 + 1, core_erode_px * 2 + 1))
    dilated = cv2.dilate(mask, kernel_expand)
    core = cv2.erode(mask, kernel_core)
    ring = cv2.subtract(dilated, core)

    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        y_bottom = int(np.max(ys))
        x0 = int(np.min(xs))
        x1 = int(np.max(xs))
        band_h = max(4, int(round((y_bottom - int(np.min(ys)) + 1) * float(refine_cfg.get("contact_band_ratio", 0.12)))))
        contact = np.zeros_like(mask, dtype=np.uint8)
        y0 = max(0, y_bottom - band_h + 1)
        contact[y0 : min(mask.shape[0], y_bottom + contact_expand_px + 1), max(0, x0 - contact_expand_px) : min(mask.shape[1], x1 + contact_expand_px + 1)] = 255
        kernel_contact = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (contact_expand_px * 2 + 1, contact_expand_px * 2 + 1))
        contact = cv2.dilate(contact, kernel_contact)
        ring = np.maximum(ring, contact)

    blur_px = float(refine_cfg.get("mask_blur_px", 5))
    ring = cv2.GaussianBlur(ring, (0, 0), sigmaX=max(0.5, blur_px), sigmaY=max(0.5, blur_px))
    ring = np.clip(ring, 0, 255).astype(np.uint8)
    return Image.fromarray(ring, mode="L")


def mask_bbox(mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    arr = np.array(mask, dtype=np.uint8)
    ys, xs = np.where(arr > 8)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_crop_box(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    placement: Placement,
    cfg: dict,
) -> Tuple[int, int, int, int]:
    refine_cfg = cfg.get("refinement", {})
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    context = float(refine_cfg.get("crop_context_ratio", 0.55))
    pad_x = int(round(max(w, placement.width) * context))
    pad_y = int(round(max(h, placement.height) * context))

    x0 -= pad_x
    y0 -= pad_y
    x1 += pad_x
    y1 += pad_y

    min_px = int(refine_cfg.get("crop_min_px", 384))
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w < min_px:
        extra = min_px - crop_w
        x0 -= extra // 2
        x1 += extra - extra // 2
    if crop_h < min_px:
        extra = min_px - crop_h
        y0 -= extra // 2
        y1 += extra - extra // 2

    img_w, img_h = image_size
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)

    max_px = int(refine_cfg.get("crop_max_px", 1280))
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w > max_px or crop_h > max_px:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half_w = min(max_px / 2.0, crop_w / 2.0)
        half_h = min(max_px / 2.0, crop_h / 2.0)
        x0 = int(round(cx - half_w))
        x1 = int(round(cx + half_w))
        y0 = int(round(cy - half_h))
        y1 = int(round(cy + half_h))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img_w, x1)
        y1 = min(img_h, y1)

    return x0, y0, x1, y1


def round_to_multiple(value: int, multiple: int = 16) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def refine_composite_with_flux_fill(
    composite: Image.Image,
    full_object_mask: Image.Image,
    placement: Placement,
    object_label: str,
    device: torch.device,
    cfg: dict,
) -> Image.Image:
    refine_cfg = cfg.get("refinement", {})
    if not refine_cfg.get("enabled", True):
        return composite

    pipe = get_fill_pipe(device, cfg)
    if pipe is None:
        if refine_cfg.get("skip_if_unavailable", True):
            print("FLUX Fill refinement unavailable; using direct composite result.")
            return composite
        raise RuntimeError("FLUX Fill refinement requested but diffusers/FluxFillPipeline is not available.")

    refine_mask = build_refinement_mask(full_object_mask, placement, cfg)
    bbox = mask_bbox(refine_mask)
    if bbox is None:
        return composite

    crop_box = expand_crop_box(bbox, composite.size, placement, cfg)
    crop_image = composite.crop(crop_box).convert("RGB")
    crop_mask = refine_mask.crop(crop_box).convert("L")

    out_w = round_to_multiple(crop_image.size[0], 16)
    out_h = round_to_multiple(crop_image.size[1], 16)

    prompt_template = refine_cfg.get("prompt_template", "Blend the placed {label} naturally into the scene.")
    prompt = str(prompt_template).format(label=object_label)

    seed = int(refine_cfg.get("seed", 0))
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed >= 0 else None

    try:
        result = pipe(
            prompt=prompt,
            image=crop_image.resize((out_w, out_h), Image.LANCZOS),
            mask_image=crop_mask.resize((out_w, out_h), Image.LANCZOS),
            height=out_h,
            width=out_w,
            guidance_scale=float(refine_cfg.get("guidance_scale", 18.0)),
            num_inference_steps=int(refine_cfg.get("steps", 28)),
            max_sequence_length=int(refine_cfg.get("max_sequence_length", 256)),
            padding_mask_crop=int(refine_cfg.get("padding_mask_crop", 32)),
            generator=generator,
        ).images[0].convert("RGB")
    except Exception as exc:
        if refine_cfg.get("skip_if_unavailable", True):
            print(f"FLUX Fill refinement failed ({exc}); using direct composite result.")
            return composite
        raise

    result = result.resize(crop_image.size, Image.LANCZOS)

    blur_px = float(refine_cfg.get("mask_blur_px", 5))
    soft_mask = crop_mask.filter(ImageFilter.GaussianBlur(radius=max(0.5, blur_px)))
    blend_strength = float(np.clip(refine_cfg.get("blend_strength", 1.0), 0.0, 1.0))
    if blend_strength < 0.999:
        soft_arr = np.array(soft_mask, dtype=np.float32) * blend_strength
        soft_mask = Image.fromarray(np.clip(soft_arr, 0, 255).astype(np.uint8), mode="L")

    blended = Image.composite(result, crop_image, soft_mask)
    final = composite.copy()
    final.paste(blended, crop_box[:2])
    return final


def composite_object(
    scene_image: Image.Image,
    extracted_object: ExtractedObject,
    placement: Placement,
    object_label: str,
    device: torch.device,
    cfg: dict,
) -> Image.Image:
    initial, full_object_mask = render_initial_composite(scene_image, extracted_object, placement, cfg)
    return refine_composite_with_flux_fill(initial, full_object_mask, placement, object_label, device, cfg)


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


def save_debug_overlay(
    scene_image: Image.Image,
    output_path: Path,
    support_boxes: List[BoundingBox],
    placement: Placement,
    candidates: Optional[List[PlacementCandidate]] = None,
    depth_map: Optional[np.ndarray] = None,
    cfg: Optional[dict] = None,
) -> None:
    debug = scene_image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug)

    if depth_map is not None and cfg is not None:
        geoms = build_support_geometries(support_boxes, scene_image.size, depth_map, cfg)
        for geom in geoms:
            col = (0, 255, 0) if geom.mode == "plane" else (0, 180, 255)
            draw.rectangle(geom.box.to_int_tuple(), outline=col, width=3)
            draw.text((geom.box.x0 + 4, geom.box.y0 + 4), f"{geom.mode} ds={geom.depth_slope:.02f}", fill=col)

            if geom.mode == "plane":
                draw.line((geom.box.x0, geom.plane_y_min, geom.box.x1, geom.plane_y_min), fill=(255, 255, 0), width=2)
                draw.line((geom.box.x0, geom.plane_y_max, geom.box.x1, geom.plane_y_max), fill=(255, 200, 0), width=2)
            else:
                draw.line((geom.box.x0, geom.plane_y_min, geom.box.x1, geom.plane_y_min), fill=(255, 255, 0), width=2)
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


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Insert a real reference object into a scene using local open models.")
    parser.add_argument("--scene", required=True, help="Path to target scene image")
    parser.add_argument("--object-image", required=True, help="Path to reference object image")
    parser.add_argument("--object-label", required=True, help="Object label to detect in reference image, e.g. lemon")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--debug-overlay", default=None, help="Optional path to save placement overlay image")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM2 even if installed")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--attempt", type=int, default=None, help="Placement attempt index override")
    args = parser.parse_args()

    global _SAM_AVAILABLE
    if args.no_sam:
        _SAM_AVAILABLE = False

    cfg = load_config(args.config)
    if args.attempt is not None:
        cfg["placement"]["attempt_index"] = int(args.attempt)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Using device: {device}")

    scene = open_rgb(args.scene)
    ref = open_rgb(args.object_image)

    depth_map = estimate_depth_map(scene, device, cfg)
    print("Estimated scene depth map.")

    extracted = extract_reference_object(ref, args.object_label, device, cfg)
    print(f"Extracted '{args.object_label}' from reference image: {extracted.rgba.size[0]}x{extracted.rgba.size[1]}")

    support_boxes = detect_objects(
        scene,
        labels=SURFACE_LABELS,
        device=device,
        cfg=cfg,
        threshold=cfg["detection"]["support_threshold"],
        text_threshold=cfg["detection"]["support_text_threshold"],
    )
    support_boxes = filter_support_boxes(support_boxes, scene.size)
    print(f"Detected {len(support_boxes)} candidate support surfaces after filtering.")

    existing_same_objects = detect_objects(
        scene,
        labels=[args.object_label],
        device=device,
        cfg=cfg,
        threshold=cfg["detection"]["object_threshold"],
        text_threshold=cfg["detection"]["object_text_threshold"],
    )
    if existing_same_objects:
        print(f"Detected {len(existing_same_objects)} existing '{args.object_label}' objects in the scene.")
    else:
        print(f"No existing '{args.object_label}' objects detected in the scene.")

    context_boxes = detect_objects(
        scene,
        labels=CONTEXT_OBJECT_LABELS,
        device=device,
        cfg=cfg,
        threshold=cfg["detection"]["context_threshold"],
        text_threshold=cfg["detection"]["context_text_threshold"],
    )

    avoid_boxes = existing_same_objects + context_boxes
    ranked_candidates = rank_placements(
        scene_image=scene,
        extracted_object=extracted,
        support_boxes=support_boxes,
        avoid_boxes=avoid_boxes,
        existing_same_objects=existing_same_objects,
        depth_map=depth_map,
        cfg=cfg,
        device=device,
    )
    placement = choose_placement_from_ranked(ranked_candidates, int(cfg["placement"].get("attempt_index", 0)))
    print(f"Placement chosen at x={placement.x}, y={placement.y}, w={placement.width}, h={placement.height}")

    result = composite_object(scene, extracted, placement, args.object_label, device, cfg)

    output_path = build_output_path(args.output, bool(cfg["output"].get("timestamp_outputs", True)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved output image to {output_path}")

    if args.debug_overlay:
        debug_path = build_output_path(args.debug_overlay, bool(cfg["output"].get("timestamp_outputs", True)))
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug_overlay(
            scene,
            debug_path,
            support_boxes,
            placement,
            candidates=ranked_candidates,
            depth_map=depth_map,
            cfg=cfg,
        )
        print(f"Saved debug overlay to {debug_path}")


if __name__ == "__main__":
    main()