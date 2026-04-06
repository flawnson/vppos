#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from dotenv import load_dotenv
from diffusers import QwenImageEditPlusPipeline

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline  # type: ignore
except Exception:
    AutoImageProcessor = None  # type: ignore
    AutoModelForDepthEstimation = None  # type: ignore
    pipeline = None  # type: ignore

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "model_id": "Qwen/Qwen-Image-Edit-2511",
        "torch_dtype_cuda": "bfloat16",
    },
    "generation": {
        "num_inference_steps": 32,
        "true_cfg_scale": 4.0,
        "seed": 0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
    },
    "reference_preprocess": {
        "target_width": 1024,
        "target_height": 576,
        "pad_value": 255,
    },
    "extraction": {
        "use_alpha_if_present": True,
        "prefer_rembg": True,
        "grabcut_iters": 5,
        "min_foreground_fraction": 0.01,
        "max_foreground_fraction": 0.85,
        "erode_px": 1,
        "feather_px": 2,
    },
    "placement": {
        "max_object_fraction_of_scene_width": 0.24,
        "min_object_fraction_of_scene_width": 0.08,
        "num_scale_candidates": 9,
        "grid_stride": 12,
        "bottom_band_px": 10,
        "clearance_margin_px": 6,
        "support_margin_px": 4,
        "min_support_ratio": 0.78,
        "edge_density_max": 0.18,
        "prefer_lower_half": True,
        "lower_half_bonus": 0.08,
        "center_bias_weight": 0.08,
        "flatness_weight": 0.45,
        "support_weight": 0.35,
        "clearance_weight": 0.20,
        "use_depth_guidance": True,
        "depth_model_id": "Intel/dpt-hybrid-midas",
        "use_semantic_support": False,
        "semantic_support_labels": [
            "table", "desk", "counter", "countertop", "shelf", "cabinet", "dresser", "bench",
            "coffee table", "dining table", "nightstand", "tv stand", "ottoman", "floor"
        ],
        "semantic_model": "openmmlab/upernet-convnext-small",
    },
    "diffusion_edit": {
        "crop_context_scale": 2.4,
        "crop_margin_px": 36,
        "mask_expand_px": 10,
        "shadow_expand_px": 22,
        "shadow_opacity": 0.35,
        "shadow_blur_px": 14,
        "lock_reference_interior": True,
        "interior_erode_px": 8,
        "blend_generated_edges": True,
        "preserve_outside_edit_region": True,
    },
    "prompting": {
        "base_instruction": "Insert the provided object from image 2 into the black masked region in image 1.",
        "hard_constraints": [
            "Image 1 is a local crop from the original scene.",
            "Only edit the black masked region and the immediate contact-shadow area around it.",
            "Everything outside the black masked region in image 1 must remain unchanged.",
            "Image 2 is the object reference and its appearance must be preserved as closely as possible.",
            "Preserve the object's exact logo, branding, labels, text, lettering, stitching, texture, proportions, and colors.",
            "Do not rewrite or invent any text.",
            "Keep the object fully supported on a flat top-facing surface.",
            "Do not intersect nearby obstacles or float the object.",
            "Keep camera viewpoint unchanged.",
            "Generate realistic contact shadow and local occlusion only where needed for insertion.",
        ],
        "object_template": "The object to insert from image 2 is: {object_label}.",
        "placement_template": "Place it naturally and plausibly into the masked region on the support surface.",
        "user_prompt_prefix": "Additional user instructions:",
    },
    "output": {
        "timestamp_outputs": True,
        "format": "png",
        "save_debug_images": True,
    },
}


@dataclass
class PlacementCandidate:
    score: float
    x: int
    y: int
    width: int
    height: int
    support_ratio: float
    flatness_score: float
    clearance_score: float


_DEPTH_CACHE: Dict[str, Any] = {}
_SEMSEG_CACHE: Dict[str, Any] = {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_CONFIG
    config_path = Path(path)
    if not config_path.exists():
        print(f"Config not found at {config_path}. Using defaults.")
        return DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.strip().lower(), torch.bfloat16)


def build_prompt(object_label: str, user_prompt: Optional[str], cfg: Dict[str, Any]) -> str:
    prompting = cfg["prompting"]
    lines = [prompting["base_instruction"].strip(), "", "Hard constraints:"]
    for item in prompting.get("hard_constraints", []):
        lines.append(f"- {str(item).strip()}")
    lines.extend([
        "",
        prompting["object_template"].format(object_label=object_label).strip(),
        prompting["placement_template"].strip(),
    ])
    if user_prompt and user_prompt.strip():
        lines.extend(["", prompting.get("user_prompt_prefix", "Additional user instructions:").strip(), user_prompt.strip()])
    return "\n".join(lines).strip()


def build_output_path(output_root: str | Path, cfg: Dict[str, Any]) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ext = str(cfg["output"].get("format", "png")).lower().strip(".")
    timestamp_outputs = bool(cfg["output"].get("timestamp_outputs", True))
    if timestamp_outputs:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_root / f"output_{ts}.{ext}"
    return output_root / f"output.{ext}"


def build_placed_object_layer(
    scene_size: Tuple[int, int],
    placed_object: Image.Image,
    bbox_xy: Tuple[int, int],
) -> Image.Image:
    x, y_bottom = bbox_xy
    ow, oh = placed_object.size
    y_top = y_bottom - oh + 1

    layer = Image.new("RGBA", scene_size, (0, 0, 0, 0))
    layer.alpha_composite(placed_object.convert("RGBA"), (x, y_top))
    return layer


def save_debug_image(path: Path, image: Image.Image, enabled: bool) -> None:
    if enabled:
        image.save(path)


def pil_to_np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def existing_alpha_mask(image: Image.Image) -> Optional[Image.Image]:
    if image.mode in {"RGBA", "LA"}:
        alpha = image.getchannel("A")
        if alpha.getbbox() is not None:
            return alpha
    return None


def extract_with_rembg(image: Image.Image) -> Optional[Tuple[Image.Image, Image.Image]]:
    try:
        from rembg import remove  # type: ignore
    except Exception:
        return None
    out = remove(image.convert("RGBA"))
    if not isinstance(out, Image.Image):
        return None
    alpha = out.getchannel("A")
    if alpha.getbbox() is None:
        return None
    rgb = out.convert("RGBA")
    return rgb.convert("RGB"), alpha


def extract_with_grabcut(image: Image.Image, cfg: Dict[str, Any]) -> Tuple[Image.Image, Image.Image]:
    if cv2 is None:
        rgba = image.convert("RGBA")
        alpha = Image.new("L", rgba.size, 255)
        return rgba.convert("RGB"), alpha

    rgb = pil_to_np_rgb(image)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (
        max(1, int(0.03 * w)),
        max(1, int(0.03 * h)),
        max(2, int(0.94 * w)),
        max(2, int(0.94 * h)),
    )
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb[:, :, ::-1], mask, rect, bgd, fgd, int(cfg["extraction"].get("grabcut_iters", 5)), cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    alpha = Image.fromarray(fg_mask, mode="L")
    extracted = image.convert("RGBA")
    extracted.putalpha(alpha)
    return extracted.convert("RGB"), alpha


def clamp_mask(mask: Image.Image, cfg: Dict[str, Any]) -> Image.Image:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    if cv2 is not None:
        erode_px = int(cfg["extraction"].get("erode_px", 1))
        if erode_px > 0:
            kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
            arr = cv2.erode(arr, kernel, iterations=1)
    out = Image.fromarray(arr, mode="L")
    feather_px = int(cfg["extraction"].get("feather_px", 2))
    if feather_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return out


def extract_object_rgba(image: Image.Image, cfg: Dict[str, Any]) -> Image.Image:
    alpha = existing_alpha_mask(image) if cfg["extraction"].get("use_alpha_if_present", True) else None
    extracted_rgb: Optional[Image.Image] = None
    if alpha is not None:
        extracted_rgb = image.convert("RGBA").convert("RGB")
    elif cfg["extraction"].get("prefer_rembg", True):
        rembg_result = extract_with_rembg(image)
        if rembg_result is not None:
            extracted_rgb, alpha = rembg_result
    if alpha is None or extracted_rgb is None:
        extracted_rgb, alpha = extract_with_grabcut(image, cfg)

    alpha = clamp_mask(alpha, cfg)
    rgba = extracted_rgb.convert("RGBA")
    rgba.putalpha(alpha)

    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError("Automatic object extraction failed; no foreground mask was found.")

    frac = float(np.asarray(alpha).mean() / 255.0)
    min_frac = float(cfg["extraction"].get("min_foreground_fraction", 0.01))
    max_frac = float(cfg["extraction"].get("max_foreground_fraction", 0.85))
    if not (min_frac <= frac <= max_frac):
        print(f"Warning: extracted foreground fraction {frac:.3f} is outside [{min_frac}, {max_frac}].")

    return rgba.crop(bbox)


def pad_to_aspect(image: Image.Image, target_w: int, target_h: int, fill: int = 255) -> Image.Image:
    src_w, src_h = image.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    if abs(src_ratio - target_ratio) < 1e-6:
        return image.copy()
    if src_ratio > target_ratio:
        out_w = src_w
        out_h = int(round(src_w / target_ratio))
    else:
        out_h = src_h
        out_w = int(round(src_h * target_ratio))
    mode = image.mode
    if "A" in mode:
        bg = (fill, fill, fill, 0)
    elif mode == "L":
        bg = fill
    else:
        bg = (fill, fill, fill)
    canvas = Image.new(mode, (out_w, out_h), bg)
    x = (out_w - src_w) // 2
    y = (out_h - src_h) // 2
    if "A" in mode:
        canvas.alpha_composite(image, (x, y))
    else:
        canvas.paste(image, (x, y))
    return canvas


def prepare_reference_image(object_rgba: Image.Image, cfg: Dict[str, Any]) -> Image.Image:
    ref_cfg = cfg["reference_preprocess"]
    tgt_w = int(ref_cfg.get("target_width", 1024))
    tgt_h = int(ref_cfg.get("target_height", 576))
    fill = int(ref_cfg.get("pad_value", 255))
    ref = pad_to_aspect(object_rgba, tgt_w, tgt_h, fill=fill)
    ref = ref.resize((tgt_w, tgt_h), Image.LANCZOS)
    bg = Image.new("RGBA", ref.size, (fill, fill, fill, 255))
    bg.alpha_composite(ref)
    return bg.convert("RGB")


def load_depth_estimator(device: str, model_id: str):
    key = f"{model_id}:{device}"
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]
    if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
        return None
    processor = AutoImageProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        token=HF_TOKEN,
    )
    model.to(device)
    model.eval()
    _DEPTH_CACHE[key] = (processor, model)
    return processor, model


@torch.inference_mode()
def estimate_depth_map(scene_image: Image.Image, device: str, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    place_cfg = cfg["placement"]
    if not place_cfg.get("use_depth_guidance", True):
        return None

    model_id = str(place_cfg.get("depth_model_id", "Intel/dpt-hybrid-midas"))
    bundle = load_depth_estimator(device, model_id)
    if bundle is None:
        return None

    processor, model = bundle

    inputs = processor(images=scene_image, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype

    casted_inputs = {}
    for k, v in inputs.items():
        v = v.to(device)
        if torch.is_floating_point(v):
            v = v.to(dtype=model_dtype)
        casted_inputs[k] = v

    with torch.no_grad():
        outputs = model(**casted_inputs)

    pred = outputs.predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=scene_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().float().cpu().numpy()

    pred = pred - pred.min()
    if pred.max() > 1e-8:
        pred = pred / pred.max()

    return pred


def estimate_support_mask_semantic(scene_image: Image.Image, device: str, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    place_cfg = cfg["placement"]
    if not place_cfg.get("use_semantic_support", False) or pipeline is None:
        return None
    key = f"{place_cfg.get('semantic_model', '')}:{device}"
    if key not in _SEMSEG_CACHE:
        _SEMSEG_CACHE[key] = pipeline(
            task="image-segmentation",
            model=place_cfg.get("semantic_model", "openmmlab/upernet-convnext-small"),
            device=0 if device == "cuda" else -1,
            token=HF_TOKEN,
        )
    segmenter = _SEMSEG_CACHE[key]
    result = segmenter(scene_image)
    if not result:
        return None

    wanted = {label.lower() for label in place_cfg.get("semantic_support_labels", [])}
    out = np.zeros((scene_image.height, scene_image.width), dtype=np.float32)
    for item in result:
        label = str(item.get("label", "")).lower()
        if label not in wanted:
            continue
        mask_img = item.get("mask")
        if isinstance(mask_img, Image.Image):
            out = np.maximum(out, np.asarray(mask_img.resize(scene_image.size, Image.NEAREST).convert("L"), dtype=np.float32) / 255.0)
    return out if out.max() > 0 else None


def compute_edge_density_map(scene_image: Image.Image) -> np.ndarray:
    gray = np.asarray(scene_image.convert("L"), dtype=np.uint8)
    if cv2 is not None:
        edges = cv2.Canny(gray, 80, 180)
        edges = edges.astype(np.float32) / 255.0
        density = cv2.blur(edges, (21, 21))
        return density
    edges = Image.fromarray(gray).filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.float32) / 255.0
    return arr


def compute_flatness_map(scene_image: Image.Image, depth_map: Optional[np.ndarray]) -> np.ndarray:
    if depth_map is not None:
        gy, gx = np.gradient(depth_map.astype(np.float32))
        grad = np.sqrt(gx * gx + gy * gy)
        grad = grad / max(float(grad.max()), 1e-6)
        flat = 1.0 - grad
        return np.clip(flat, 0.0, 1.0)
    gray = np.asarray(scene_image.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / max(float(grad.max()), 1e-6)
    flat = 1.0 - grad
    return np.clip(flat, 0.0, 1.0)


def resize_object_to_width(object_rgba: Image.Image, target_width: int) -> Image.Image:
    w, h = object_rgba.size
    scale = target_width / max(w, 1)
    new_h = max(1, int(round(h * scale)))
    return object_rgba.resize((target_width, new_h), Image.LANCZOS)


def choose_placement(scene_image: Image.Image, object_rgba: Image.Image, device: str, cfg: Dict[str, Any]) -> PlacementCandidate:
    place_cfg = cfg["placement"]
    scene_w, scene_h = scene_image.size
    depth_map = estimate_depth_map(scene_image, device, cfg)
    flatness = compute_flatness_map(scene_image, depth_map)
    edge_density = compute_edge_density_map(scene_image)
    semantic_support = estimate_support_mask_semantic(scene_image, device, cfg)

    min_frac = float(place_cfg.get("min_object_fraction_of_scene_width", 0.08))
    max_frac = float(place_cfg.get("max_object_fraction_of_scene_width", 0.24))
    num_scales = int(place_cfg.get("num_scale_candidates", 9))
    stride = int(place_cfg.get("grid_stride", 12))
    bottom_band_px = int(place_cfg.get("bottom_band_px", 10))
    clearance_margin = int(place_cfg.get("clearance_margin_px", 6))
    support_margin = int(place_cfg.get("support_margin_px", 4))
    min_support_ratio = float(place_cfg.get("min_support_ratio", 0.78))
    edge_density_max = float(place_cfg.get("edge_density_max", 0.18))

    candidates: List[PlacementCandidate] = []
    scale_fracs = np.linspace(min_frac, max_frac, num_scales)

    for frac in scale_fracs:
        target_w = max(16, int(round(scene_w * float(frac))))
        obj_scaled = resize_object_to_width(object_rgba, target_w)
        ow, oh = obj_scaled.size
        obj_alpha = np.asarray(obj_scaled.getchannel("A"), dtype=np.float32) / 255.0
        footprint = (obj_alpha > 0.1).astype(np.float32)
        if footprint.sum() < 8:
            continue
        bottom_idx = np.where(footprint.sum(axis=1) > 0)[0]
        if len(bottom_idx) == 0:
            continue
        last_row = int(bottom_idx[-1])
        support_band = footprint[max(0, last_row - bottom_band_px + 1):last_row + 1, :]
        support_cols = np.where(support_band.max(axis=0) > 0)[0]
        if len(support_cols) == 0:
            continue
        left_active = int(support_cols[0])
        right_active = int(support_cols[-1])

        for x in range(0, max(1, scene_w - ow + 1), stride):
            support_x0 = x + left_active
            support_x1 = x + right_active
            if support_x1 >= scene_w:
                continue
            y_min = max(0, oh // 3)
            y_start = max(y_min, scene_h // 3)
            for y in range(y_start, scene_h - 1, stride):
                top = y - oh + 1
                bottom = y + 1
                if top < 0 or bottom > scene_h:
                    continue
                left = x
                right = x + ow
                patch_flat = flatness[top:bottom, left:right]
                patch_edge = edge_density[top:bottom, left:right]
                if patch_flat.size == 0:
                    continue
                support_y0 = min(scene_h - 1, y + support_margin)
                support_y1 = min(scene_h, y + support_margin + bottom_band_px)
                support_map = flatness[support_y0:support_y1, max(0, support_x0):min(scene_w, support_x1 + 1)]
                if support_map.size == 0:
                    continue
                support_ratio = float((support_map > 0.75).mean())
                if semantic_support is not None:
                    semantic_map = semantic_support[support_y0:support_y1, max(0, support_x0):min(scene_w, support_x1 + 1)]
                    if semantic_map.size > 0:
                        support_ratio = 0.6 * support_ratio + 0.4 * float(semantic_map.mean())
                if support_ratio < min_support_ratio:
                    continue

                clearance_top = max(0, top - clearance_margin)
                clearance_bottom = min(scene_h, bottom)
                clearance_left = max(0, left - clearance_margin)
                clearance_right = min(scene_w, right + clearance_margin)
                clearance_map = patch_edge
                clearance_score = 1.0 - float(clearance_map.mean())
                if 1.0 - clearance_score > edge_density_max:
                    continue

                flatness_score = float((patch_flat * footprint).sum() / max(footprint.sum(), 1.0))
                center_x = left + ow / 2.0
                center_bias = 1.0 - abs(center_x - scene_w / 2.0) / max(scene_w / 2.0, 1.0)
                lower_bonus = float(place_cfg.get("lower_half_bonus", 0.08)) if (place_cfg.get("prefer_lower_half", True) and y > scene_h * 0.55) else 0.0
                score = (
                    float(place_cfg.get("flatness_weight", 0.45)) * flatness_score
                    + float(place_cfg.get("support_weight", 0.35)) * support_ratio
                    + float(place_cfg.get("clearance_weight", 0.20)) * clearance_score
                    + float(place_cfg.get("center_bias_weight", 0.08)) * center_bias
                    + lower_bonus
                )
                candidates.append(PlacementCandidate(score, x, y, ow, oh, support_ratio, flatness_score, clearance_score))

    if not candidates:
        raise RuntimeError("No plausible flat-top placement candidates were found.")
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0]


def make_shadow_mask(size: Tuple[int, int], bbox: Tuple[int, int, int, int], cfg: Dict[str, Any]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    shadow_cfg = cfg["diffusion_edit"]
    expand = int(shadow_cfg.get("shadow_expand_px", 22))
    opacity = float(shadow_cfg.get("shadow_opacity", 0.35))
    blur_px = int(shadow_cfg.get("shadow_blur_px", 14))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    height = max(4, (y1 - y0) // 7)
    ellipse = [x0 - expand, y1 - height // 2, x1 + expand, y1 + height + expand // 2]
    draw.ellipse(ellipse, fill=int(round(255 * opacity)))
    if blur_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_px))
    return mask


def make_edit_masks(scene_size: Tuple[int, int], placed_object: Image.Image, bbox_xy: Tuple[int, int], cfg: Dict[str, Any]) -> Tuple[Image.Image, Image.Image, Image.Image]:
    x, y_bottom = bbox_xy
    ow, oh = placed_object.size
    y_top = y_bottom - oh + 1
    alpha = placed_object.getchannel("A")

    obj_canvas = Image.new("L", scene_size, 0)
    obj_canvas.paste(alpha, (x, y_top))

    expand_px = int(cfg["diffusion_edit"].get("mask_expand_px", 10))
    if cv2 is not None and expand_px > 0:
        arr = np.asarray(obj_canvas, dtype=np.uint8)
        kernel = np.ones((expand_px * 2 + 1, expand_px * 2 + 1), np.uint8)
        arr = cv2.dilate(arr, kernel, iterations=1)
        edit_mask = Image.fromarray(arr, mode="L")
    else:
        edit_mask = obj_canvas.filter(ImageFilter.MaxFilter(size=max(3, expand_px * 2 + 1)))

    bbox = (x, y_top, x + ow, y_top + oh)
    shadow_mask = make_shadow_mask(scene_size, bbox, cfg)
    full_mask = ImageChops.lighter(edit_mask, shadow_mask)
    return obj_canvas, shadow_mask, full_mask


def composite_rough_insert(scene_image: Image.Image, placed_object: Image.Image, bbox_xy: Tuple[int, int], shadow_mask: Image.Image) -> Image.Image:
    x, y_bottom = bbox_xy
    ow, oh = placed_object.size
    y_top = y_bottom - oh + 1
    base = scene_image.convert("RGBA")

    shadow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_rgba = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_rgba.putalpha(shadow_mask)
    shadow_layer.alpha_composite(shadow_rgba)
    base = Image.alpha_composite(base, shadow_layer)
    base.alpha_composite(placed_object, (x, y_top))
    return base.convert("RGB")


def crop_box_around_mask(mask: Image.Image, scene_size: Tuple[int, int], cfg: Dict[str, Any]) -> Tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        return 0, 0, scene_size[0], scene_size[1]
    x0, y0, x1, y1 = bbox
    crop_cfg = cfg["diffusion_edit"]
    margin = int(crop_cfg.get("crop_margin_px", 36))
    context_scale = float(crop_cfg.get("crop_context_scale", 2.4))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw = x1 - x0
    bh = y1 - y0
    cw = int(round(max(bw + 2 * margin, bw * context_scale)))
    ch = int(round(max(bh + 2 * margin, bh * context_scale)))
    sx, sy = scene_size
    left = max(0, int(round(cx - cw / 2.0)))
    top = max(0, int(round(cy - ch / 2.0)))
    right = min(sx, left + cw)
    bottom = min(sy, top + ch)
    left = max(0, right - cw)
    top = max(0, bottom - ch)
    return left, top, right, bottom


def black_out_mask_region(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB").copy()
    arr = np.asarray(rgb).copy()
    m = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    arr[m] = 0
    return Image.fromarray(arr, mode="RGB")


def load_pipeline(device: str, cfg: Dict[str, Any]) -> QwenImageEditPlusPipeline:
    model_cfg = cfg["model"]
    dtype = resolve_dtype(model_cfg.get("torch_dtype_cuda", "bfloat16"), device)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=dtype if device == "cuda" else torch.float32,
        token=HF_TOKEN,
    )
    pipe.to("cuda" if device == "cuda" else "cpu")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=None)
    return pipe


@torch.inference_mode()
def run_local_edit(
    pipe: QwenImageEditPlusPipeline,
    masked_crop: Image.Image,
    reference_image: Image.Image,
    prompt: str,
    cfg: Dict[str, Any],
    device: str,
) -> Image.Image:
    gen_cfg = cfg["generation"]
    seed = int(gen_cfg.get("seed", 0))
    generator = torch.Generator(device=device).manual_seed(seed) if device == "cuda" else torch.Generator().manual_seed(seed)
    result = pipe(
        image=[masked_crop, reference_image],
        prompt=prompt,
        generator=generator,
        true_cfg_scale=float(gen_cfg.get("true_cfg_scale", 4.0)),
        negative_prompt=str(gen_cfg.get("negative_prompt", " ")),
        num_inference_steps=int(gen_cfg.get("num_inference_steps", 32)),
        num_images_per_prompt=int(gen_cfg.get("num_images_per_prompt", 1)),
    )
    if not hasattr(result, "images") or not result.images:
        raise RuntimeError("Model returned no images.")
    return result.images[0].convert("RGB")


def lock_reference_interior(
    generated_crop: Image.Image,
    original_crop: Image.Image,
    placed_object_crop: Image.Image,
    object_mask_crop: Image.Image,
    cfg: Dict[str, Any],
) -> Image.Image:
    if not cfg["diffusion_edit"].get("lock_reference_interior", True):
        return generated_crop

    erode_px = int(cfg["diffusion_edit"].get("interior_erode_px", 8))
    interior = object_mask_crop.convert("L")

    if cv2 is not None and erode_px > 0:
        arr = np.asarray(interior, dtype=np.uint8)
        kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        arr = cv2.erode(arr, kernel, iterations=1)
        interior = Image.fromarray(arr, mode="L")
    elif erode_px > 0:
        interior = interior.filter(ImageFilter.MinFilter(size=max(3, erode_px * 2 + 1)))

    preserve_base = original_crop.convert("RGBA")

    placed_rgba = placed_object_crop.convert("RGBA")
    if placed_rgba.size != preserve_base.size:
        raise ValueError(
            f"placed_object_crop size {placed_rgba.size} does not match original_crop size {preserve_base.size}"
        )

    preserve_base = Image.alpha_composite(preserve_base, placed_rgba)
    locked = generated_crop.convert("RGBA")
    locked = Image.composite(preserve_base, locked, interior)
    return locked.convert("RGB")

def paste_back_unchanged_outside(scene_image: Image.Image, generated_crop: Image.Image, crop_box: Tuple[int, int, int, int], replace_mask_crop: Image.Image) -> Image.Image:
    left, top, right, bottom = crop_box
    scene_rgba = scene_image.convert("RGBA")
    crop_rgba = generated_crop.convert("RGBA")
    overlay = Image.new("RGBA", scene_rgba.size, (0, 0, 0, 0))
    overlay.paste(crop_rgba, (left, top))

    full_mask = Image.new("L", scene_rgba.size, 0)
    full_mask.paste(replace_mask_crop, (left, top))
    return Image.composite(overlay, scene_rgba, full_mask).convert("RGB")


def render_debug_overlay(scene_image: Image.Image, candidate: PlacementCandidate) -> Image.Image:
    out = scene_image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x0 = candidate.x
    y0 = candidate.y - candidate.height + 1
    x1 = candidate.x + candidate.width
    y1 = candidate.y + 1
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
    draw.text((x0 + 4, max(0, y0 - 18)), f"score={candidate.score:.3f}", fill=(255, 0, 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Local masked diffusion object insertion using Qwen-Image-Edit-2511.")
    parser.add_argument("--scene", required=True, help="Path to target scene image")
    parser.add_argument("--object-image", required=True, help="Path to reference object image")
    parser.add_argument("--object-label", required=True, help="Object label, e.g. lemon")
    parser.add_argument("--output-root", required=True, help="Output directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--prompt", default="", help="Extra user instructions appended to the fixed prompt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    print(f"Using device: {device}")

    scene = Image.open(args.scene).convert("RGB")
    object_input = Image.open(args.object_image)
    object_rgba = extract_object_rgba(object_input, cfg)
    reference_image = prepare_reference_image(object_rgba, cfg)

    candidate = choose_placement(scene, object_rgba, device, cfg)
    placed_object = resize_object_to_width(object_rgba, candidate.width)

    obj_mask, shadow_mask, full_edit_mask = make_edit_masks(scene.size, placed_object, (candidate.x, candidate.y), cfg)
    rough_scene = composite_rough_insert(scene, placed_object, (candidate.x, candidate.y), shadow_mask)
    placed_object_layer = build_placed_object_layer(scene.size, placed_object, (candidate.x, candidate.y))

    crop_box = crop_box_around_mask(full_edit_mask, scene.size, cfg)
    masked_scene = black_out_mask_region(scene, full_edit_mask)
    masked_crop = masked_scene.crop(crop_box)
    original_crop = scene.crop(crop_box)
    placed_object_crop = placed_object_layer.crop(crop_box)
    object_mask_crop = obj_mask.crop(crop_box)
    replace_mask_crop = full_edit_mask.crop(crop_box)

    prompt = build_prompt(args.object_label, args.prompt, cfg)
    print("\n===== FINAL PROMPT =====\n")
    print(prompt)
    print("\n========================\n")

    pipe = load_pipeline(device, cfg)
    generated_crop = run_local_edit(
        pipe=pipe,
        masked_crop=masked_crop,
        reference_image=reference_image,
        prompt=prompt,
        cfg=cfg,
        device=device,
    )

    generated_crop = lock_reference_interior(
        generated_crop=generated_crop,
        original_crop=original_crop,
        placed_object_crop=placed_object_crop,
        object_mask_crop=object_mask_crop,
        cfg=cfg,
    )

    if cfg["diffusion_edit"].get("preserve_outside_edit_region", True):
        final = paste_back_unchanged_outside(scene, generated_crop, crop_box, replace_mask_crop)
    else:
        final = scene.copy()
        final.paste(generated_crop, crop_box[:2])

    output_path = build_output_path(output_root, cfg)
    final.save(output_path)
    print(f"Saved output image to: {output_path}")

    if cfg["output"].get("save_debug_images", True):
        stem = output_path.with_suffix("")
        save_debug_image(stem.parent / f"{stem.name}_debug_reference.png", reference_image, True)
        save_debug_image(stem.parent / f"{stem.name}_debug_extracted_object.png", object_rgba, True)
        save_debug_image(stem.parent / f"{stem.name}_debug_masked_scene.png", masked_scene, True)
        save_debug_image(stem.parent / f"{stem.name}_debug_rough_insert.png", rough_scene, True)
        save_debug_image(stem.parent / f"{stem.name}_debug_candidate.png", render_debug_overlay(scene, candidate), True)
        save_debug_image(stem.parent / f"{stem.name}_debug_masked_crop.png", masked_crop, True)
        save_debug_image(stem.parent / f"{stem.name}_debug_generated_crop.png", generated_crop, True)


if __name__ == "__main__":
    main()
