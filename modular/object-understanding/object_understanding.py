"""
python object-understanding\object_understanding.py `
   --image ".\data\input\flowers.jpg" `
   --object-label "vase with flowers" `
   --output ".\data\output\object_understanding" `
   --config ".\config\object-understanding-config\object_understanding_config.yaml" `
   --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from transformers import pipeline


# -----------------------------------------------------------------------------
# Default config
# -----------------------------------------------------------------------------

DEFAULT_OBJECT_UNDERSTANDING_CFG: Dict[str, Any] = {
    "models": {
        "vlm_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    },
    "runtime": {
        "device": "cuda",
        "dtype": "auto",  # auto | float16 | bfloat16 | float32
        "max_new_tokens": 384,
        "do_sample": False,
        "temperature": 0.0,
    },
    "input": {
        "max_side": 1280,
        "allow_embedded_alpha": True,
        "crop_to_alpha_if_present": True,
    },
    "priors": {
        "dimension_confidence_floor": 0.20,
        "dimension_confidence_default": 0.55,
        "dimension_confidence_cap": 0.90,
        "min_width_m": 0.01,
        "min_depth_m": 0.005,
        "min_height_m": 0.01,
        "max_width_m": 2.5,
        "max_depth_m": 2.5,
        "max_height_m": 2.5,
        "fallback_units": "meters",
    },
    "validation": {
        "allow_llm_fallback_to_geometry_only": True,
        "reject_non_json": False,
        "clip_contact_patch_ratio_min": 0.02,
        "clip_contact_patch_ratio_max": 0.95,
    },
    "output": {
        "save_json": True,
        "save_debug": True,
    },
}


# -----------------------------------------------------------------------------
# Lazy globals
# -----------------------------------------------------------------------------

_VLM_PIPE = None


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ObjectIdentity:
    user_label: str
    canonical_label: str
    summary: str
    confidence: float


@dataclass
class MetricDimensions:
    width_m: float
    depth_m: float
    height_m: float
    confidence: float
    source: str


@dataclass
class PhysicalAttributes:
    is_multipart: bool
    is_porous: bool
    has_thin_structures: bool
    has_true_holes: bool
    has_solid_base: bool
    upper_region_sparse: bool
    lower_region_dense: bool
    stable_base: bool
    top_heavy: bool
    footprint_shape: str
    typical_orientation: str
    material: str
    transparency: str


@dataclass
class ExtractionPriors:
    mask_hole_policy: str
    mask_island_policy: str
    foreground_mode: str
    container_region_bias: str
    background_pocket_removal: str
    notes: List[str] = field(default_factory=list)


@dataclass
class PlacementPriors:
    support_mode_preference: List[str]
    preferred_support_labels: List[str]
    avoid_support_labels: List[str]
    allow_hanging_over_edge: bool
    contact_patch_ratio: float
    fragility: str
    notes: List[str] = field(default_factory=list)


@dataclass
class ObjectPriors:
    identity: ObjectIdentity
    metric_dimensions: MetricDimensions
    physical_attributes: PhysicalAttributes
    extraction_priors: ExtractionPriors
    placement_priors: PlacementPriors
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectUnderstandingResult:
    json_path: Optional[str]
    debug_dir: Optional[str]
    priors: ObjectPriors


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
    return _deep_update(DEFAULT_OBJECT_UNDERSTANDING_CFG, user_cfg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_image(path: str | Path) -> Image.Image:
    return Image.open(Path(path))


def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _np_to_pil_l(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return default


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _choose_device(device_pref: str) -> torch.device:
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_torch_dtype(dtype_name: str) -> Optional[torch.dtype]:
    name = str(dtype_name).strip().lower()
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


def _maybe_resize(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    w, h = image.size
    scale = min(float(max_side) / float(max(w, h)), 1.0)
    if scale >= 1.0:
        return image, 1.0
    new_w = max(32, int(round(w * scale / 8) * 8))
    new_h = max(32, int(round(h * scale / 8) * 8))
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def _alpha_bbox(alpha_u8: np.ndarray, thr: int = 10) -> Tuple[int, int, int, int]:
    ys, xs = np.where(alpha_u8 > thr)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop_to_embedded_alpha_if_present(img: Image.Image, cfg: dict) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    if not bool(cfg["input"].get("allow_embedded_alpha", True)):
        return img.convert("RGB")

    rgba = img.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
    if alpha.max() <= 0:
        return img.convert("RGB")

    if not bool(cfg["input"].get("crop_to_alpha_if_present", True)):
        return rgba

    x0, y0, x1, y1 = _alpha_bbox(alpha, thr=10)
    if x1 <= x0 or y1 <= y0:
        return rgba

    arr = np.array(rgba)
    cropped = arr[y0:y1, x0:x1]
    return Image.fromarray(cropped, mode="RGBA")


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _compute_image_geometry_features(image: Image.Image) -> Dict[str, Any]:
    rgb = _pil_to_np_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_dark = (gray < thr).astype(np.uint8) * 255
    fg_light = (gray > thr).astype(np.uint8) * 255

    def best_component(mask_u8: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
        if n <= 1:
            return mask_u8
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == idx, 255, 0).astype(np.uint8)

    cand_a = best_component(fg_dark)
    cand_b = best_component(fg_light)
    mask = cand_a if int((cand_a > 0).sum()) >= int((cand_b > 0).sum()) else cand_b

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        bbox = (0, 0, image.size[0], image.size[1])
        bbox_fill_ratio = 1.0
        aspect_ratio = image.size[0] / max(1.0, image.size[1])
    else:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bbox = (x0, y0, x1, y1)
        bbox_area = max(1.0, float((x1 - x0) * (y1 - y0)))
        bbox_fill_ratio = float((mask[y0:y1, x0:x1] > 0).sum()) / bbox_area
        aspect_ratio = float(x1 - x0) / max(1.0, float(y1 - y0))

    h, w = gray.shape
    top_half = mask[: h // 2, :]
    bot_half = mask[h // 2 :, :]
    top_density = float((top_half > 0).mean()) if top_half.size > 0 else 0.0
    bot_density = float((bot_half > 0).mean()) if bot_half.size > 0 else 0.0

    return {
        "image_width_px": int(image.size[0]),
        "image_height_px": int(image.size[1]),
        "edge_density": edge_density,
        "bbox_xyxy": [int(v) for v in bbox],
        "bbox_fill_ratio": bbox_fill_ratio,
        "aspect_ratio_bbox": aspect_ratio,
        "top_half_mask_density": top_density,
        "bottom_half_mask_density": bot_density,
    }


# -----------------------------------------------------------------------------
# Prompting
# -----------------------------------------------------------------------------

def _make_vlm_prompt(
    object_label: str,
    geometry: Dict[str, Any],
) -> str:
    return f"""
You are estimating object priors for image compositing.

Return STRICT JSON only. No prose. No markdown. No extra keys outside the schema.

Task:
Given the object image and optional user label, estimate realistic physical priors for extraction and placement.

Use cautious, physically plausible estimates.
Use meters.
If uncertain, choose a conservative household-scale estimate and lower confidence.

Optional user label:
"{object_label}"

Observed geometry hints from preprocessing:
- bbox_fill_ratio: {geometry["bbox_fill_ratio"]:.4f}
- bbox_aspect_ratio: {geometry["aspect_ratio_bbox"]:.4f}
- edge_density: {geometry["edge_density"]:.4f}
- top_half_mask_density: {geometry["top_half_mask_density"]:.4f}
- bottom_half_mask_density: {geometry["bottom_half_mask_density"]:.4f}

Schema:
{{
  "identity": {{
    "canonical_label": "string",
    "summary": "short physical description",
    "confidence": 0.0
  }},
  "metric_dimensions": {{
    "width_m": 0.0,
    "depth_m": 0.0,
    "height_m": 0.0,
    "confidence": 0.0
  }},
  "physical_attributes": {{
    "is_multipart": false,
    "is_porous": false,
    "has_thin_structures": false,
    "has_true_holes": false,
    "has_solid_base": false,
    "upper_region_sparse": false,
    "lower_region_dense": false,
    "stable_base": false,
    "top_heavy": false,
    "footprint_shape": "round|rectangular|irregular|multi_point|unknown",
    "typical_orientation": "upright|flat|leaning|unknown",
    "material": "string",
    "transparency": "opaque|semi_transparent|transparent|unknown"
  }},
  "extraction_priors": {{
    "mask_hole_policy": "fill_small_holes_only|preserve_true_holes|preserve_structural_holes_only",
    "mask_island_policy": "largest_only|largest_plus_attached|keep_relevant_parts",
    "foreground_mode": "generic|structure_aware|sparse_top_dense_bottom|transparent_sensitive",
    "container_region_bias": "none|bottom_dense|bottom_dense_top_sparse",
    "background_pocket_removal": "low|medium|high",
    "notes": ["string"]
  }},
  "placement_priors": {{
    "support_mode_preference": ["plane"],
    "preferred_support_labels": ["table", "desk", "countertop", "shelf"],
    "avoid_support_labels": [],
    "allow_hanging_over_edge": false,
    "contact_patch_ratio": 0.0,
    "fragility": "low|medium|high",
    "notes": ["string"]
  }}
}}

Rules:
- Dimensions must be plausible for one real-world object, not a whole room.
- Height should reflect the full visible object, including tall sparse parts.
- If the object appears to have a dense lower base and sparse upper structure, mark:
  lower_region_dense=true, upper_region_sparse=true, has_solid_base=true.
- If gaps are likely background pockets between sparse foreground elements, set:
  foreground_mode="structure_aware" or "sparse_top_dense_bottom"
  and background_pocket_removal="high".
- If the object has genuine open holes that should remain empty, set:
  has_true_holes=true and mask_hole_policy="preserve_true_holes" or "preserve_structural_holes_only".
- contact_patch_ratio is the approximate fraction of the object bottom footprint that makes meaningful support contact.
""".strip()


# -----------------------------------------------------------------------------
# Model loading / inference
# -----------------------------------------------------------------------------

def _get_vlm_pipe(device: torch.device, cfg: dict):
    global _VLM_PIPE
    if _VLM_PIPE is None:
        token = os.getenv("HF_TOKEN")
        device_index = 0 if device.type == "cuda" else -1
        torch_dtype = _resolve_torch_dtype(cfg["runtime"].get("dtype", "auto"))
        kwargs: Dict[str, Any] = {
            "task": "image-text-to-text",
            "model": cfg["models"]["vlm_model_id"],
            "device": device_index,
            "token": token,
        }
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        _VLM_PIPE = pipeline(**kwargs)
    return _VLM_PIPE


def _run_vlm_json_inference(
    image: Image.Image,
    prompt: str,
    device: torch.device,
    cfg: dict,
) -> Dict[str, Any]:
    pipe = _get_vlm_pipe(device, cfg)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    out = pipe(
        text=messages,
        max_new_tokens=int(cfg["runtime"]["max_new_tokens"]),
        do_sample=bool(cfg["runtime"]["do_sample"]),
        temperature=float(cfg["runtime"]["temperature"]),
        return_full_text=False,
    )

    raw_text = ""
    if isinstance(out, list) and out:
        item = out[0]
        if isinstance(item, dict):
            if "generated_text" in item:
                generated = item["generated_text"]
                if isinstance(generated, list):
                    if generated:
                        last = generated[-1]
                        if isinstance(last, dict):
                            raw_text = str(last.get("content", ""))
                        else:
                            raw_text = str(last)
                else:
                    raw_text = str(generated)
            elif "text" in item:
                raw_text = str(item["text"])

    parsed = _extract_json_object(raw_text)
    return {
        "prompt": prompt,
        "raw_text": raw_text,
        "parsed": parsed,
    }


# -----------------------------------------------------------------------------
# Validation / fallback
# -----------------------------------------------------------------------------

def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _geometry_fallback_priors(
    object_label: str,
    geometry: Dict[str, Any],
    cfg: dict,
) -> Dict[str, Any]:
    aspect = float(geometry["aspect_ratio_bbox"])
    top_d = float(geometry["top_half_mask_density"])
    bot_d = float(geometry["bottom_half_mask_density"])
    edge_density = float(geometry["edge_density"])

    tall = aspect < 0.75
    flat = aspect > 1.6
    sparse_top = top_d < bot_d * 0.8
    dense_bottom = bot_d > top_d * 1.1
    porous_guess = edge_density > 0.08 and geometry["bbox_fill_ratio"] < 0.72

    width_m = 0.22 if tall else (0.32 if flat else 0.26)
    depth_m = 0.18 if tall else (0.20 if flat else 0.22)
    height_m = 0.42 if tall else (0.10 if flat else 0.24)

    return {
        "identity": {
            "canonical_label": _normalize_text(object_label) or "object",
            "summary": "fallback geometry-based estimate",
            "confidence": 0.30,
        },
        "metric_dimensions": {
            "width_m": width_m,
            "depth_m": depth_m,
            "height_m": height_m,
            "confidence": 0.30,
        },
        "physical_attributes": {
            "is_multipart": bool(porous_guess),
            "is_porous": bool(porous_guess),
            "has_thin_structures": bool(porous_guess or tall),
            "has_true_holes": False,
            "has_solid_base": bool(dense_bottom),
            "upper_region_sparse": bool(sparse_top),
            "lower_region_dense": bool(dense_bottom),
            "stable_base": bool(dense_bottom or not tall),
            "top_heavy": bool(tall and sparse_top),
            "footprint_shape": "rectangular" if flat else "round",
            "typical_orientation": "flat" if flat else "upright",
            "material": "unknown",
            "transparency": "unknown",
        },
        "extraction_priors": {
            "mask_hole_policy": "preserve_structural_holes_only" if porous_guess else "fill_small_holes_only",
            "mask_island_policy": "keep_relevant_parts" if porous_guess else "largest_plus_attached",
            "foreground_mode": "sparse_top_dense_bottom" if (sparse_top and dense_bottom) else ("structure_aware" if porous_guess else "generic"),
            "container_region_bias": "bottom_dense_top_sparse" if (sparse_top and dense_bottom) else ("bottom_dense" if dense_bottom else "none"),
            "background_pocket_removal": "high" if porous_guess else "medium",
            "notes": ["Generated from geometry fallback because structured VLM output was unavailable."],
        },
        "placement_priors": {
            "support_mode_preference": ["plane"],
            "preferred_support_labels": ["table", "desk", "countertop", "shelf"],
            "avoid_support_labels": [],
            "allow_hanging_over_edge": False,
            "contact_patch_ratio": 0.18 if tall else 0.35,
            "fragility": "medium",
            "notes": ["Geometry fallback estimate only."],
        },
    }


def _validate_and_normalize_priors(
    raw: Dict[str, Any],
    object_label: str,
    geometry: Dict[str, Any],
    cfg: dict,
) -> ObjectPriors:
    priors_cfg = cfg["priors"]
    val_cfg = cfg["validation"]

    identity_raw = raw.get("identity", {}) or {}
    dims_raw = raw.get("metric_dimensions", {}) or {}
    pa_raw = raw.get("physical_attributes", {}) or {}
    ep_raw = raw.get("extraction_priors", {}) or {}
    pp_raw = raw.get("placement_priors", {}) or {}

    identity = ObjectIdentity(
        user_label=object_label,
        canonical_label=_normalize_text(str(identity_raw.get("canonical_label", "") or object_label or "object")),
        summary=_normalize_text(str(identity_raw.get("summary", "") or "object prior estimate")),
        confidence=_clip(
            _safe_float(identity_raw.get("confidence"), priors_cfg["dimension_confidence_default"]),
            0.0,
            1.0,
        ),
    )

    metric_dimensions = MetricDimensions(
        width_m=_clip(
            _safe_float(dims_raw.get("width_m"), 0.20),
            priors_cfg["min_width_m"],
            priors_cfg["max_width_m"],
        ),
        depth_m=_clip(
            _safe_float(dims_raw.get("depth_m"), 0.15),
            priors_cfg["min_depth_m"],
            priors_cfg["max_depth_m"],
        ),
        height_m=_clip(
            _safe_float(dims_raw.get("height_m"), 0.25),
            priors_cfg["min_height_m"],
            priors_cfg["max_height_m"],
        ),
        confidence=_clip(
            _safe_float(dims_raw.get("confidence"), priors_cfg["dimension_confidence_default"]),
            priors_cfg["dimension_confidence_floor"],
            priors_cfg["dimension_confidence_cap"],
        ),
        source="vlm",
    )

    physical_attributes = PhysicalAttributes(
        is_multipart=_safe_bool(pa_raw.get("is_multipart"), False),
        is_porous=_safe_bool(pa_raw.get("is_porous"), False),
        has_thin_structures=_safe_bool(pa_raw.get("has_thin_structures"), False),
        has_true_holes=_safe_bool(pa_raw.get("has_true_holes"), False),
        has_solid_base=_safe_bool(pa_raw.get("has_solid_base"), False),
        upper_region_sparse=_safe_bool(pa_raw.get("upper_region_sparse"), False),
        lower_region_dense=_safe_bool(pa_raw.get("lower_region_dense"), False),
        stable_base=_safe_bool(pa_raw.get("stable_base"), False),
        top_heavy=_safe_bool(pa_raw.get("top_heavy"), False),
        footprint_shape=str(pa_raw.get("footprint_shape", "unknown") or "unknown"),
        typical_orientation=str(pa_raw.get("typical_orientation", "unknown") or "unknown"),
        material=str(pa_raw.get("material", "unknown") or "unknown"),
        transparency=str(pa_raw.get("transparency", "unknown") or "unknown"),
    )

    extraction_priors = ExtractionPriors(
        mask_hole_policy=str(ep_raw.get("mask_hole_policy", "fill_small_holes_only")),
        mask_island_policy=str(ep_raw.get("mask_island_policy", "largest_plus_attached")),
        foreground_mode=str(ep_raw.get("foreground_mode", "generic")),
        container_region_bias=str(ep_raw.get("container_region_bias", "none")),
        background_pocket_removal=str(ep_raw.get("background_pocket_removal", "medium")),
        notes=list(ep_raw.get("notes", []) or []),
    )

    placement_priors = PlacementPriors(
        support_mode_preference=list(pp_raw.get("support_mode_preference", ["plane"])) or ["plane"],
        preferred_support_labels=list(pp_raw.get("preferred_support_labels", ["table", "desk", "countertop", "shelf"])) or ["table", "desk", "countertop", "shelf"],
        avoid_support_labels=list(pp_raw.get("avoid_support_labels", []) or []),
        allow_hanging_over_edge=_safe_bool(pp_raw.get("allow_hanging_over_edge"), False),
        contact_patch_ratio=_clip(
            _safe_float(pp_raw.get("contact_patch_ratio"), 0.20),
            val_cfg["clip_contact_patch_ratio_min"],
            val_cfg["clip_contact_patch_ratio_max"],
        ),
        fragility=str(pp_raw.get("fragility", "medium") or "medium"),
        notes=list(pp_raw.get("notes", []) or []),
    )

    return ObjectPriors(
        identity=identity,
        metric_dimensions=metric_dimensions,
        physical_attributes=physical_attributes,
        extraction_priors=extraction_priors,
        placement_priors=placement_priors,
        debug={
            "geometry_features": geometry,
        },
    )


# -----------------------------------------------------------------------------
# Debug rendering
# -----------------------------------------------------------------------------

def _draw_geometry_debug(image: Image.Image, geometry: Dict[str, Any], out_path: Path) -> None:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = geometry["bbox_xyxy"]
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
    txt = (
        f'fill={geometry["bbox_fill_ratio"]:.3f} '
        f'aspect={geometry["aspect_ratio_bbox"]:.3f} '
        f'edge={geometry["edge_density"]:.3f}'
    )
    draw.text((x0 + 4, max(0, y0 - 16)), txt, fill=(255, 255, 255))
    img.save(out_path)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def understand_object(
    image: Image.Image,
    object_label: str,
    cfg: dict,
    device: str = "cuda",
    debug_dir: Optional[Path] = None,
) -> ObjectPriors:
    load_dotenv()
    torch_device = _choose_device(device)

    if debug_dir is not None:
        _ensure_dir(debug_dir)

    img = _crop_to_embedded_alpha_if_present(image, cfg)
    work_img, scale = _maybe_resize(img, int(cfg["input"]["max_side"]))
    geometry = _compute_image_geometry_features(work_img.convert("RGB"))

    prompt = _make_vlm_prompt(
        object_label=object_label,
        geometry=geometry,
    )

    raw_vlm = None
    parsed = None
    raw_text = ""
    try:
        raw_vlm = _run_vlm_json_inference(
            image=work_img.convert("RGB"),
            prompt=prompt,
            device=torch_device,
            cfg=cfg,
        )
        raw_text = str(raw_vlm.get("raw_text", "") or "")
        parsed = raw_vlm.get("parsed")
    except Exception:
        parsed = None

    if not isinstance(parsed, dict):
        if bool(cfg["validation"].get("allow_llm_fallback_to_geometry_only", True)):
            parsed = _geometry_fallback_priors(
                object_label=object_label,
                geometry=geometry,
                cfg=cfg,
            )
        elif bool(cfg["validation"].get("reject_non_json", False)):
            raise RuntimeError("Object understanding failed: VLM did not return valid JSON.")
        else:
            parsed = _geometry_fallback_priors(
                object_label=object_label,
                geometry=geometry,
                cfg=cfg,
            )

    priors = _validate_and_normalize_priors(
        raw=parsed,
        object_label=object_label,
        geometry=geometry,
        cfg=cfg,
    )
    priors.debug.update(
        {
            "processing_scale": scale,
            "vlm_prompt": prompt,
            "vlm_raw_text": raw_text,
            "vlm_json_valid": isinstance(raw_vlm, dict) and raw_vlm.get("parsed") is not None,
            "vlm_model_id": cfg["models"]["vlm_model_id"],
        }
    )

    if debug_dir is not None and bool(cfg["output"].get("save_debug", True)):
        _draw_geometry_debug(work_img.convert("RGB"), geometry, debug_dir / "01_geometry_debug.png")
        _save_text(debug_dir / "02_vlm_prompt.txt", prompt)
        _save_text(debug_dir / "03_vlm_raw.txt", raw_text if raw_text else "")
        _save_text(debug_dir / "04_object_priors.json", json.dumps({
            "object_identity": asdict(priors.identity),
            "object_priors": {
                "metric_dimensions": asdict(priors.metric_dimensions),
                "physical_attributes": asdict(priors.physical_attributes),
                "extraction_priors": asdict(priors.extraction_priors),
                "placement_priors": asdict(priors.placement_priors),
            },
            "debug": priors.debug,
        }, indent=2))

    return priors


def understand_object_from_path(
    image_path: str | Path,
    object_label: str,
    cfg: dict,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> ObjectUnderstandingResult:
    image = _open_image(image_path)

    debug_dir = None
    if output_dir is not None and bool(cfg["output"].get("save_debug", True)):
        debug_dir = Path(output_dir) / "debug"
        _ensure_dir(debug_dir)

    priors = understand_object(
        image=image,
        object_label=object_label,
        cfg=cfg,
        device=device,
        debug_dir=debug_dir,
    )

    json_path = None
    if output_dir is not None and bool(cfg["output"].get("save_json", True)):
        _ensure_dir(Path(output_dir))
        payload = {
            "object_identity": asdict(priors.identity),
            "object_priors": {
                "metric_dimensions": asdict(priors.metric_dimensions),
                "physical_attributes": asdict(priors.physical_attributes),
                "extraction_priors": asdict(priors.extraction_priors),
                "placement_priors": asdict(priors.placement_priors),
            },
            "debug": priors.debug,
        }
        json_path = str(Path(output_dir) / "object_understanding.json")
        with Path(json_path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return ObjectUnderstandingResult(
        json_path=json_path,
        debug_dir=None if debug_dir is None else str(debug_dir),
        priors=priors,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM-first object understanding for extraction and placement priors."
    )
    parser.add_argument("--image", required=True, type=str, help="Path to object/source image.")
    parser.add_argument("--object-label", required=True, type=str, help="Optional label text for the object.")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--config", required=True, type=str, help="Path to object understanding YAML config.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], type=str)
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.config))
    result = understand_object_from_path(
        image_path=args.image,
        object_label=args.object_label,
        cfg=cfg,
        device=args.device,
        output_dir=Path(args.output),
    )

    if result.json_path is not None:
        print(result.json_path)
    else:
        print("done")


if __name__ == "__main__":
    main()