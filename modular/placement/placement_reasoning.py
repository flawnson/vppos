from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor


# -----------------------------------------------------------------------------
# Default config
# -----------------------------------------------------------------------------

DEFAULT_PLACEMENT_CFG: Dict[str, Any] = {
    "input": {
        "object_pad_px": 2,
    },
    "supports": {
        "max_supports_to_consider": 12,
        "preselect_top_k_before_llm": 12,
        "disallow_low_usable_area_below": 0.12,
        "disallow_occupied_area_above": 0.72,
        "plane_bonus": 0.35,
        "edge_bonus": 0.10,
        "homography_bonus": 0.25,
        "semantic_affordance_weight": 1.7,
        "semantic_hard_disallow_threshold": 0.18,
        "llm_surface_selection_enabled": True,
        "llm_surface_top_k": 2,
        "llm_surface_min_score": 0.20,
        "llm_surface_priority_only": True,
        "validity_require_plane_or_edge_geometry": False,
    },
    "proposal": {
        "top_k_to_keep": 16,
        "attempt_index": 0,
        "u_samples_plane": 7,
        "v_samples_plane": 5,
        "x_step_divisor_edge": 4,
        "per_support_candidate_cap": 6,
        "final_llm_top_k": 4,
        "min_object_height_ratio": 0.04,
        "max_object_height_ratio": 0.26,
        "edge_margin_ratio": 0.05,
        "plane_u_margin": 0.08,
        "plane_v_min": 0.14,
        "plane_v_max": 0.76,
    },
    "scaling": {
        "support_width_ratio_min": 0.10,
        "support_width_ratio_max": 0.34,
        "depth_perspective_min": 0.78,
        "depth_perspective_max": 1.18,
        "plane_v_perspective_min": 0.84,
        "plane_v_perspective_max": 1.22,
        "neighbor_consistency_weight": 0.0,
        "use_scene_priors": True,
        "min_prior_confidence": 0.35,
        "prior_support_width_fraction_min": 0.10,
        "prior_support_width_fraction_max": 0.40,
        "prior_support_depth_fraction_min": 0.10,
        "prior_support_depth_fraction_max": 0.36,
        "prior_scene_height_fraction_min": 0.04,
        "prior_scene_height_fraction_max": 0.26,
        "plane_depth_bias_toward_front": 1.08,
        "edge_depth_assumption_fraction": 0.18,
        "blend_weight_min": 0.10,
        "blend_weight_max": 0.90,
        "blend_prior_confidence_weight": 0.55,
        "blend_geometry_quality_weight": 0.30,
        "blend_homography_quality_weight": 0.15,
        "blend_prior_confidence_center": 0.30,
        "blend_prior_confidence_span": 0.50,
    },
    "semantic": {
        "enabled": True,
        "device": "cuda",
        "dtype": "auto",
        "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "max_new_tokens": 220,
        "do_sample": False,
        "temperature": 0.0,
        "use_support_llm": True,
        "use_candidate_llm": True,
        "candidate_preview_max_side": 768,
        "support_crop_context": 0.24,
        "prompt_style": "strict_json",
    },
    "obstacle_avoidance": {
        "occupied_weight": 8.0,
        "occluder_weight": 6.5,
        "support_outside_weight": 9.0,
        "edge_contact_outside_weight": 5.0,
        "scene_structure_weight": 2.0,
        "border_penalty_weight": 3.5,
        "strong_obstacle_threshold": 0.20,
        "hard_reject_if_occluder_above": 0.35,
        "hard_reject_if_outside_support": 0.25,
        "plane_contact_point_max_distance_px": 18.0,
        "plane_body_below_contact_ratio_max": 0.08,
    },
    "ranking": {
        "support_priority_weight": 0.35,
        "center_alignment_weight": 0.40,
        "plane_depth_preference_weight": 0.60,
        "object_crop_tightness_weight": 0.15,
        "free_space_weight": 1.30,
        "size_prior_weight": 0.60,
        "support_mode_mismatch_penalty": 0.0,
        "semantic_penalty_weight": 1.6,
        "semantic_bonus_weight": 1.2,
        "llm_candidate_weight": 2.2,
        "plane_contact_alignment_weight": 1.0,
        "plane_body_below_contact_weight": 2.8,
    },
    "refinement": {
        "enabled": True,
        "color_match_strength": 0.70,
        "detail_preserve_strength": 0.82,
        "boundary_blend_width_px": 9,
        "use_seamless_clone": True,
        "shadow_strength": 0.22,
        "shadow_blur_px": 11,
        "shadow_offset_y_px": 4,
    },
    "output": {
        "save_debug": True,
        "save_json": True,
    },
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class PlacementCandidate:
    support_id: str
    support_label: str
    support_mode: str
    x: int
    y: int
    width: int
    height: int
    score: float
    support_priority: float
    contact_x: Optional[float] = None
    contact_y: Optional[float] = None
    u_plane: Optional[float] = None
    v_plane: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacementResult:
    composite_path: Optional[str]
    placement_json_path: Optional[str]
    debug_dir: Optional[str]
    selected_candidate: PlacementCandidate
    ranked_candidates: List[PlacementCandidate]


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
    return _deep_update(DEFAULT_PLACEMENT_CFG, user_cfg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_rgb(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def _open_rgba(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGBA")


def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _pil_to_np_rgba(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA"))


def _np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _alpha_bbox(alpha_u8: np.ndarray, thr: int = 10) -> Tuple[int, int, int, int]:
    ys, xs = np.where(alpha_u8 > thr)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop_rgba_to_alpha(rgba: Image.Image, pad: int = 0) -> Image.Image:
    arr = _pil_to_np_rgba(rgba)
    alpha = arr[:, :, 3]
    x0, y0, x1, y1 = _alpha_bbox(alpha, thr=10)
    h, w = alpha.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return Image.fromarray(arr[y0:y1, x0:x1], mode="RGBA")


def _resize_rgba(arr_rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    interp = cv2.INTER_AREA if (width < arr_rgba.shape[1] or height < arr_rgba.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(arr_rgba, (width, height), interpolation=interp)


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


def _extract_patch(arr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1].copy()


def _expand_box(box: Tuple[int, int, int, int], image_shape: Tuple[int, int], pad: int) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _placed_object_box(x: int, y: int, w: int, h: int, scene_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    H, W = scene_shape[:2]
    return max(0, x), max(0, y), min(W, x + w), min(H, y + h)


def _match_object_to_local_context(obj_rgba: np.ndarray, scene_rgb: np.ndarray, x: int, y: int, cfg: dict) -> np.ndarray:
    ref_cfg = cfg.get("refinement", {}) or {}
    strength = float(ref_cfg.get("color_match_strength", 0.70))
    if strength <= 0.0:
        return obj_rgba

    box = _placed_object_box(x, y, obj_rgba.shape[1], obj_rgba.shape[0], scene_rgb.shape)
    pad = max(8, int(round(0.18 * max(box[2] - box[0], box[3] - box[1]))))
    ctx_box = _expand_box(box, scene_rgb.shape, pad)
    ctx = _extract_patch(scene_rgb, ctx_box)
    if ctx.size == 0:
        return obj_rgba

    obj_rgb = obj_rgba[:, :, :3].astype(np.uint8)
    alpha = obj_rgba[:, :, 3]
    if int((alpha > 0).sum()) < 12:
        return obj_rgba

    scene_lab = cv2.cvtColor(ctx, cv2.COLOR_RGB2LAB).astype(np.float32)
    obj_lab = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mask = alpha > 8
    obj_vals = obj_lab[mask]
    scene_vals = scene_lab.reshape(-1, 3)
    if len(obj_vals) < 12 or len(scene_vals) < 12:
        return obj_rgba

    obj_mean = obj_vals.mean(axis=0)
    obj_std = obj_vals.std(axis=0) + 1e-6
    scene_mean = scene_vals.mean(axis=0)
    scene_std = scene_vals.std(axis=0) + 1e-6
    matched = obj_lab.copy()
    matched[mask] = (obj_lab[mask] - obj_mean) * (scene_std / obj_std) + scene_mean
    matched = obj_lab * (1.0 - strength) + matched * strength
    matched = np.clip(matched, 0.0, 255.0).astype(np.uint8)
    out_rgb = cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)
    return np.dstack([out_rgb, alpha]).astype(np.uint8)


def _contact_shadow(scene_rgb: np.ndarray, alpha_u8: np.ndarray, x: int, y: int, cfg: dict) -> np.ndarray:
    ref_cfg = cfg.get("refinement", {}) or {}
    strength = float(ref_cfg.get("shadow_strength", 0.22))
    if strength <= 0.0:
        return scene_rgb
    H, W = scene_rgb.shape[:2]
    h, w = alpha_u8.shape[:2]
    foot = _object_foot_mask(alpha_u8)
    if foot.max() == 0:
        return scene_rgb
    blur_px = max(1, int(ref_cfg.get("shadow_blur_px", 11)))
    offset_y = int(ref_cfg.get("shadow_offset_y_px", 4))
    shadow = np.zeros((H, W), dtype=np.uint8)
    x1 = max(0, x); y1 = max(0, y + offset_y); x2 = min(W, x + w); y2 = min(H, y + offset_y + h)
    ox1 = x1 - x; oy1 = y1 - (y + offset_y); ox2 = ox1 + (x2 - x1); oy2 = oy1 + (y2 - y1)
    if x1 >= x2 or y1 >= y2:
        return scene_rgb
    shadow[y1:y2, x1:x2] = foot[oy1:oy2, ox1:ox2]
    shadow = cv2.GaussianBlur(shadow, (0, 0), blur_px).astype(np.float32) / 255.0
    shadow = shadow[..., None] * strength
    out = scene_rgb.astype(np.float32) * (1.0 - shadow)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _boundary_weight(alpha_u8: np.ndarray, width_px: int) -> np.ndarray:
    width_px = max(1, int(width_px))
    mask = (alpha_u8 > 0).astype(np.uint8) * 255
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width_px + 1, 2 * width_px + 1)))
    edge = cv2.subtract(mask, eroded)
    weight = cv2.GaussianBlur(edge.astype(np.float32) / 255.0, (0, 0), max(1, width_px / 2.0))
    return np.clip(weight, 0.0, 1.0)


def _preserve_object_details(base_patch: np.ndarray, refined_patch: np.ndarray, obj_rgba: np.ndarray, cfg: dict) -> np.ndarray:
    ref_cfg = cfg.get("refinement", {}) or {}
    strength = float(ref_cfg.get("detail_preserve_strength", 0.82))
    alpha = obj_rgba[:, :, 3].astype(np.float32) / 255.0
    if alpha.max() <= 0:
        return refined_patch
    interior = cv2.erode((alpha > 0.1).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    interior_f = (interior.astype(np.float32) / 255.0)[..., None] * strength
    out = refined_patch.astype(np.float32) * (1.0 - interior_f) + base_patch.astype(np.float32) * interior_f
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _refine_composite(scene_rgb: np.ndarray, obj_rgba: np.ndarray, x: int, y: int, cfg: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    ref_cfg = cfg.get("refinement", {}) or {}
    matched = _match_object_to_local_context(obj_rgba, scene_rgb, x, y, cfg)
    base = _place_rgba_over_rgb(scene_rgb, matched, x, y)
    base = _contact_shadow(base, matched[:, :, 3], x, y, cfg)
    debug = {"used_refinement": bool(ref_cfg.get("enabled", True)), "used_seamless_clone": False}
    if not bool(ref_cfg.get("enabled", True)):
        return base, debug

    if bool(ref_cfg.get("use_seamless_clone", True)):
        box = _placed_object_box(x, y, matched.shape[1], matched.shape[0], scene_rgb.shape)
        if (box[2] - box[0]) > 4 and (box[3] - box[1]) > 4:
            try:
                mask = (matched[:, :, 3] > 8).astype(np.uint8) * 255
                center = (int(round((box[0] + box[2]) / 2.0)), int(round((box[1] + box[3]) / 2.0)))
                clone = cv2.seamlessClone(matched[:, :, :3].astype(np.uint8), scene_rgb.astype(np.uint8), mask, center, cv2.NORMAL_CLONE)
                clone = _contact_shadow(clone, matched[:, :, 3], x, y, cfg)
                boundary = _boundary_weight(matched[:, :, 3], int(ref_cfg.get("boundary_blend_width_px", 9)))[..., None]
                refined = base.astype(np.float32) * (1.0 - boundary) + clone.astype(np.float32) * boundary
                refined = np.clip(refined, 0.0, 255.0).astype(np.uint8)
                box_patch = _placed_object_box(x, y, matched.shape[1], matched.shape[0], refined.shape)
                if box_patch[2] > box_patch[0] and box_patch[3] > box_patch[1]:
                    ox1 = box_patch[0] - x; oy1 = box_patch[1] - y; ox2 = ox1 + (box_patch[2] - box_patch[0]); oy2 = oy1 + (box_patch[3] - box_patch[1])
                    base_patch = base[box_patch[1]:box_patch[3], box_patch[0]:box_patch[2]]
                    refined_patch = refined[box_patch[1]:box_patch[3], box_patch[0]:box_patch[2]]
                    obj_patch = matched[oy1:oy2, ox1:ox2]
                    refined[box_patch[1]:box_patch[3], box_patch[0]:box_patch[2]] = _preserve_object_details(base_patch, refined_patch, obj_patch, cfg)
                debug["used_seamless_clone"] = True
                return refined, debug
            except Exception as e:
                debug["seamless_clone_error"] = str(e)
    return base, debug


def _normalize_map(arr: np.ndarray, hi_pct: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    hi = float(np.percentile(arr, hi_pct))
    lo = float(np.percentile(arr, 1.0))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _label_matches_preference(support_label: str, preferred_labels: Sequence[str]) -> bool:
    s = support_label.lower().strip()
    for pref in preferred_labels:
        p = str(pref).lower().strip()
        if p and p in s:
            return True
    return False


def _load_mask(path: Path, shape_if_missing: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape_if_missing, dtype=np.uint8)
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _point_project(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([u, v, 1.0], dtype=np.float32)
    q = H @ p
    q /= max(1e-6, q[2])
    return float(q[0]), float(q[1])


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _as_torch_dtype(name: str):
    n = str(name).lower()
    if n in {"float16", "fp16", "half"}:
        return torch.float16
    if n in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if n in {"float32", "fp32"}:
        return torch.float32
    return None


def _point_in_polygon(x: float, y: float, poly_xy: Sequence[Sequence[float]]) -> bool:
    if not poly_xy or len(poly_xy) < 3:
        return False
    inside = False
    n = len(poly_xy)
    for i in range(n):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / max(1e-6, (y2 - y1)) + x1)
        if intersects:
            inside = not inside
    return inside


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-8:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return float(math.hypot(px - qx, py - qy))


def _bottom_center_of_mask(mask_u8: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    y_max = int(ys.max())
    xs_bottom = xs[ys >= y_max - 1]
    if len(xs_bottom) == 0:
        xs_bottom = xs
    return float(xs_bottom.mean()), float(y_max)


def _body_below_contact_ratio(mask_u8: np.ndarray, contact_y: float) -> float:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return 1.0
    below = ys > int(round(contact_y))
    return float(below.sum()) / float(len(ys))


# -----------------------------------------------------------------------------
# Scene structure helpers
# -----------------------------------------------------------------------------

def _compute_scene_structure_maps(scene_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mag = cv2.GaussianBlur(grad_mag, (0, 0), 0.8)
    grad_mag = _normalize_map(grad_mag, 98.0)

    return {
        "gray": gray,
        "grad_mag": grad_mag,
    }


# -----------------------------------------------------------------------------
# Loading scene-understanding output
# -----------------------------------------------------------------------------

def _load_scene_understanding(scene_json_path: Path) -> Dict[str, Any]:
    with scene_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_object_understanding(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _support_debug_mask_paths(scene_json_path: Path, support_id: str) -> Dict[str, Path]:
    out_dir = scene_json_path.parent
    debug_dir = out_dir / "debug"
    return {
        "support_mask": debug_dir / f"{support_id}_support_mask.png",
        "occupied_mask": debug_dir / f"{support_id}_occupied_mask.png",
        "usable_mask": debug_dir / f"{support_id}_usable_mask.png",
        "occluder_mask": debug_dir / f"{support_id}_occluder_mask.png",
    }


# -----------------------------------------------------------------------------
# Scene/object prior accessors
# -----------------------------------------------------------------------------

def _get_scene_prior_payload(scene_understanding: Dict[str, Any]) -> Dict[str, Any]:
    return scene_understanding.get("scene_priors", {}) or {}


def _get_object_metric_dimensions(object_understanding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not object_understanding:
        return {}
    return object_understanding.get("object_priors", {}).get("metric_dimensions", {}) or {}


def _get_object_physical_attributes(object_understanding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not object_understanding:
        return {}
    return object_understanding.get("object_priors", {}).get("physical_attributes", {}) or {}


def _get_object_placement_priors(object_understanding: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not object_understanding:
        return {}
    return object_understanding.get("object_priors", {}).get("placement_priors", {}) or {}


def _get_object_label(object_understanding: Optional[Dict[str, Any]]) -> str:
    if not object_understanding:
        return "object"
    ident = object_understanding.get("object_identity", {}) or {}
    return str(ident.get("user_label") or ident.get("canonical_name") or "object")


# -----------------------------------------------------------------------------
# Local semantic reasoner
# -----------------------------------------------------------------------------

class LocalSemanticReasoner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("semantic", {}).get("enabled", True))
        self.processor = None
        self.model = None
        self.device = str(cfg.get("semantic", {}).get("device", "cuda"))
        self.model_id = str(cfg.get("semantic", {}).get("model_id", "HuggingFaceTB/SmolVLM-500M-Instruct"))
        self.max_new_tokens = int(cfg.get("semantic", {}).get("max_new_tokens", 220))
        self._loaded = False

    def _load(self) -> None:
        if not self.enabled or self._loaded:
            return
        dtype_cfg = str(self.cfg.get("semantic", {}).get("dtype", "auto"))
        torch_dtype = _as_torch_dtype(dtype_cfg)
        if torch_dtype is None and self.device.startswith("cuda"):
            torch_dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def _generate_json(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        self._load()
        assert self.processor is not None
        assert self.model is not None

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=bool(self.cfg["semantic"].get("do_sample", False)),
                temperature=float(self.cfg["semantic"].get("temperature", 0.0)),
            )
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        start = decoded.find("{")
        end = decoded.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(decoded[start:end + 1])
            except Exception:
                pass
        return {}

    def score_support(
        self,
        scene: Image.Image,
        support_crop: Image.Image,
        object_label: str,
        scene_type: str,
        support: Dict[str, Any],
        object_physical: Dict[str, Any],
        object_place: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.enabled or not bool(self.cfg["semantic"].get("use_support_llm", True)):
            return {"score": 0.5, "hard_disallow": False, "preferred_zone": None, "reasoning": ["semantic_disabled"]}

        prompt = (
            "You are choosing the most realistic support surface for placing an object in a scene. "
            "Return STRICT JSON only with keys: score, hard_disallow, preferred_zone, reasoning. "
            f"Object: {object_label}. "
            f"Scene type: {scene_type}. "
            f"Support label: {support.get('label', 'unknown')}. "
            f"Support mode: {support.get('support_mode', 'unknown')}. "
            f"Object physical attributes: {json.dumps(object_physical, ensure_ascii=False)}. "
            f"Object placement priors: {json.dumps(object_place, ensure_ascii=False)}. "
            "Score must be 0 to 1 and should reflect how natural and realistic this support is for the object. "
            "Use strong distinctions: "
            "0.85-1.00 = highly natural support for this object in this scene; "
            "0.55-0.84 = plausible but not best; "
            "0.25-0.54 = physically possible but awkward or uncommon; "
            "0.00-0.24 = implausible support. "
            "Set hard_disallow=true only if this support is clearly wrong, unsafe, or unrealistic for the object. "
            "preferred_zone should be one of: center, center_back, center_front, left, right, back, front, none."
        )
        out = self._generate_json(support_crop, prompt)
        return {
            "score": _clamp01(_safe_float(out.get("score"), 0.5)),
            "hard_disallow": bool(out.get("hard_disallow", False)),
            "preferred_zone": out.get("preferred_zone", None),
            "reasoning": list(out.get("reasoning", [])) if isinstance(out.get("reasoning", []), list) else [],
        }

    def score_candidate(
        self,
        preview: Image.Image,
        object_label: str,
        support: Dict[str, Any],
        candidate: PlacementCandidate,
    ) -> Dict[str, Any]:
        if not self.enabled or not bool(self.cfg["semantic"].get("use_candidate_llm", True)):
            return {"plausibility": 0.5, "obstruction_risk": 0.5, "awkwardness": 0.5, "reasoning": ["semantic_disabled"]}

        prompt = (
            "Judge if the placed object looks realistic in the scene. "
            "Return STRICT JSON only with keys: plausibility, obstruction_risk, awkwardness, reasoning. "
            f"Object: {object_label}. "
            f"Support label: {support.get('label', 'unknown')}. "
            f"Support mode: {support.get('support_mode', 'unknown')}. "
            f"Candidate box: x={candidate.x}, y={candidate.y}, w={candidate.width}, h={candidate.height}. "
            "Scores must be 0 to 1, where higher plausibility is better, and higher obstruction_risk / awkwardness are worse."
        )
        out = self._generate_json(preview, prompt)
        return {
            "plausibility": _clamp01(_safe_float(out.get("plausibility"), 0.5)),
            "obstruction_risk": _clamp01(_safe_float(out.get("obstruction_risk"), 0.5)),
            "awkwardness": _clamp01(_safe_float(out.get("awkwardness"), 0.5)),
            "reasoning": list(out.get("reasoning", [])) if isinstance(out.get("reasoning", []), list) else [],
        }


# -----------------------------------------------------------------------------
# Support validity filter and LLM-first selection
# -----------------------------------------------------------------------------

def _support_is_valid_for_llm_selection(support: Dict[str, Any], cfg: dict) -> bool:
    usable = float(support.get("usable_area_ratio", 0.0))
    occupied = float(support.get("occupied_area_ratio", 0.0))
    mode = str(support.get("support_mode", "edge")).lower()

    if usable < float(cfg["supports"]["disallow_low_usable_area_below"]):
        return False
    if occupied > float(cfg["supports"]["disallow_occupied_area_above"]):
        return False

    require_geom = bool(cfg["supports"].get("validity_require_plane_or_edge_geometry", False))
    if require_geom:
        if mode == "plane" and support.get("homography_unit_to_img") is None:
            return False
        if mode == "edge" and not support.get("contact_band_xyxy"):
            return False

    return True


def _support_priority(support: Dict[str, Any], cfg: dict) -> float:
    semantic = support.get("semantic_affordance", {}) or {}
    semantic_score = float(semantic.get("score", 0.5))

    if semantic.get("hard_disallow", False):
        return -1e9
    if semantic_score < float(cfg["supports"].get("llm_surface_min_score", 0.20)):
        return -1e9

    if bool(cfg["supports"].get("llm_surface_priority_only", True)):
        return semantic_score

    usable = float(support.get("usable_area_ratio", 0.0))
    occupied = float(support.get("occupied_area_ratio", 0.0))
    conf = float(support.get("confidence", 0.0))
    mode = str(support.get("support_mode", "edge")).lower()
    area = max(
        1.0,
        float((support["box_xyxy"][2] - support["box_xyxy"][0]) * (support["box_xyxy"][3] - support["box_xyxy"][1]))
    )
    border_touch = _safe_float(support.get("border_touch_ratio"), 0.0)

    p = 2.2 * conf + 2.0 * usable - 1.8 * occupied - 0.6 * border_touch + 0.15 * math.log(area)
    if mode == "plane":
        p += float(cfg["supports"].get("plane_bonus", 0.35))
    else:
        p += float(cfg["supports"].get("edge_bonus", 0.10))
    if support.get("homography_unit_to_img") is not None:
        p += float(cfg["supports"].get("homography_bonus", 0.25))

    p += semantic_score * float(cfg["supports"].get("semantic_affordance_weight", 1.7))
    return float(p)


# -----------------------------------------------------------------------------
# Object sizing with scene priors
# -----------------------------------------------------------------------------

def _support_metric_prior_valid(support: Dict[str, Any], cfg: dict) -> bool:
    if not bool(cfg["scaling"].get("use_scene_priors", True)):
        return False
    prior = support.get("prior_dimensions_m", {}) or {}
    conf = _safe_float(prior.get("confidence"), 0.0)
    width_m = _safe_float(prior.get("width_m"), 0.0)
    top_surface_height_m = _safe_float(prior.get("top_surface_height_m"), 0.0)
    return conf >= float(cfg["scaling"]["min_prior_confidence"]) and width_m > 0 and top_surface_height_m > 0


def _metric_target_size_from_support_prior(
    scene_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    support: Dict[str, Any],
    scene_priors: Dict[str, Any],
    v_plane: Optional[float],
    cfg: dict,
) -> Optional[Tuple[int, int, Dict[str, Any]]]:
    if not _support_metric_prior_valid(support, cfg):
        return None

    scene_w, scene_h = scene_size
    obj_w, obj_h = obj_size
    aspect = obj_w / max(1.0, obj_h)

    prior = support.get("prior_dimensions_m", {}) or {}
    scale = support.get("prior_scale", {}) or {}
    room = scene_priors.get("room_dimensions_m", {}) or {}

    support_mode = str(support.get("support_mode", "edge")).lower()
    support_width_m = _safe_float(prior.get("width_m"), 0.0)
    support_depth_m = _safe_float(prior.get("depth_m"), 0.0)
    top_surface_height_m = _safe_float(prior.get("top_surface_height_m"), 0.0)
    support_conf = _safe_float(prior.get("confidence"), 0.0)

    room_height_m = _safe_float(room.get("height_m"), 2.5)
    height_m_per_px = _safe_float(scale.get("height_m_per_px"), 0.0)
    width_m_per_px = _safe_float(scale.get("width_m_per_px"), 0.0)

    metric_height_candidates: List[float] = []

    if support_mode == "plane":
        if support_depth_m > 0 and v_plane is not None:
            v_norm = _clamp01(float(v_plane))
            frac = _lerp(
                float(cfg["scaling"]["prior_support_depth_fraction_min"]),
                float(cfg["scaling"]["prior_support_depth_fraction_max"]),
                v_norm,
            )
            frac *= float(cfg["scaling"].get("plane_depth_bias_toward_front", 1.08))
            metric_height_candidates.append(support_depth_m * frac)
        if support_width_m > 0:
            frac_w = _lerp(
                float(cfg["scaling"]["prior_support_width_fraction_min"]),
                float(cfg["scaling"]["prior_support_width_fraction_max"]),
                0.35 if v_plane is None else _clamp01(float(v_plane)),
            )
            metric_height_candidates.append(support_width_m * frac_w * 0.72)
    else:
        if support_width_m > 0:
            frac_w = _lerp(
                float(cfg["scaling"]["prior_support_width_fraction_min"]),
                float(cfg["scaling"]["prior_support_width_fraction_max"]),
                float(cfg["scaling"].get("edge_depth_assumption_fraction", 0.18)),
            )
            metric_height_candidates.append(support_width_m * frac_w)

    if top_surface_height_m > 0:
        metric_height_candidates.append(top_surface_height_m * 0.55)

    if not metric_height_candidates:
        return None

    target_h_m = float(np.median(metric_height_candidates))
    scene_frac_min = float(cfg["scaling"]["prior_scene_height_fraction_min"])
    scene_frac_max = float(cfg["scaling"]["prior_scene_height_fraction_max"])
    target_h_m = float(np.clip(target_h_m, room_height_m * scene_frac_min, room_height_m * scene_frac_max))

    target_h_px_candidates: List[float] = []
    if height_m_per_px > 1e-8:
        target_h_px_candidates.append(target_h_m / height_m_per_px)
    if width_m_per_px > 1e-8 and aspect > 1e-8:
        target_w_m = target_h_m * aspect
        target_h_px_candidates.append((target_w_m / width_m_per_px) / aspect)
    if not target_h_px_candidates:
        return None

    target_h_px = float(np.median(target_h_px_candidates))
    min_h = scene_h * float(cfg["proposal"]["min_object_height_ratio"])
    max_h = scene_h * float(cfg["proposal"]["max_object_height_ratio"])
    target_h = int(round(np.clip(target_h_px, min_h, max_h)))
    target_w = int(round(target_h * aspect))

    debug = {
        "method": "metric_prior",
        "support_width_m": support_width_m,
        "support_depth_m": support_depth_m,
        "top_surface_height_m": top_surface_height_m,
        "room_height_m": room_height_m,
        "prior_confidence": support_conf,
        "width_m_per_px": width_m_per_px if width_m_per_px > 0 else None,
        "height_m_per_px": height_m_per_px if height_m_per_px > 0 else None,
        "target_height_m": target_h_m,
        "target_height_px_unclamped": target_h_px,
    }
    return max(12, target_w), max(12, target_h), debug


def _metric_target_size_from_object_prior(
    scene_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    support: Dict[str, Any],
    object_metric: Dict[str, Any],
    cfg: dict,
) -> Optional[Tuple[int, int, Dict[str, Any]]]:
    obj_w_m = _safe_float(object_metric.get("width_m"), 0.0)
    obj_d_m = _safe_float(object_metric.get("depth_m"), 0.0)
    obj_h_m = _safe_float(object_metric.get("height_m"), 0.0)
    obj_conf = _safe_float(object_metric.get("confidence"), 0.0)

    if obj_w_m <= 0 or obj_h_m <= 0:
        return None

    prior = support.get("prior_dimensions_m", {}) or {}
    scale = support.get("prior_scale", {}) or {}
    support_w_m = _safe_float(prior.get("width_m"), 0.0)
    support_d_m = _safe_float(prior.get("depth_m"), 0.0)

    if support_w_m > 0 and obj_w_m > support_w_m * 0.92:
        return None
    if support_d_m > 0 and obj_d_m > 0 and obj_d_m > support_d_m * 0.92:
        return None

    scene_w, scene_h = scene_size
    aspect = obj_size[0] / max(1.0, obj_size[1])

    height_m_per_px = _safe_float(scale.get("height_m_per_px"), 0.0)
    width_m_per_px = _safe_float(scale.get("width_m_per_px"), 0.0)

    px_candidates: List[float] = []
    if height_m_per_px > 1e-8:
        px_candidates.append(obj_h_m / height_m_per_px)
    if width_m_per_px > 1e-8:
        target_w_px = obj_w_m / width_m_per_px
        px_candidates.append(target_w_px / max(1e-6, aspect))
    if not px_candidates:
        return None

    target_h_px = float(np.median(px_candidates))
    min_h = scene_h * float(cfg["proposal"]["min_object_height_ratio"])
    max_h = scene_h * float(cfg["proposal"]["max_object_height_ratio"])
    target_h = int(round(np.clip(target_h_px, min_h, max_h)))
    target_w = int(round(target_h * aspect))

    return max(12, target_w), max(12, target_h), {
        "method": "object_metric_prior",
        "object_width_m": obj_w_m,
        "object_depth_m": obj_d_m,
        "object_height_m": obj_h_m,
        "object_confidence": obj_conf,
        "target_height_px_unclamped": target_h_px,
    }


def _heuristic_target_size(
    scene_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    support: Dict[str, Any],
    v_plane: Optional[float],
    cfg: dict,
) -> Tuple[int, int, Dict[str, Any]]:
    scene_w, scene_h = scene_size
    obj_w, obj_h = obj_size
    aspect = obj_w / max(1.0, obj_h)

    sx0, sy0, sx1, sy1 = support["box_xyxy"]
    support_w = max(1.0, float(sx1 - sx0))
    support_h = max(1.0, float(sy1 - sy0))
    support_mode = str(support["support_mode"]).lower()

    width_ratio = np.clip(
        support_w / max(1.0, scene_w),
        float(cfg["scaling"]["support_width_ratio_min"]),
        float(cfg["scaling"]["support_width_ratio_max"]),
    )

    if support_mode == "plane":
        base_h = scene_h * (0.06 + 0.14 * width_ratio)
    else:
        base_h = scene_h * (0.05 + 0.10 * width_ratio)

    if v_plane is not None:
        depth_scale = float(np.interp(
            v_plane,
            [0.0, 1.0],
            [float(cfg["scaling"]["plane_v_perspective_min"]), float(cfg["scaling"]["plane_v_perspective_max"])]
        ))
        base_h *= depth_scale

    min_h = scene_h * float(cfg["proposal"]["min_object_height_ratio"])
    max_h = scene_h * float(cfg["proposal"]["max_object_height_ratio"])
    target_h = int(round(np.clip(base_h, min_h, max_h)))
    target_w = int(round(target_h * aspect))

    debug = {
        "method": "heuristic",
        "support_width_px": support_w,
        "support_height_px": support_h,
        "support_width_ratio": float(width_ratio),
        "base_height_px": float(base_h),
    }
    return max(12, target_w), max(12, target_h), debug


def _compute_dynamic_metric_blend_weight(
    support: Dict[str, Any],
    metric_debug: Dict[str, Any],
    cfg: dict,
) -> Tuple[float, Dict[str, Any]]:
    scaling_cfg = cfg["scaling"]

    prior_conf_raw = _safe_float(metric_debug.get("prior_confidence"), metric_debug.get("object_confidence", 0.0))
    conf_center = float(scaling_cfg.get("blend_prior_confidence_center", 0.30))
    conf_span = max(1e-6, float(scaling_cfg.get("blend_prior_confidence_span", 0.50)))
    prior_conf_norm = _clamp01((prior_conf_raw - conf_center) / conf_span)

    usable = _clamp01(_safe_float(support.get("usable_area_ratio"), 0.0))
    occupied = _clamp01(_safe_float(support.get("occupied_area_ratio"), 0.0))
    border_touch = _clamp01(_safe_float(support.get("border_touch_ratio"), 0.0))
    geometry_quality = _clamp01(0.60 * usable + 0.30 * (1.0 - occupied) + 0.10 * (1.0 - border_touch))

    support_mode = str(support.get("support_mode", "edge")).lower()
    has_h = support.get("homography_unit_to_img") is not None
    if support_mode == "plane":
        homography_quality = 1.0 if has_h else 0.0
    else:
        homography_quality = 0.60

    blend_raw = (
        prior_conf_norm * float(scaling_cfg.get("blend_prior_confidence_weight", 0.55))
        + geometry_quality * float(scaling_cfg.get("blend_geometry_quality_weight", 0.30))
        + homography_quality * float(scaling_cfg.get("blend_homography_quality_weight", 0.15))
    )

    blend_min = float(scaling_cfg.get("blend_weight_min", 0.10))
    blend_max = float(scaling_cfg.get("blend_weight_max", 0.90))
    blend = float(np.clip(blend_raw, blend_min, blend_max))

    return blend, {
        "prior_confidence_raw": prior_conf_raw,
        "prior_confidence_normalized": prior_conf_norm,
        "geometry_quality": geometry_quality,
        "homography_quality": homography_quality,
        "blend_raw": blend_raw,
        "blend_clamped": blend,
    }


def _choose_target_size(
    scene_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    support: Dict[str, Any],
    scene_priors: Dict[str, Any],
    object_metric: Dict[str, Any],
    v_plane: Optional[float],
    cfg: dict,
) -> Tuple[int, int, Dict[str, Any]]:
    heuristic_w, heuristic_h, heuristic_debug = _heuristic_target_size(
        scene_size=scene_size,
        obj_size=obj_size,
        support=support,
        v_plane=v_plane,
        cfg=cfg,
    )

    metric = _metric_target_size_from_object_prior(
        scene_size=scene_size,
        obj_size=obj_size,
        support=support,
        object_metric=object_metric,
        cfg=cfg,
    )

    if metric is None:
        metric = _metric_target_size_from_support_prior(
            scene_size=scene_size,
            obj_size=obj_size,
            support=support,
            scene_priors=scene_priors,
            v_plane=v_plane,
            cfg=cfg,
        )

    if metric is None:
        return heuristic_w, heuristic_h, {
            "sizing_method": "heuristic_only",
            "heuristic": heuristic_debug,
            "metric": None,
            "blend": None,
        }

    metric_w, metric_h, metric_debug = metric
    blend_weight, blend_debug = _compute_dynamic_metric_blend_weight(
        support=support,
        metric_debug=metric_debug,
        cfg=cfg,
    )

    target_w = int(round((1.0 - blend_weight) * heuristic_w + blend_weight * metric_w))
    target_h = int(round((1.0 - blend_weight) * heuristic_h + blend_weight * metric_h))

    scene_w, scene_h = scene_size
    min_h = scene_h * float(cfg["proposal"]["min_object_height_ratio"])
    max_h = scene_h * float(cfg["proposal"]["max_object_height_ratio"])
    target_h = int(round(np.clip(target_h, min_h, max_h)))
    aspect = obj_size[0] / max(1.0, obj_size[1])
    target_w = int(round(target_h * aspect))

    return max(12, target_w), max(12, target_h), {
        "sizing_method": "confidence_weighted_metric_blend",
        "heuristic": heuristic_debug,
        "metric": metric_debug,
        "blend": blend_debug,
    }


# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------

def _candidate_positions_on_plane(
    support: Dict[str, Any],
    target_w: int,
    target_h: int,
    scene_w: int,
    scene_h: int,
    cfg: dict,
) -> List[Tuple[int, int, float, float, float, float]]:
    H_unit_to_img = support.get("homography_unit_to_img")
    if H_unit_to_img is None:
        return []

    H_mat = np.array(H_unit_to_img, dtype=np.float32)
    u_samples = int(cfg["proposal"]["u_samples_plane"])
    v_samples = int(cfg["proposal"]["v_samples_plane"])

    semantic = support.get("semantic_affordance", {}) or {}
    preferred_zone = str(semantic.get("preferred_zone") or "none")

    u_margin = float(cfg["proposal"]["plane_u_margin"])
    v_min = float(cfg["proposal"]["plane_v_min"])
    v_max = float(cfg["proposal"]["plane_v_max"])

    u_centers = np.linspace(u_margin, 1.0 - u_margin, u_samples)
    v_centers = np.linspace(v_min, v_max, v_samples)

    if preferred_zone == "center_back":
        u_centers = np.array(sorted(u_centers, key=lambda u: abs(u - 0.5)))
        v_centers = np.array(sorted(v_centers, key=lambda v: abs(v - 0.30)))
    elif preferred_zone == "back":
        v_centers = np.array(sorted(v_centers, key=lambda v: abs(v - 0.25)))
    elif preferred_zone == "front":
        v_centers = np.array(sorted(v_centers, key=lambda v: abs(v - 0.72)))
    elif preferred_zone == "left":
        u_centers = np.array(sorted(u_centers, key=lambda u: abs(u - 0.25)))
    elif preferred_zone == "right":
        u_centers = np.array(sorted(u_centers, key=lambda u: abs(u - 0.75)))

    out: List[Tuple[int, int, float, float, float, float]] = []
    for v in v_centers:
        for u in u_centers:
            px, py = _point_project(H_mat, float(u), float(v))
            x = int(round(px - target_w * 0.5))
            y = int(round(py - target_h))
            if x < 0 or y < 0 or x + target_w > scene_w or y + target_h > scene_h:
                continue
            out.append((x, y, float(u), float(v), float(px), float(py)))

    uniq = []
    seen = set()
    for x, y, u, v, px, py in out:
        key = (x, y)
        if key not in seen:
            seen.add(key)
            uniq.append((x, y, u, v, px, py))
    return uniq[: int(cfg["proposal"].get("per_support_candidate_cap", 6))]


def _candidate_positions_on_edge(
    support: Dict[str, Any],
    target_w: int,
    target_h: int,
    scene_w: int,
    scene_h: int,
    cfg: dict,
) -> List[Tuple[int, int, Optional[float], Optional[float], Optional[float], Optional[float]]]:
    x0, y0, x1, y1 = support["contact_band_xyxy"]
    band_w = max(1, x1 - x0)

    edge_margin = int(round(band_w * float(cfg["proposal"]["edge_margin_ratio"])))
    usable_left = x0 + edge_margin
    usable_right = x1 - edge_margin

    centers = [usable_left, int(round((usable_left + usable_right - target_w) * 0.5)), usable_right - target_w]
    centers = sorted(set([c for c in centers if c >= usable_left and c + target_w <= usable_right]))

    foot_y = int(round((y0 + y1) * 0.5))
    out: List[Tuple[int, int, Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    for x in centers:
        y = int(round(foot_y - target_h))
        if x >= 0 and y >= 0 and x + target_w <= scene_w and y + target_h <= scene_h:
            contact_x = float(x + target_w * 0.5)
            contact_y = float(y + target_h)
            out.append((x, y, None, None, contact_x, contact_y))
    return out[: int(cfg["proposal"].get("per_support_candidate_cap", 6))]


# -----------------------------------------------------------------------------
# Candidate scoring
# -----------------------------------------------------------------------------

def _object_foot_mask(obj_alpha_u8: np.ndarray) -> np.ndarray:
    h, w = obj_alpha_u8.shape
    foot_h = max(3, int(round(h * 0.18)))
    foot = np.zeros_like(obj_alpha_u8, dtype=np.uint8)
    foot[h - foot_h:h, :] = obj_alpha_u8[h - foot_h:h, :]
    return np.where(foot > 0, 255, 0).astype(np.uint8)


def _place_mask(mask_u8: np.ndarray, x: int, y: int, canvas_shape: Tuple[int, int]) -> np.ndarray:
    H, W = canvas_shape
    h, w = mask_u8.shape
    out = np.zeros((H, W), dtype=np.uint8)

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

    out[y1:y2, x1:x2] = mask_u8[oy1:oy2, ox1:ox2]
    return out


def _mask_overlap_ratio(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    denom = max(1, int(a.sum()))
    return float(np.logical_and(a, b).sum()) / float(denom)


def _outside_support_ratio(obj_mask_u8: np.ndarray, support_mask_u8: np.ndarray) -> float:
    obj = obj_mask_u8 > 0
    sup = support_mask_u8 > 0
    denom = max(1, int(obj.sum()))
    outside = np.logical_and(obj, ~sup)
    return float(outside.sum()) / float(denom)


def _foot_outside_support_ratio(foot_mask_u8: np.ndarray, usable_mask_u8: np.ndarray) -> float:
    foot = foot_mask_u8 > 0
    usable = usable_mask_u8 > 0
    denom = max(1, int(foot.sum()))
    outside = np.logical_and(foot, ~usable)
    return float(outside.sum()) / float(denom)


def _scene_structure_penalty(scene_grad: np.ndarray, obj_box: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = obj_box
    H, W = scene_grad.shape
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    patch = scene_grad[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def _free_space_score(
    support_mask_u8: np.ndarray,
    occupied_mask_u8: np.ndarray,
    candidate_box: Tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = candidate_box
    H, W = support_mask_u8.shape
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))

    support_patch = support_mask_u8[y0:y1, x0:x1] > 0
    occ_patch = occupied_mask_u8[y0:y1, x0:x1] > 0

    support_px = int(support_patch.sum())
    if support_px <= 0:
        return 0.0
    free_px = int(np.logical_and(support_patch, ~occ_patch).sum())
    return float(free_px) / float(support_px)


def _border_penalty(candidate_box: Tuple[int, int, int, int], scene_size: Tuple[int, int]) -> float:
    x0, y0, x1, y1 = candidate_box
    W, H = scene_size
    margin = min(W, H) * 0.03
    penalty = 0.0
    if x0 < margin:
        penalty += (margin - x0) / margin
    if y0 < margin:
        penalty += (margin - y0) / margin
    if x1 > W - margin:
        penalty += (x1 - (W - margin)) / margin
    if y1 > H - margin:
        penalty += (y1 - (H - margin)) / margin
    return float(max(0.0, penalty))


def _semantic_zone_bonus(
    support: Dict[str, Any],
    x: int,
    y: int,
    width: int,
    height: int,
    support_mode: str,
    u_plane: Optional[float],
    v_plane: Optional[float],
) -> float:
    semantic = support.get("semantic_affordance", {}) or {}
    preferred_zone = str(semantic.get("preferred_zone") or "none")

    if support_mode == "plane" and u_plane is not None and v_plane is not None:
        su = float(u_plane)
        sv = float(v_plane)
    else:
        sx0, sy0, sx1, sy1 = support["box_xyxy"]
        cx = x + width * 0.5
        cy = y + height
        su = (cx - sx0) / max(1.0, sx1 - sx0)
        sv = (cy - sy0) / max(1.0, sy1 - sy0)

    if preferred_zone == "center":
        return 1.0 - min(1.0, abs(su - 0.5) * 1.6)
    if preferred_zone == "center_back":
        return max(0.0, 1.0 - (abs(su - 0.5) * 1.5 + abs(sv - 0.30) * 1.8))
    if preferred_zone == "back":
        return max(0.0, 1.0 - abs(sv - 0.25) * 2.0)
    if preferred_zone == "front":
        return max(0.0, 1.0 - abs(sv - 0.72) * 2.0)
    if preferred_zone == "left":
        return max(0.0, 1.0 - abs(su - 0.25) * 2.0)
    if preferred_zone == "right":
        return max(0.0, 1.0 - abs(su - 0.75) * 2.0)
    return 0.0


def _plane_contact_geometry_penalties(
    support: Dict[str, Any],
    foot_canvas: np.ndarray,
    contact_x: float,
    contact_y: float,
    cfg: dict,
) -> Tuple[float, float]:
    bottom_center = _bottom_center_of_mask(foot_canvas)
    if bottom_center is None:
        return 1.0, 1.0

    bc_x, bc_y = bottom_center
    dist = math.hypot(bc_x - contact_x, bc_y - contact_y)
    max_dist = max(1.0, float(cfg["obstacle_avoidance"].get("plane_contact_point_max_distance_px", 18.0)))
    contact_penalty = min(1.0, dist / max_dist)

    body_below_ratio = _body_below_contact_ratio(foot_canvas, contact_y)
    body_below_penalty = min(
        1.0,
        body_below_ratio / max(1e-6, float(cfg["obstacle_avoidance"].get("plane_body_below_contact_ratio_max", 0.08))),
    )
    return float(contact_penalty), float(body_below_penalty)


def _plane_contact_point_valid(
    support: Dict[str, Any],
    contact_x: float,
    contact_y: float,
) -> bool:
    poly = support.get("visible_top_polygon_xy") or support.get("plane_quad_xy") or []
    if poly and _point_in_polygon(contact_x, contact_y, poly):
        return True

    front_edge = support.get("front_edge_xy") or []
    if len(front_edge) == 2:
        (ax, ay), (bx, by) = front_edge
        d = _distance_point_to_segment(contact_x, contact_y, ax, ay, bx, by)
        if d <= 10.0:
            return True

    return False


def _score_candidate(
    support: Dict[str, Any],
    support_priority: float,
    support_masks: Dict[str, np.ndarray],
    scene_maps: Dict[str, np.ndarray],
    obj_alpha_u8: np.ndarray,
    obj_foot_u8: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    u_plane: Optional[float],
    v_plane: Optional[float],
    contact_x: Optional[float],
    contact_y: Optional[float],
    scene_size: Tuple[int, int],
    cfg: dict,
    size_debug: Optional[Dict[str, Any]] = None,
) -> Optional[PlacementCandidate]:
    scene_w, scene_h = scene_size
    support_mode = str(support["support_mode"]).lower()

    obj_alpha_resized = _resize_rgba(np.dstack([np.zeros_like(obj_alpha_u8)] * 3 + [obj_alpha_u8]), width, height)[:, :, 3]
    foot_resized = cv2.resize(obj_foot_u8, (width, height), interpolation=cv2.INTER_NEAREST)

    obj_canvas = _place_mask(obj_alpha_resized, x, y, (scene_h, scene_w))
    foot_canvas = _place_mask(foot_resized, x, y, (scene_h, scene_w))

    x0, y0, x1, y1 = x, y, x + width, y + height
    candidate_box = (x0, y0, x1, y1)

    support_mask = support_masks["support_mask"]
    occupied_mask = support_masks["occupied_mask"]
    usable_mask = support_masks["usable_mask"]
    occluder_mask = support_masks["occluder_mask"]

    if support_mode == "plane":
        outside_support = 0.0
        foot_outside = _foot_outside_support_ratio(foot_canvas, usable_mask)
        free_score = 1.0 - foot_outside
    else:
        outside_support = _outside_support_ratio(obj_canvas, support_mask)
        foot_outside = _foot_outside_support_ratio(foot_canvas, usable_mask)
        free_score = _free_space_score(support_mask, occupied_mask, candidate_box)

    occ_overlap = _mask_overlap_ratio(obj_canvas, occupied_mask)
    occluder_overlap = _mask_overlap_ratio(obj_canvas, occluder_mask)
    structure_pen = _scene_structure_penalty(scene_maps["grad_mag"], candidate_box)
    border_pen = _border_penalty(candidate_box, (scene_w, scene_h))

    plane_contact_penalty = 0.0
    plane_body_below_penalty = 0.0

    if support_mode == "plane":
        if contact_x is None or contact_y is None:
            return None
        if not _plane_contact_point_valid(support, contact_x, contact_y):
            return None
        plane_contact_penalty, plane_body_below_penalty = _plane_contact_geometry_penalties(
            support=support,
            foot_canvas=foot_canvas,
            contact_x=contact_x,
            contact_y=contact_y,
            cfg=cfg,
        )

    if occluder_overlap > float(cfg["obstacle_avoidance"]["hard_reject_if_occluder_above"]):
        return None
    if outside_support > float(cfg["obstacle_avoidance"]["hard_reject_if_outside_support"]):
        return None

    support_cx = 0.5 * (support["box_xyxy"][0] + support["box_xyxy"][2])
    cand_cx = 0.5 * (x0 + x1)
    center_alignment = abs(cand_cx - support_cx) / max(1.0, support["box_xyxy"][2] - support["box_xyxy"][0])

    plane_depth_pref = 0.0
    if support_mode == "plane" and v_plane is not None:
        plane_depth_pref = abs(v_plane - 0.42)

    alpha_bbox = _alpha_bbox(obj_alpha_resized, thr=10)
    obj_crop_tightness = 0.0
    if alpha_bbox != (0, 0, 0, 0):
        bx0, by0, bx1, by1 = alpha_bbox
        fill_ratio = ((bx1 - bx0) * (by1 - by0)) / max(1.0, width * height)
        obj_crop_tightness = 1.0 - fill_ratio

    size_prior = 0.0
    scene_h_ratio = height / max(1.0, scene_h)
    min_h = float(cfg["proposal"]["min_object_height_ratio"])
    max_h = float(cfg["proposal"]["max_object_height_ratio"])
    if scene_h_ratio < min_h:
        size_prior += (min_h - scene_h_ratio) / max(1e-6, min_h)
    if scene_h_ratio > max_h:
        size_prior += (scene_h_ratio - max_h) / max(1e-6, max_h)

    semantic_zone = _semantic_zone_bonus(
        support=support,
        x=x,
        y=y,
        width=width,
        height=height,
        support_mode=support_mode,
        u_plane=u_plane,
        v_plane=v_plane,
    )
    semantic_score = _safe_float((support.get("semantic_affordance", {}) or {}).get("score"), 0.5)
    candidate_semantic_bonus = 0.7 * semantic_zone + 0.6 * semantic_score
    candidate_semantic_penalty = 0.0

    object_physical = (size_debug or {}).get("object_physical_attributes", {}) or {}
    object_place = (size_debug or {}).get("object_placement_priors", {}) or {}

    if object_place:
        preferred_labels = list(object_place.get("preferred_support_labels", []) or [])
        avoid_labels = list(object_place.get("avoid_support_labels", []) or [])
        support_label_l = str(support.get("label", "")).lower()

        if preferred_labels:
            if _label_matches_preference(support_label_l, preferred_labels):
                candidate_semantic_bonus += 1.8
            else:
                candidate_semantic_penalty += 2.2

        if avoid_labels and _label_matches_preference(support_label_l, avoid_labels):
            candidate_semantic_penalty += 3.0

        if bool(object_place.get("allow_hanging_over_edge", False)) is False:
            candidate_semantic_penalty += foot_outside * 2.0
        contact_patch_ratio = _safe_float(object_place.get("contact_patch_ratio"), 0.2)
        if support_mode == "edge" and contact_patch_ratio > 0.25:
            candidate_semantic_penalty += 1.0

    if object_physical:
        if bool(object_physical.get("top_heavy", False)):
            candidate_semantic_penalty += center_alignment * 0.7
            candidate_semantic_penalty += border_pen * 0.8
        orientation = str(object_physical.get("typical_orientation", "unknown"))
        if orientation == "upright" and height < width * 0.5:
            candidate_semantic_penalty += 0.8
        if orientation == "flat" and height > width * 1.2:
            candidate_semantic_penalty += 0.8
        if bool(object_physical.get("stable_base", False)) is False and support_mode == "edge":
            candidate_semantic_penalty += 1.2

    score = (
        -support_priority * float(cfg["ranking"]["support_priority_weight"])
        + occ_overlap * float(cfg["obstacle_avoidance"]["occupied_weight"])
        + occluder_overlap * float(cfg["obstacle_avoidance"]["occluder_weight"])
        + outside_support * float(cfg["obstacle_avoidance"]["support_outside_weight"])
        + foot_outside * float(cfg["obstacle_avoidance"]["edge_contact_outside_weight"])
        + structure_pen * float(cfg["obstacle_avoidance"]["scene_structure_weight"])
        + border_pen * float(cfg["obstacle_avoidance"]["border_penalty_weight"])
        + center_alignment * float(cfg["ranking"]["center_alignment_weight"])
        + plane_depth_pref * float(cfg["ranking"]["plane_depth_preference_weight"])
        - free_score * float(cfg["ranking"]["free_space_weight"])
        + obj_crop_tightness * float(cfg["ranking"]["object_crop_tightness_weight"])
        + size_prior * float(cfg["ranking"]["size_prior_weight"])
        + candidate_semantic_penalty * float(cfg["ranking"].get("semantic_penalty_weight", 1.6))
        - candidate_semantic_bonus * float(cfg["ranking"].get("semantic_bonus_weight", 1.2))
        + plane_contact_penalty * float(cfg["ranking"].get("plane_contact_alignment_weight", 1.0))
        + plane_body_below_penalty * float(cfg["ranking"].get("plane_body_below_contact_weight", 2.8))
    )

    return PlacementCandidate(
        support_id=str(support["id"]),
        support_label=str(support["label"]),
        support_mode=support_mode,
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
        score=float(score),
        support_priority=float(support_priority),
        contact_x=None if contact_x is None else float(contact_x),
        contact_y=None if contact_y is None else float(contact_y),
        u_plane=None if u_plane is None else float(u_plane),
        v_plane=None if v_plane is None else float(v_plane),
        debug={
            "occupied_overlap": float(occ_overlap),
            "occluder_overlap": float(occluder_overlap),
            "outside_support": float(outside_support),
            "foot_outside_usable": float(foot_outside),
            "scene_structure_penalty": float(structure_pen),
            "free_space_score": float(free_score),
            "border_penalty": float(border_pen),
            "center_alignment": float(center_alignment),
            "plane_depth_preference": float(plane_depth_pref),
            "size_prior_penalty": float(size_prior),
            "semantic_zone_bonus": float(semantic_zone),
            "semantic_support_score": float(semantic_score),
            "candidate_semantic_bonus": float(candidate_semantic_bonus),
            "candidate_semantic_penalty": float(candidate_semantic_penalty),
            "plane_contact_penalty": float(plane_contact_penalty),
            "plane_body_below_contact_penalty": float(plane_body_below_penalty),
            "u_plane": None if u_plane is None else float(u_plane),
            "v_plane": None if v_plane is None else float(v_plane),
            "contact_x": None if contact_x is None else float(contact_x),
            "contact_y": None if contact_y is None else float(contact_y),
            "size_debug": size_debug or {},
        },
    )


# -----------------------------------------------------------------------------
# Debug rendering
# -----------------------------------------------------------------------------

def _draw_candidate_overlay(
    scene: Image.Image,
    selected: PlacementCandidate,
    ranked: Sequence[PlacementCandidate],
    support_lookup: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    img = scene.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for cand in ranked[:10]:
        x0, y0 = cand.x, cand.y
        x1, y1 = cand.x + cand.width, cand.y + cand.height
        draw.rectangle([x0, y0, x1, y1], outline=(255, 180, 0), width=2)
        if cand.contact_x is not None and cand.contact_y is not None:
            r = 4
            draw.ellipse(
                [cand.contact_x - r, cand.contact_y - r, cand.contact_x + r, cand.contact_y + r],
                outline=(255, 180, 0),
                width=2,
            )

    support = support_lookup.get(selected.support_id, {})
    plane_quad = support.get("plane_quad_xy") or []
    if plane_quad and len(plane_quad) >= 4:
        draw.line([tuple(map(float, p)) for p in plane_quad] + [tuple(map(float, plane_quad[0]))], fill=(0, 255, 255), width=3)

    front_edge = support.get("front_edge_xy") or []
    if len(front_edge) == 2:
        draw.line([tuple(map(float, front_edge[0])), tuple(map(float, front_edge[1]))], fill=(255, 0, 255), width=3)

    sx0, sy0 = selected.x, selected.y
    sx1, sy1 = selected.x + selected.width, selected.y + selected.height
    draw.rectangle([sx0, sy0, sx1, sy1], outline=(0, 255, 0), width=3)

    if selected.contact_x is not None and selected.contact_y is not None:
        r = 5
        draw.ellipse(
            [selected.contact_x - r, selected.contact_y - r, selected.contact_x + r, selected.contact_y + r],
            outline=(0, 255, 0),
            width=3,
        )

    draw.text((sx0 + 4, max(0, sy0 - 16)), f"{selected.support_label} {selected.score:.3f}", fill=(255, 255, 255))
    img.save(out_path)


def _crop_support_with_context(scene: Image.Image, support_box: Sequence[float], context_ratio: float) -> Image.Image:
    W, H = scene.size
    x0, y0, x1, y1 = [int(round(v)) for v in support_box]
    bw = x1 - x0
    bh = y1 - y0
    pad_x = int(round(bw * context_ratio))
    pad_y = int(round(bh * context_ratio))
    cx0 = max(0, x0 - pad_x)
    cy0 = max(0, y0 - pad_y)
    cx1 = min(W, x1 + pad_x)
    cy1 = min(H, y1 + pad_y)
    return scene.crop((cx0, cy0, cx1, cy1))


def _resize_preview(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BICUBIC)


# -----------------------------------------------------------------------------
# Core placement engine
# -----------------------------------------------------------------------------

class PlacementReasoner:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.semantic = LocalSemanticReasoner(cfg)

    def _preselect_supports(self, supports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = [s for s in supports if _support_is_valid_for_llm_selection(s, self.cfg)]
        keep = int(self.cfg["supports"].get("preselect_top_k_before_llm", len(out)))
        return out[:keep]

    def _semantic_support_pass(
        self,
        scene: Image.Image,
        scene_type: str,
        supports: List[Dict[str, Any]],
        object_label: str,
        object_physical: Dict[str, Any],
        object_place: Dict[str, Any],
    ) -> None:
        context_ratio = float(self.cfg["semantic"].get("support_crop_context", 0.24))
        for support in supports:
            crop = _crop_support_with_context(scene, support["box_xyxy"], context_ratio)
            sem = self.semantic.score_support(
                scene=scene,
                support_crop=crop,
                object_label=object_label,
                scene_type=scene_type,
                support=support,
                object_physical=object_physical,
                object_place=object_place,
            )
            support["semantic_affordance"] = sem

    def _select_supports_with_llm(self, supports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        min_score = float(self.cfg["supports"].get("llm_surface_min_score", 0.20))
        for s in supports:
            sem = s.get("semantic_affordance", {}) or {}
            if sem.get("hard_disallow", False):
                continue
            if float(sem.get("score", 0.5)) < min_score:
                continue
            valid.append(s)

        if not valid:
            ranked_all = sorted(
                supports,
                key=lambda s: float((s.get("semantic_affordance", {}) or {}).get("score", 0.5)),
                reverse=True,
            )
            return ranked_all[:1]

        valid.sort(
            key=lambda s: float((s.get("semantic_affordance", {}) or {}).get("score", 0.5)),
            reverse=True,
        )
        top_k = int(self.cfg["supports"].get("llm_surface_top_k", 2))
        return valid[:top_k]

    def _candidate_llm_rerank(
        self,
        scene_np: np.ndarray,
        obj_rgba_np: np.ndarray,
        candidates: List[PlacementCandidate],
        support_lookup: Dict[str, Dict[str, Any]],
        object_label: str,
    ) -> None:
        top_k = int(self.cfg["proposal"].get("final_llm_top_k", 4))
        max_side = int(self.cfg["semantic"].get("candidate_preview_max_side", 768))
        subset = candidates[:top_k]
        for cand in subset:
            preview_np = _place_rgba_over_rgb(scene_np, _resize_rgba(obj_rgba_np, cand.width, cand.height), cand.x, cand.y)
            preview = _resize_preview(_np_to_pil_rgb(preview_np), max_side=max_side)
            llm = self.semantic.score_candidate(
                preview=preview,
                object_label=object_label,
                support=support_lookup[cand.support_id],
                candidate=cand,
            )
            llm_delta = (
                - llm["plausibility"] * float(self.cfg["ranking"].get("llm_candidate_weight", 2.2))
                + llm["obstruction_risk"] * 1.6
                + llm["awkwardness"] * 1.8
            )
            cand.score += llm_delta
            cand.debug["candidate_llm"] = llm
            cand.debug["candidate_llm_delta"] = float(llm_delta)

    def place(
        self,
        scene_path: Path,
        scene_understanding_json: Path,
        object_rgba_path: Path,
        output_dir: Path,
        object_understanding: Optional[Dict[str, Any]] = None,
    ) -> PlacementResult:
        _ensure_dir(output_dir)
        debug_dir = output_dir / "debug"
        if bool(self.cfg["output"].get("save_debug", True)):
            _ensure_dir(debug_dir)

        scene = _open_rgb(scene_path)
        scene_np = _pil_to_np_rgb(scene)
        scene_h, scene_w = scene_np.shape[:2]
        scene_maps = _compute_scene_structure_maps(scene_np)

        su = _load_scene_understanding(scene_understanding_json)
        scene_priors = _get_scene_prior_payload(su)
        scene_type = str(scene_priors.get("scene_type") or su.get("scene_type") or "unknown")

        object_metric = _get_object_metric_dimensions(object_understanding)
        object_physical = _get_object_physical_attributes(object_understanding)
        object_place = _get_object_placement_priors(object_understanding)
        object_label = _get_object_label(object_understanding)

        raw_supports = list(su.get("supports", []))
        raw_supports = raw_supports[: int(self.cfg["supports"]["max_supports_to_consider"])]

        supports_prefiltered = self._preselect_supports(raw_supports)

        self._semantic_support_pass(
            scene=scene,
            scene_type=scene_type,
            supports=supports_prefiltered,
            object_label=object_label,
            object_physical=object_physical,
            object_place=object_place,
        )

        if bool(self.cfg["supports"].get("llm_surface_selection_enabled", True)):
            supports = self._select_supports_with_llm(supports_prefiltered)
        else:
            supports = supports_prefiltered

        support_lookup = {str(s["id"]): s for s in supports}

        obj_rgba = _crop_rgba_to_alpha(_open_rgba(object_rgba_path), pad=int(self.cfg["input"]["object_pad_px"]))
        obj_rgba_np = _pil_to_np_rgba(obj_rgba)
        obj_alpha_u8 = obj_rgba_np[:, :, 3]
        obj_foot_u8 = _object_foot_mask(obj_alpha_u8)

        ranked_candidates: List[PlacementCandidate] = []

        for support in supports:
            s_priority = _support_priority(support, self.cfg)
            if s_priority <= -1e8:
                continue

            mask_paths = _support_debug_mask_paths(scene_understanding_json, str(support["id"]))
            support_masks = {
                "support_mask": _load_mask(mask_paths["support_mask"], (scene_h, scene_w)),
                "occupied_mask": _load_mask(mask_paths["occupied_mask"], (scene_h, scene_w)),
                "usable_mask": _load_mask(mask_paths["usable_mask"], (scene_h, scene_w)),
                "occluder_mask": _load_mask(mask_paths["occluder_mask"], (scene_h, scene_w)),
            }

            support_mode = str(support["support_mode"]).lower()
            if support_mode == "plane" and support.get("homography_unit_to_img") is not None:
                preferred_zone = str((support.get("semantic_affordance", {}) or {}).get("preferred_zone") or "none")
                if preferred_zone in {"back", "center_back"}:
                    v_pref = 0.30
                elif preferred_zone == "front":
                    v_pref = 0.70
                else:
                    v_pref = 0.42

                tw, th, size_debug = _choose_target_size(
                    scene_size=(scene_w, scene_h),
                    obj_size=obj_rgba.size,
                    support=support,
                    scene_priors=scene_priors,
                    object_metric=object_metric,
                    v_plane=v_pref,
                    cfg=self.cfg,
                )
                size_debug = dict(size_debug)
                size_debug["object_metric_dimensions"] = object_metric
                size_debug["object_physical_attributes"] = object_physical
                size_debug["object_placement_priors"] = object_place
                size_debug["support_semantic_affordance"] = support.get("semantic_affordance", {})

                positions = _candidate_positions_on_plane(
                    support=support,
                    target_w=tw,
                    target_h=th,
                    scene_w=scene_w,
                    scene_h=scene_h,
                    cfg=self.cfg,
                )
                for x, y, u, v, contact_x, contact_y in positions:
                    cand = _score_candidate(
                        support=support,
                        support_priority=s_priority,
                        support_masks=support_masks,
                        scene_maps=scene_maps,
                        obj_alpha_u8=obj_alpha_u8,
                        obj_foot_u8=obj_foot_u8,
                        x=x,
                        y=y,
                        width=tw,
                        height=th,
                        u_plane=u,
                        v_plane=v,
                        contact_x=contact_x,
                        contact_y=contact_y,
                        scene_size=(scene_w, scene_h),
                        cfg=self.cfg,
                        size_debug=size_debug,
                    )
                    if cand is not None:
                        ranked_candidates.append(cand)
            else:
                tw, th, size_debug = _choose_target_size(
                    scene_size=(scene_w, scene_h),
                    obj_size=obj_rgba.size,
                    support=support,
                    scene_priors=scene_priors,
                    object_metric=object_metric,
                    v_plane=None,
                    cfg=self.cfg,
                )
                size_debug = dict(size_debug)
                size_debug["object_metric_dimensions"] = object_metric
                size_debug["object_physical_attributes"] = object_physical
                size_debug["object_placement_priors"] = object_place
                size_debug["support_semantic_affordance"] = support.get("semantic_affordance", {})

                positions = _candidate_positions_on_edge(
                    support=support,
                    target_w=tw,
                    target_h=th,
                    scene_w=scene_w,
                    scene_h=scene_h,
                    cfg=self.cfg,
                )
                for x, y, u, v, contact_x, contact_y in positions:
                    cand = _score_candidate(
                        support=support,
                        support_priority=s_priority,
                        support_masks=support_masks,
                        scene_maps=scene_maps,
                        obj_alpha_u8=obj_alpha_u8,
                        obj_foot_u8=obj_foot_u8,
                        x=x,
                        y=y,
                        width=tw,
                        height=th,
                        u_plane=u,
                        v_plane=v,
                        contact_x=contact_x,
                        contact_y=contact_y,
                        scene_size=(scene_w, scene_h),
                        cfg=self.cfg,
                        size_debug=size_debug,
                    )
                    if cand is not None:
                        ranked_candidates.append(cand)

        if not ranked_candidates:
            raise RuntimeError("No plausible placement candidates were found.")

        ranked_candidates.sort(key=lambda c: c.score)
        ranked_candidates = ranked_candidates[: int(self.cfg["proposal"]["top_k_to_keep"])]

        self._candidate_llm_rerank(
            scene_np=scene_np,
            obj_rgba_np=obj_rgba_np,
            candidates=ranked_candidates,
            support_lookup=support_lookup,
            object_label=object_label,
        )

        ranked_candidates.sort(key=lambda c: c.score)
        ranked_candidates = ranked_candidates[: int(self.cfg["proposal"]["top_k_to_keep"])]

        idx = max(0, min(int(self.cfg["proposal"]["attempt_index"]), len(ranked_candidates) - 1))
        selected = ranked_candidates[idx]

        obj_scaled = _resize_rgba(obj_rgba_np, selected.width, selected.height)
        composite_np, refine_debug = _refine_composite(scene_np, obj_scaled, selected.x, selected.y, self.cfg)
        composite = _np_to_pil_rgb(composite_np)

        composite_path = output_dir / "composite_raw.png"
        composite.save(composite_path)
        selected.debug["refinement"] = refine_debug

        placement_json_path = None
        if bool(self.cfg["output"].get("save_json", True)):
            placement_json_path = output_dir / "placement_result.json"
            payload = {
                "scene_priors_used": scene_priors,
                "object_understanding_used": object_understanding or {},
                "supports_prefiltered": supports_prefiltered,
                "supports_selected_by_llm": supports,
                "selected_candidate": asdict(selected),
                "ranked_candidates": [asdict(c) for c in ranked_candidates],
            }
            with placement_json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        if bool(self.cfg["output"].get("save_debug", True)):
            _draw_candidate_overlay(
                scene=scene,
                selected=selected,
                ranked=ranked_candidates,
                support_lookup=support_lookup,
                out_path=debug_dir / "01_candidates_overlay.png",
            )
            obj_preview = _np_to_pil_rgb(composite_np)
            obj_preview.save(debug_dir / "02_selected_composite_preview.png")
            coarse_preview = _np_to_pil_rgb(_place_rgba_over_rgb(scene_np, obj_scaled, selected.x, selected.y))
            coarse_preview.save(debug_dir / "02b_selected_composite_coarse.png")
            with (debug_dir / "03_candidates.txt").open("w", encoding="utf-8") as f:
                for i, cand in enumerate(ranked_candidates):
                    f.write(
                        f"[{i}] "
                        f"support={cand.support_id} "
                        f"label={cand.support_label} "
                        f"mode={cand.support_mode} "
                        f"x={cand.x} y={cand.y} w={cand.width} h={cand.height} "
                        f"contact_x={cand.contact_x} contact_y={cand.contact_y} "
                        f"u={cand.u_plane} v={cand.v_plane} "
                        f"score={cand.score:.5f} "
                        f"debug={json.dumps(cand.debug, sort_keys=True)}\n"
                    )

        return PlacementResult(
            composite_path=str(composite_path),
            placement_json_path=None if placement_json_path is None else str(placement_json_path),
            debug_dir=str(debug_dir) if bool(self.cfg["output"].get("save_debug", True)) else None,
            selected_candidate=selected,
            ranked_candidates=ranked_candidates,
        )


# -----------------------------------------------------------------------------
# Public function
# -----------------------------------------------------------------------------

def place_object_in_scene(
    scene_path: str | Path,
    scene_understanding_json: str | Path,
    object_rgba_path: str | Path,
    output_dir: str | Path,
    cfg: dict,
    object_understanding: Optional[Dict[str, Any]] = None,
) -> PlacementResult:
    engine = PlacementReasoner(cfg=cfg)
    return engine.place(
        scene_path=Path(scene_path),
        scene_understanding_json=Path(scene_understanding_json),
        object_rgba_path=Path(object_rgba_path),
        output_dir=Path(output_dir),
        object_understanding=object_understanding,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placement reasoning using scene-understanding output and extracted object RGBA."
    )
    parser.add_argument("--scene", required=True, type=str, help="Path to the scene image.")
    parser.add_argument("--scene-understanding", required=True, type=str, help="Path to scene_understanding.json.")
    parser.add_argument("--object-rgba", required=True, type=str, help="Path to extracted/cutout RGBA object image.")
    parser.add_argument("--object-understanding", required=False, type=str, help="Path to object_understanding.json.")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--config", required=True, type=str, help="Path to placement YAML config.")
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.config))
    object_understanding = (
        _load_object_understanding(Path(args.object_understanding))
        if args.object_understanding
        else None
    )

    result = place_object_in_scene(
        scene_path=args.scene,
        scene_understanding_json=args.scene_understanding,
        object_rgba_path=args.object_rgba,
        output_dir=args.output,
        cfg=cfg,
        object_understanding=object_understanding,
    )
    print(result.composite_path)


if __name__ == "__main__":
    main()