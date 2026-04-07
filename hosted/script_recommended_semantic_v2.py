
import argparse
import base64
import io
import json
import os
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import InferenceClient

try:
    import torch
except Exception:
    torch = None


DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit"
DEFAULT_OBJECT_SEG_MODEL = "briaai/RMBG-2.0"
DEFAULT_SCENE_SEG_MODEL = "facebook/mask2former-swin-large-coco-panoptic"
DEFAULT_PROVIDER = "auto"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# Internal policy knobs.
INTERNAL_SEG_MAX_SCENE_SIDE = 1536
INTERNAL_EDIT_MAX_SIDE = 1024
MIN_LOCAL_PADDING_PX = 80
MAX_LOCAL_PADDING_FRACTION = 0.25
LOCAL_MASK_BLUR_RADIUS = 12

# Conservative sizing policy. These are intentionally a bit smaller than before.
MIN_OBJECT_WIDTH_FRAC = 0.055
DEFAULT_OBJECT_WIDTH_FRAC = 0.085
MAX_OBJECT_WIDTH_FRAC = 0.135
ANCHOR_SEARCH_SIDE_FRAC = 0.22
MIN_SUPPORT_SPAN_FRAC = 0.10

# New placement policy knobs.
HARD_BLOCKER_MARGIN_PX = 10
SOFT_BLOCKER_MARGIN_PX = 5
SUPPORT_EDGE_INSET_PX = 6
MIN_BLOB_AREA_FRAC = 0.0035
MAX_COMPONENTS_PER_SUPPORT = 8
MAX_TOTAL_CANDIDATES = 24
FOOTPRINT_SWEEP_STEPS = 13
MIN_SUPPORTED_FRACTION = 0.82
HUMAN_OVERLAP_REJECT_FRAC = 0.001
HARD_OVERLAP_REJECT_FRAC = 0.004


@dataclass
class Placement:
    left: int
    top: int
    width: int
    height: int
    support_label: Optional[str]


@dataclass
class SupportCandidate:
    label: str
    mask: Image.Image
    bbox: tuple[int, int, int, int]
    support_y: int
    anchor_x: int
    local_span_width: int
    flatness: float
    score: float
    surface_type: str = "support"
    top_visibility: float = 0.0
    depth_value: float = 0.5
    occlusion_penalty: float = 0.0
    support_fraction: float = 1.0
    human_overlap_penalty: float = 0.0
    hard_overlap_penalty: float = 0.0
    soft_overlap_penalty: float = 0.0


_DEPTH_PIPELINE = None


def _build_depth_pipeline():
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is not None:
        return _DEPTH_PIPELINE
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "Monocular depth support detection requires transformers. Install it with: pip install transformers"
        ) from exc

    device = 0 if (torch is not None and torch.cuda.is_available()) else -1
    _DEPTH_PIPELINE = pipeline("depth-estimation", model=DEFAULT_DEPTH_MODEL, device=device)
    return _DEPTH_PIPELINE


def estimate_monocular_depth(scene_rgb: Image.Image) -> np.ndarray:
    pipe = _build_depth_pipeline()
    result = pipe(scene_rgb.convert("RGB"))

    depth_arr = None
    depth_img = result.get("depth") if isinstance(result, dict) else None
    if isinstance(depth_img, Image.Image):
        depth_arr = np.asarray(depth_img.resize(scene_rgb.size, Image.LANCZOS), dtype=np.float32)

    if depth_arr is None:
        pred = result.get("predicted_depth") if isinstance(result, dict) else None
        if pred is None:
            raise RuntimeError("Depth pipeline did not return a usable depth map.")
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 4:
            pred = pred[0, 0]
        elif pred.ndim == 3:
            pred = pred[0]
        depth_arr = pred
        if depth_arr.shape[::-1] != scene_rgb.size:
            depth_img = Image.fromarray(depth_arr)
            depth_arr = np.asarray(depth_img.resize(scene_rgb.size, Image.LANCZOS), dtype=np.float32)

    depth_arr = np.asarray(depth_arr, dtype=np.float32)
    depth_arr -= float(depth_arr.min())
    denom = float(depth_arr.max())
    if denom > 1e-6:
        depth_arr /= denom

    row_profile = depth_arr.mean(axis=1)
    row_idx = np.linspace(0.0, 1.0, len(row_profile), dtype=np.float32)
    corr = float(np.corrcoef(row_profile, row_idx)[0, 1]) if len(row_profile) > 4 else 0.0
    if np.isfinite(corr) and corr < 0:
        depth_arr = 1.0 - depth_arr
    return depth_arr


def depth_to_preview(depth: np.ndarray, size: tuple[int, int]) -> Image.Image:
    arr = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").resize(size, Image.LANCZOS)


def _box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=False)
    out = arr.astype(np.float32, copy=False)
    k = 2 * radius + 1
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        zeros_shape = list(padded.shape)
        zeros_shape[axis] = 1
        padded = np.concatenate([np.zeros(zeros_shape, dtype=np.float32), padded], axis=axis)
        csum = np.cumsum(padded, axis=axis, dtype=np.float32)
        slicer_hi = [slice(None)] * out.ndim
        slicer_lo = [slice(None)] * out.ndim
        slicer_hi[axis] = slice(k, k + out.shape[axis])
        slicer_lo[axis] = slice(0, out.shape[axis])
        out = (csum[tuple(slicer_hi)] - csum[tuple(slicer_lo)]) / float(k)
    return out


def _make_rect_mask(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask


def _mask_bbox(mask: Image.Image) -> Optional[tuple[int, int, int, int]]:
    bbox = mask.getbbox()
    if bbox is None:
        return None
    return tuple(int(v) for v in bbox)


def _iter_spans(active: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(active.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                spans.append((start, i - 1))
            start = None
    return spans


def _depth_preference_to_target(depth_pref: int) -> float:
    depth_pref = max(1, min(10, int(depth_pref)))
    return (depth_pref - 1) / 9.0


def _depth_match_score(candidate_depth: float, depth_pref: int) -> float:
    target = _depth_preference_to_target(depth_pref)
    return max(0.0, 1.0 - abs(float(candidate_depth) - target) * 1.35)


def _compute_rgb_occlusion_penalty(scene_rgb: Image.Image, x0: int, x1: int, y: int, probe_h: int) -> float:
    scene_w, scene_h = scene_rgb.size
    rx0 = max(0, min(scene_w, int(x0)))
    rx1 = max(rx0 + 1, min(scene_w, int(x1)))
    ry0 = max(0, min(scene_h, int(y - probe_h * 0.35)))
    ry1 = max(ry0 + 1, min(scene_h, int(y + probe_h * 0.95)))
    crop = scene_rgb.crop((rx0, ry0, rx1, ry1)).convert("L")
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    if arr.size == 0:
        return 0.0
    gy, gx = np.gradient(arr)
    energy = np.sqrt(gx * gx + gy * gy)
    dense_edges = float(np.mean(energy > max(0.10, float(np.quantile(energy, 0.72)))))
    texture = float(np.mean(energy))
    return min(1.0, dense_edges * 0.7 + texture * 1.8)


def infer_object_dimensions(object_label: str, obj_rgba: Image.Image) -> tuple[float, float]:
    label = object_label.lower()
    priors = [
        (["can", "soda", "ginger ale", "coke", "pepsi", "beer can"], (0.066, 0.122)),
        (["bottle", "wine", "water bottle"], (0.073, 0.285)),
        (["mug", "cup"], (0.085, 0.095)),
        (["laptop"], (0.320, 0.220)),
        (["phone", "iphone", "smartphone"], (0.072, 0.148)),
        (["book"], (0.155, 0.235)),
        (["plate"], (0.260, 0.030)),
        (["bowl"], (0.160, 0.075)),
    ]
    for keys, dims in priors:
        if any(k in label for k in keys):
            return dims
    w, h = obj_rgba.size
    aspect = w / float(max(1, h))
    default_height = 0.14
    default_width = max(0.05, min(0.22, default_height * aspect))
    return default_width, default_height


def estimate_target_height(scene: Image.Image, candidate: Optional[SupportCandidate], object_label: str, obj_rgba: Image.Image) -> int:
    _, scene_h = scene.size
    _, real_h = infer_object_dimensions(object_label, obj_rgba)
    baseline_height = scene_h * min(0.22, max(0.06, real_h * 1.35))
    if candidate is None:
        return int(baseline_height)
    bottomness = min(1.0, max(0.0, candidate.support_y / float(max(1, scene_h))))
    perspective_factor = 0.52 + 0.72 * (bottomness ** 1.45)
    return max(16, int(baseline_height * perspective_factor))


class ImageStatProxy:
    @staticmethod
    def nonzero_fraction(mask: Image.Image) -> float:
        hist = mask.histogram()
        nonzero = sum(hist[1:])
        total = sum(hist)
        return 0.0 if total == 0 else nonzero / float(total)


class DebugRun:
    def __init__(self, requested_output: str):
        requested = Path(requested_output)
        base_dir = requested.parent if requested.parent != Path("") else Path.cwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = requested.stem or "run"
        self.run_dir = base_dir / f"{stem}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.final_output = self.run_dir / requested.name

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def save_image(self, image: Image.Image, name: str, mode: Optional[str] = None, **save_kwargs) -> Path:
        path = self.path(name)
        out = image.convert(mode) if mode else image
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path, **save_kwargs)
        return path

    def save_json(self, payload: dict, name: str = "run_metadata.json") -> Path:
        path = self.path(name)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert a product into a scene while preserving the original full-resolution background outside a local refinement region."
    )
    parser.add_argument("--scene", required=True, help="Path to the scene image")
    parser.add_argument("--object-image", required=True, help="Path to the product/object image")
    parser.add_argument("--object-label", required=True, help="Human label for the object, e.g. 'ginger ale can'")
    parser.add_argument("--output", required=True, help="Path to save the final image")
    parser.add_argument("--prompt", default=None, help="Optional edit instruction")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="HF Inference provider, default: auto")
    parser.add_argument("--edit-model", default=DEFAULT_EDIT_MODEL, help=f"Image edit model, default: {DEFAULT_EDIT_MODEL}")
    parser.add_argument("--object-seg-model", default=DEFAULT_OBJECT_SEG_MODEL, help=f"Object segmentation model, default: {DEFAULT_OBJECT_SEG_MODEL}")
    parser.add_argument("--scene-seg-model", default=DEFAULT_SCENE_SEG_MODEL, help=f"Scene segmentation model, default: {DEFAULT_SCENE_SEG_MODEL}")
    parser.add_argument("--scale", type=float, default=None, help="Optional manual scale multiplier relative to auto scale")
    parser.add_argument("--x", type=float, default=None, help="Optional manual x center as fraction of scene width, e.g. 0.5")
    parser.add_argument("--y", type=float, default=None, help="Optional manual y bottom as fraction of scene height, e.g. 0.78")
    parser.add_argument("--skip-refine", action="store_true", help="Save only the local precomposition without calling the image edit API")
    parser.add_argument(
        "--object-scene-depth",
        type=int,
        default=6,
        help="Desired scene depth from 1-10, where 1 pushes placement toward deeper/background supports and 10 prefers foreground supports.",
    )
    return parser.parse_args()


def build_client(provider: str) -> InferenceClient:
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Missing HF token. Put HF_TOKEN in your .env or environment before running this script.")
    return InferenceClient(provider=provider, api_key=token)


def load_rgba(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def resize_long_side(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def alpha_bbox(alpha: Image.Image) -> Optional[tuple[int, int, int, int]]:
    return alpha.getbbox()


def choose_best_mask(seg_outputs):
    best = None
    best_score = -1.0
    for output in seg_outputs:
        mask = output.mask.convert("L")
        bbox = mask.getbbox()
        if bbox is None:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area <= 0:
            continue
        score = float(getattr(output, "score", 0.0)) * area
        if score > best_score:
            best = mask
            best_score = score
    return best


def extract_object_rgba(obj_image: Image.Image, client: InferenceClient, seg_model: str) -> Image.Image:
    alpha = obj_image.getchannel("A")
    bbox = alpha_bbox(alpha)
    if bbox and ImageStatProxy.nonzero_fraction(alpha) > 0.01:
        return obj_image.crop(bbox)

    seg = client.image_segmentation(pil_to_png_bytes(obj_image.convert("RGB")), model=seg_model)
    mask = choose_best_mask(seg)
    if mask is None:
        raise RuntimeError("Could not extract an object mask from the object image.")

    if mask.size != obj_image.size:
        mask = mask.resize(obj_image.size, Image.LANCZOS)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    out = obj_image.copy()
    out.putalpha(mask)
    bbox = alpha_bbox(mask)
    if bbox is None:
        raise RuntimeError("The extracted object mask is empty.")
    return out.crop(bbox)


def _normalize_label(label: str) -> str:
    return str(label).lower().replace("-", " ").replace("_", " ").strip()


def _semantic_support_allowed(label: str) -> bool:
    norm = _normalize_label(label)
    allow = [
        "table", "desk", "counter", "countertop", "shelf", "cabinet", "nightstand",
        "coffee table", "dining table", "end table", "worktop", "kitchen island", "dresser",
    ]
    deny = [
        "wall", "floor", "ceiling", "window", "door", "picture", "monitor", "screen", "book",
        "sink", "sofa", "chair", "plant", "lamp", "stove", "oven", "refrigerator", "person",
        "man", "woman", "child", "boy", "girl", "face", "mirror",
    ]
    if any(d in norm for d in deny):
        return False
    return any(a in norm for a in allow)


def _is_hard_blocker_label(label: str) -> bool:
    norm = _normalize_label(label)
    hard = [
        "person", "man", "woman", "child", "boy", "girl", "people", "face", "head", "hand",
        "arm", "leg", "rider", "skier", "surfer",
    ]
    return any(h in norm for h in hard)


def _is_backgroundish_label(label: str) -> bool:
    norm = _normalize_label(label)
    bg = [
        "wall", "floor", "ceiling", "window", "door", "curtain", "blind", "rug", "carpet",
        "background", "sky", "road", "sidewalk", "grass", "tree", "mountain", "water",
    ]
    return any(b in norm for b in bg)


def _segment_scene_supports(scene_rgb: Image.Image, client: InferenceClient, scene_seg_model: str):
    outputs = client.image_segmentation(pil_to_png_bytes(scene_rgb.convert("RGB")), model=scene_seg_model)
    return list(outputs)


def _mask_to_binary(mask: Image.Image, size: tuple[int, int], threshold: int = 96) -> np.ndarray:
    m = mask.convert("L")
    if m.size != size:
        m = m.resize(size, Image.LANCZOS)
    return np.asarray(m, dtype=np.uint8) >= threshold


def _binary_to_mask(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(arr, 255, 0).astype(np.uint8), mode="L")


def _dilate_binary(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(bool, copy=False)
    arrf = arr.astype(np.float32)
    blurred = _box_blur(arrf, radius)
    return blurred > 0.0


def _erode_binary(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(bool, copy=False)
    inv = ~arr.astype(bool)
    return ~_dilate_binary(inv, radius)


def _connected_components(binary: np.ndarray) -> list[np.ndarray]:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    min_area = max(24, int(h * w * MIN_BLOB_AREA_FRAC))
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            pts = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                if cy > 0 and binary[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    q.append((cy - 1, cx))
                if cy + 1 < h and binary[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    q.append((cy + 1, cx))
                if cx > 0 and binary[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    q.append((cy, cx - 1))
                if cx + 1 < w and binary[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    q.append((cy, cx + 1))
            if len(pts) < min_area:
                continue
            comp = np.zeros((h, w), dtype=bool)
            ys, xs = zip(*pts)
            comp[np.asarray(ys), np.asarray(xs)] = True
            components.append(comp)
    components.sort(key=lambda c: int(c.sum()), reverse=True)
    return components


def _build_obstacle_masks(scene_size: tuple[int, int], seg_outputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    human = np.zeros((scene_size[1], scene_size[0]), dtype=bool)
    hard_other = np.zeros_like(human)
    soft = np.zeros_like(human)

    for output in seg_outputs:
        label = str(getattr(output, "label", ""))
        arr = _mask_to_binary(output.mask, scene_size)
        if not arr.any():
            continue
        if _is_hard_blocker_label(label):
            human |= arr
        elif _semantic_support_allowed(label):
            continue
        elif _is_backgroundish_label(label):
            continue
        else:
            soft |= arr

    hard = _dilate_binary(human | hard_other, HARD_BLOCKER_MARGIN_PX)
    soft = _dilate_binary(soft, SOFT_BLOCKER_MARGIN_PX)
    return human, hard, soft


def _local_free_space_score(scene_rgb: Image.Image, bbox: tuple[int, int, int, int], support_y: int, target_h: int) -> float:
    x0, _, x1, _ = bbox
    scene_w, scene_h = scene_rgb.size
    rx0 = max(0, min(scene_w, x0))
    rx1 = max(rx0 + 1, min(scene_w, x1))
    ry0 = max(0, support_y - max(12, int(target_h * 1.05)))
    ry1 = max(ry0 + 1, min(scene_h, support_y - 2))
    crop = scene_rgb.crop((rx0, ry0, rx1, ry1)).convert("L")
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    if arr.size == 0:
        return 0.0
    gy, gx = np.gradient(arr)
    energy = np.sqrt(gx * gx + gy * gy)
    clutter = float(np.mean(energy > max(0.12, float(np.quantile(energy, 0.76)))))
    return max(0.0, 1.0 - clutter * 1.2)


def _candidate_from_component(
    scene_rgb: Image.Image,
    depth: np.ndarray,
    comp: np.ndarray,
    label: str,
    object_label: str,
    obj_rgba: Image.Image,
    depth_pref: int,
    human_mask: np.ndarray,
    hard_mask: np.ndarray,
    soft_mask: np.ndarray,
    seg_score: float = 0.5,
) -> Optional[SupportCandidate]:
    ys, xs = np.where(comp)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    scene_w, scene_h = scene_rgb.size
    if x1 - x0 < max(18, int(scene_w * MIN_SUPPORT_SPAN_FRAC * 0.45)):
        return None

    bbox_h = max(1, y1 - y0)
    bbox_w = max(1, x1 - x0)
    support_y = int(np.quantile(ys, 0.18))
    lower_band_y0 = max(y0, support_y - max(2, int(bbox_h * 0.10)))
    lower_band_y1 = min(y1, support_y + max(3, int(bbox_h * 0.22)))
    band = comp[lower_band_y0:lower_band_y1]
    if band.size == 0:
        return None
    col_mass = band.mean(axis=0)
    spans = _iter_spans(
        col_mass > max(0.18, float(np.mean(col_mass) * 0.80)),
        max(10, int(scene_w * MIN_SUPPORT_SPAN_FRAC * 0.35)),
    )
    if spans:
        sx0, sx1 = max(spans, key=lambda s: s[1] - s[0])
        x0 = int(x0 + sx0)
        x1 = int(x0 + (sx1 - sx0 + 1))

    mask_img = _binary_to_mask(comp)
    center_x = int(round(float(np.mean(xs))))
    target_h = estimate_target_height(scene_rgb, None, object_label, obj_rgba)
    depth_value = float(depth[min(depth.shape[0] - 1, support_y), min(depth.shape[1] - 1, center_x)])
    occlusion_penalty = _compute_rgb_occlusion_penalty(scene_rgb, x0, x1, support_y, max(18, target_h))
    free_space = _local_free_space_score(scene_rgb, (x0, y0, x1, y1), support_y, target_h)

    box_area = float(max(1, bbox_w * bbox_h))
    comp_area = float(comp.sum())
    support_fraction = min(1.0, comp_area / box_area)
    top_visibility = min(1.0, max(0.0, (y1 - support_y) / float(max(1, y1 - y0))))

    comp_crop = comp[y0:y1, x0:x1]
    human_overlap = float(human_mask[y0:y1, x0:x1][comp_crop].mean()) if comp_crop.any() else 0.0
    hard_overlap = float(hard_mask[y0:y1, x0:x1][comp_crop].mean()) if comp_crop.any() else 0.0
    soft_overlap = float(soft_mask[y0:y1, x0:x1][comp_crop].mean()) if comp_crop.any() else 0.0

    score = (
        bbox_w * 0.8
        + free_space * 100.0
        + _depth_match_score(depth_value, depth_pref) * 75.0
        + top_visibility * 55.0
        + support_fraction * 80.0
        + seg_score * 24.0
        - occlusion_penalty * 80.0
        - human_overlap * 1000.0
        - hard_overlap * 400.0
        - soft_overlap * 120.0
    )

    return SupportCandidate(
        label=label,
        mask=mask_img,
        bbox=(x0, y0, x1, y1),
        support_y=support_y,
        anchor_x=center_x,
        local_span_width=max(1, x1 - x0),
        flatness=0.88,
        score=float(score),
        surface_type="semantic_support_blob",
        top_visibility=top_visibility,
        depth_value=depth_value,
        occlusion_penalty=occlusion_penalty,
        support_fraction=support_fraction,
        human_overlap_penalty=human_overlap,
        hard_overlap_penalty=hard_overlap,
        soft_overlap_penalty=soft_overlap,
    )


def _fallback_semantic_candidate(scene_rgb: Image.Image, depth: np.ndarray, depth_pref: int) -> SupportCandidate:
    scene_w, scene_h = scene_rgb.size
    support_y = int(scene_h * (0.58 + 0.20 * _depth_preference_to_target(depth_pref)))
    x0 = int(scene_w * 0.20)
    x1 = int(scene_w * 0.80)
    y0 = max(0, support_y - int(scene_h * 0.10))
    y1 = min(scene_h, support_y + int(scene_h * 0.08))
    mask = _make_rect_mask((scene_w, scene_h), (x0, y0, x1, y1))
    depth_value = float(depth[min(depth.shape[0]-1, support_y), min(depth.shape[1]-1, (x0+x1)//2)])
    return SupportCandidate(
        label="support",
        mask=mask,
        bbox=(x0, y0, x1, y1),
        support_y=support_y,
        anchor_x=(x0 + x1)//2,
        local_span_width=x1 - x0,
        flatness=0.7,
        score=100.0,
        surface_type="semantic_support_fallback",
        top_visibility=0.75,
        depth_value=depth_value,
        occlusion_penalty=0.2,
        support_fraction=1.0,
        human_overlap_penalty=0.0,
        hard_overlap_penalty=0.0,
        soft_overlap_penalty=0.0,
    )


def find_support_candidates(scene_rgb: Image.Image, depth: np.ndarray, depth_pref: int = 6, client: Optional[InferenceClient] = None, object_rgb: Optional[Image.Image] = None, object_label: str = "object", scene_seg_model: str = DEFAULT_SCENE_SEG_MODEL, obj_rgba: Optional[Image.Image] = None) -> tuple[list[SupportCandidate], dict]:
    del object_rgb
    if client is None or obj_rgba is None:
        return [_fallback_semantic_candidate(scene_rgb, depth, depth_pref)], {}

    try:
        seg_outputs = _segment_scene_supports(scene_rgb, client, scene_seg_model)
    except Exception:
        return [_fallback_semantic_candidate(scene_rgb, depth, depth_pref)], {}

    human_mask, hard_mask, soft_mask = _build_obstacle_masks(scene_rgb.size, seg_outputs)
    candidates: list[SupportCandidate] = []
    support_masks_debug: list[Image.Image] = []

    for output in seg_outputs:
        label = str(getattr(output, "label", "support"))
        if not _semantic_support_allowed(label):
            continue

        support_arr = _mask_to_binary(output.mask, scene_rgb.size)
        if not support_arr.any():
            continue

        support_arr = _erode_binary(support_arr, SUPPORT_EDGE_INSET_PX)
        support_masks_debug.append(_binary_to_mask(support_arr))

        allowed_arr = support_arr & (~hard_mask) & (~soft_mask)
        if not allowed_arr.any():
            allowed_arr = support_arr & (~hard_mask)
        if not allowed_arr.any():
            continue

        components = _connected_components(allowed_arr)[:MAX_COMPONENTS_PER_SUPPORT]
        for comp in components:
            cand = _candidate_from_component(
                scene_rgb=scene_rgb,
                depth=depth,
                comp=comp,
                label=label,
                object_label=object_label,
                obj_rgba=obj_rgba,
                depth_pref=depth_pref,
                human_mask=human_mask,
                hard_mask=hard_mask,
                soft_mask=soft_mask,
                seg_score=float(getattr(output, "score", 0.5)),
            )
            if cand is not None:
                candidates.append(cand)

    if not candidates:
        return [_fallback_semantic_candidate(scene_rgb, depth, depth_pref)], {
            "human_mask": _binary_to_mask(human_mask),
            "hard_mask": _binary_to_mask(hard_mask),
            "soft_mask": _binary_to_mask(soft_mask),
            "support_masks": support_masks_debug,
            "usable_masks": [],
        }

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:MAX_TOTAL_CANDIDATES]

    return candidates, {
        "human_mask": _binary_to_mask(human_mask),
        "hard_mask": _binary_to_mask(hard_mask),
        "soft_mask": _binary_to_mask(soft_mask),
        "support_masks": support_masks_debug,
        "usable_masks": [c.mask for c in candidates[:10]],
    }


def choose_support_candidate(candidates: list[SupportCandidate], scene_size: tuple[int, int], depth_pref: int = 6) -> Optional[SupportCandidate]:
    if not candidates:
        return None
    scene_w, _ = scene_size
    best = None
    best_score = -1e18
    for c in candidates:
        width_frac = c.local_span_width / float(max(1, scene_w))
        score = (
            c.score
            + c.flatness * 45.0
            + c.top_visibility * 45.0
            + width_frac * 35.0
            + c.support_fraction * 70.0
            - c.occlusion_penalty * 85.0
            - c.human_overlap_penalty * 1200.0
            - c.hard_overlap_penalty * 400.0
            - c.soft_overlap_penalty * 140.0
        )
        if score > best_score:
            best_score = score
            best = c
    return best


def compute_target_width(scene: Image.Image, obj_rgba: Image.Image, candidate: Optional[SupportCandidate], args: argparse.Namespace) -> int:
    scene_w, scene_h = scene.size
    obj_w, obj_h = obj_rgba.size
    obj_aspect = obj_w / float(max(1, obj_h))

    target_height = estimate_target_height(scene, candidate, args.object_label, obj_rgba)
    target = int(target_height * obj_aspect)

    if candidate is not None:
        span_cap_ratio = 0.30
        support_cap_ratio = 0.20
        span_cap = int(candidate.local_span_width * span_cap_ratio)
        support_cap = int((candidate.bbox[2] - candidate.bbox[0]) * support_cap_ratio)
        target = min(target, max(16, span_cap), max(16, support_cap))
        if candidate.flatness < 0.35:
            target = int(target * 0.9)
        if candidate.local_span_width < int(scene_w * 0.18):
            target = int(target * 0.9)
        target = int(target * (0.90 + 0.18 * _depth_match_score(candidate.depth_value, args.object_scene_depth)))
        if candidate.occlusion_penalty > 0.44:
            target = int(target * 0.9)
        if candidate.support_fraction < 0.75:
            target = int(target * 0.92)

    target = max(int(scene_w * MIN_OBJECT_WIDTH_FRAC), target)
    target = min(int(scene_w * MAX_OBJECT_WIDTH_FRAC), target)

    target = int(target * (0.90 + 0.22 * _depth_preference_to_target(args.object_scene_depth)))

    if args.scale is not None:
        target = max(16, int(target * args.scale))

    return max(16, target)


def _footprint_supported_fraction(candidate_mask: np.ndarray, left: int, top: int, width: int, height: int) -> float:
    h, w = candidate_mask.shape
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(w, left + width)
    y1 = min(h, top + height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = candidate_mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    sample_y0 = max(0, region.shape[0] - max(3, region.shape[0] // 5))
    support_band = region[sample_y0:]
    if support_band.size == 0:
        support_band = region
    return float(np.mean(support_band))


def _footprint_overlap_fraction(mask: np.ndarray, left: int, top: int, width: int, height: int) -> float:
    h, w = mask.shape
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(w, left + width)
    y1 = min(h, top + height)
    if x1 <= x0 or y1 <= y0:
        return 1.0
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 1.0
    return float(np.mean(region))


def _choose_blob_position(
    scene: Image.Image,
    candidate: SupportCandidate,
    target_width: int,
    target_height: int,
    hard_mask_img: Optional[Image.Image],
    soft_mask_img: Optional[Image.Image],
    human_mask_img: Optional[Image.Image],
    args: argparse.Namespace,
) -> tuple[int, int]:
    scene_w, scene_h = scene.size
    candidate_arr = np.asarray(candidate.mask.convert("L"), dtype=np.uint8) >= 96
    human_arr = np.asarray(human_mask_img.resize(scene.size, Image.LANCZOS), dtype=np.uint8) >= 96 if human_mask_img is not None else np.zeros((scene_h, scene_w), dtype=bool)
    hard_arr = np.asarray(hard_mask_img.resize(scene.size, Image.LANCZOS), dtype=np.uint8) >= 96 if hard_mask_img is not None else np.zeros((scene_h, scene_w), dtype=bool)
    soft_arr = np.asarray(soft_mask_img.resize(scene.size, Image.LANCZOS), dtype=np.uint8) >= 96 if soft_mask_img is not None else np.zeros((scene_h, scene_w), dtype=bool)

    x0, _, x1, _ = candidate.bbox
    span_margin = max(2, int(target_width * 0.06))
    min_center = max(target_width // 2, x0 + span_margin + target_width // 2)
    max_center = min(scene_w - target_width // 2, x1 - span_margin - target_width // 2)

    if min_center > max_center:
        center_x = max(target_width // 2, min(scene_w - target_width // 2, candidate.anchor_x))
        target_bottom = candidate.support_y + max(2, int(scene_h * 0.004))
        top = max(0, min(scene_h - target_height, target_bottom - target_height))
        left = max(0, min(scene_w - target_width, center_x - target_width // 2))
        return left, top

    centers = np.linspace(min_center, max_center, num=max(5, FOOTPRINT_SWEEP_STEPS)).astype(int)
    best = None
    best_score = -1e18
    lift = max(2, int(scene_h * 0.004))

    for center_x in centers.tolist():
        left = int(center_x - target_width / 2)
        target_bottom = candidate.support_y + lift
        top = int(target_bottom - target_height)
        left = max(0, min(scene_w - target_width, left))
        top = max(0, min(scene_h - target_height, top))

        supported = _footprint_supported_fraction(candidate_arr, left, top, target_width, target_height)
        human_overlap = _footprint_overlap_fraction(human_arr, left, top, target_width, target_height)
        hard_overlap = _footprint_overlap_fraction(hard_arr, left, top, target_width, target_height)
        soft_overlap = _footprint_overlap_fraction(soft_arr, left, top, target_width, target_height)

        if human_overlap > HUMAN_OVERLAP_REJECT_FRAC:
            continue
        if hard_overlap > HARD_OVERLAP_REJECT_FRAC:
            continue

        edge_margin = min(center_x - min_center, max_center - center_x) / float(max(1, max_center - min_center))
        score = (
            supported * 220.0
            + (1.0 - human_overlap) * 500.0
            + (1.0 - hard_overlap) * 140.0
            + (1.0 - soft_overlap) * 95.0
            + edge_margin * 18.0
            + _depth_match_score(candidate.depth_value, args.object_scene_depth) * 40.0
            - candidate.occlusion_penalty * 40.0
        )
        if supported < MIN_SUPPORTED_FRACTION:
            score -= (MIN_SUPPORTED_FRACTION - supported) * 260.0

        if score > best_score:
            best_score = score
            best = (left, top)

    if best is not None:
        return best

    # Graceful fallback: use anchor even if support fit was imperfect.
    center_x = max(target_width // 2, min(scene_w - target_width // 2, candidate.anchor_x))
    target_bottom = candidate.support_y + lift
    top = max(0, min(scene_h - target_height, target_bottom - target_height))
    left = max(0, min(scene_w - target_width, center_x - target_width // 2))
    return left, top


def compute_auto_placement(
    scene: Image.Image,
    obj_rgba: Image.Image,
    chosen_candidate: Optional[SupportCandidate],
    args: argparse.Namespace,
    hard_mask_img: Optional[Image.Image] = None,
    soft_mask_img: Optional[Image.Image] = None,
    human_mask_img: Optional[Image.Image] = None,
) -> Placement:
    scene_w, scene_h = scene.size
    obj_w, obj_h = obj_rgba.size

    aspect = obj_h / float(max(1, obj_w))
    target_width = compute_target_width(scene, obj_rgba, chosen_candidate, args)
    target_height = max(16, int(target_width * aspect))

    if chosen_candidate is None:
        center_x = int(scene_w * args.x) if args.x is not None else scene_w // 2
        target_bottom = int(scene_h * args.y) if args.y is not None else int(scene_h * 0.80)
        target_bottom = min(scene_h - 2, max(target_height + 2, target_bottom))
        left = int(center_x - target_width / 2)
        top = int(target_bottom - target_height)
        left = max(0, min(scene_w - target_width, left))
        top = max(0, min(scene_h - target_height, top))
        return Placement(left=left, top=top, width=target_width, height=target_height, support_label=None)

    if args.x is not None or args.y is not None:
        center_x = int(scene_w * args.x) if args.x is not None else chosen_candidate.anchor_x
        target_bottom = int(scene_h * args.y) if args.y is not None else (chosen_candidate.support_y + max(2, int(scene_h * 0.004)))
        target_bottom = min(scene_h - 2, max(target_height + 2, target_bottom))
        left = int(center_x - target_width / 2)
        top = int(target_bottom - target_height)
        left = max(0, min(scene_w - target_width, left))
        top = max(0, min(scene_h - target_height, top))
    else:
        left, top = _choose_blob_position(
            scene=scene,
            candidate=chosen_candidate,
            target_width=target_width,
            target_height=target_height,
            hard_mask_img=hard_mask_img,
            soft_mask_img=soft_mask_img,
            human_mask_img=human_mask_img,
            args=args,
        )

    support_label = chosen_candidate.label if chosen_candidate is not None else None
    return Placement(left=left, top=top, width=target_width, height=target_height, support_label=support_label)


def make_shadow(alpha: Image.Image, scene_size: tuple[int, int], placement: Placement) -> Image.Image:
    shadow = Image.new("RGBA", scene_size, (0, 0, 0, 0))
    shadow_alpha = alpha.resize((placement.width, placement.height), Image.LANCZOS)
    footprint_h = max(6, placement.height // 6)
    shadow_alpha = shadow_alpha.resize((placement.width, footprint_h), Image.LANCZOS)
    shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(radius=max(4, placement.width // 18)))

    shadow_layer = Image.new("RGBA", (placement.width, footprint_h), (0, 0, 0, 90))
    shadow_layer.putalpha(shadow_alpha)
    shadow.paste(shadow_layer, (placement.left, placement.top + placement.height - max(2, footprint_h // 3)), shadow_layer)
    return shadow


def precompose(scene: Image.Image, obj_rgba: Image.Image, placement: Placement) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    canvas = scene.copy().convert("RGBA")
    resized_obj = obj_rgba.resize((placement.width, placement.height), Image.LANCZOS)
    shadow = make_shadow(resized_obj.getchannel("A"), canvas.size, placement)
    with_shadow = Image.alpha_composite(canvas, shadow)
    object_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    object_layer.paste(resized_obj, (placement.left, placement.top), resized_obj)
    final = Image.alpha_composite(with_shadow, object_layer)
    return final, resized_obj, shadow, object_layer


def default_prompt(object_label: str, support_label: Optional[str]) -> str:
    support_text = f" on the {support_label}" if support_label else " in the scene"
    return (
        f"Place the {object_label} naturally{support_text}. "
        f"Preserve the {object_label} exactly, including logo, text, colors, and patterns. "
        f"Keep the rest of the scene unchanged. "
        f"Only refine local contact shadow, perspective fit, and edge blending around the inserted object."
    )


def refine_with_hf(precomp: Image.Image, client: InferenceClient, prompt: str, edit_model: str) -> Image.Image:
    png_bytes = pil_to_png_bytes(precomp.convert("RGB"))
    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

    try:
        return client.image_to_image(
            png_bytes,
            prompt=prompt,
            model=edit_model,
            image_urls=[data_url],
        )
    except KeyError as exc:
        raise RuntimeError(
            "HF fal-ai returned a response without an 'images' field. "
            "This is a known issue with Qwen/Qwen-Image-Edit on fal-ai via InferenceClient. "
            "Try passing image_urls=[data_url], switching provider, or using --skip-refine."
        ) from exc

def draw_support_candidates_preview(scene: Image.Image, candidates: list[SupportCandidate], chosen: Optional[SupportCandidate]) -> Image.Image:
    preview = scene.convert("RGBA").copy()
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, c in enumerate(candidates[:8]):
        is_chosen = chosen is not None and c.anchor_x == chosen.anchor_x and c.support_y == chosen.support_y and c.label == chosen.label and c.bbox == chosen.bbox
        base_color = (0, 220, 140, 0) if is_chosen else (90, 180, 255, 0)
        mask_rgba = Image.new("RGBA", preview.size, base_color)
        soft_mask = c.mask.filter(ImageFilter.GaussianBlur(radius=2))
        mask_rgba.putalpha(soft_mask.point(lambda p: min(110 if is_chosen else 72, p)))
        overlay = Image.alpha_composite(overlay, mask_rgba)

        x0, y0, x1, y1 = c.bbox
        color = (255, 120, 80, 255) if is_chosen else (180, 220, 255, 220)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3 if is_chosen else 2)
        draw.line((x0, c.support_y, x1, c.support_y), fill=(255, 220, 0, 255) if is_chosen else (210, 230, 255, 220), width=3)
        draw.ellipse((c.anchor_x - 5, c.support_y - 5, c.anchor_x + 5, c.support_y + 5), fill=color)
        label = f"{idx+1}:{c.label[:16]} s={int(c.score)} sf={c.support_fraction:.2f}"
        tx = max(0, min(preview.size[0] - 210, x0))
        ty = max(0, y0 - 18)
        draw.text((tx, ty), label, fill=color)

    return Image.alpha_composite(preview, overlay)


def draw_placement_preview(scene: Image.Image, placement: Placement, chosen_candidate: Optional[SupportCandidate]) -> Image.Image:
    preview = scene.convert("RGBA").copy()
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if chosen_candidate is not None:
        mask_rgba = Image.new("RGBA", preview.size, (0, 200, 255, 0))
        soft_mask = chosen_candidate.mask.filter(ImageFilter.GaussianBlur(radius=2))
        mask_rgba.putalpha(soft_mask.point(lambda p: min(110, p)))
        overlay = Image.alpha_composite(overlay, mask_rgba)
        x0, _, x1, _ = chosen_candidate.bbox
        draw.line((x0, chosen_candidate.support_y, x1, chosen_candidate.support_y), fill=(255, 220, 0, 255), width=3)
        draw.ellipse((chosen_candidate.anchor_x - 7, chosen_candidate.support_y - 7, chosen_candidate.anchor_x + 7, chosen_candidate.support_y + 7), fill=(255, 120, 80, 255))

    rect = [placement.left, placement.top, placement.left + placement.width, placement.top + placement.height]
    draw.rectangle(rect, outline=(255, 80, 80, 255), width=4)
    draw.line(
        (
            placement.left,
            placement.top + placement.height,
            placement.left + placement.width,
            placement.top + placement.height,
        ),
        fill=(255, 220, 0, 255),
        width=3,
    )
    return Image.alpha_composite(preview, overlay)


def compute_local_crop_box(placement: Placement, scene_size: tuple[int, int]) -> tuple[int, int, int, int]:
    scene_w, scene_h = scene_size
    pad_x = max(MIN_LOCAL_PADDING_PX, int(placement.width * 0.55))
    pad_y = max(MIN_LOCAL_PADDING_PX, int(placement.height * 0.55))

    pad_x = min(pad_x, int(scene_w * MAX_LOCAL_PADDING_FRACTION))
    pad_y = min(pad_y, int(scene_h * MAX_LOCAL_PADDING_FRACTION))

    x0 = max(0, placement.left - pad_x)
    y0 = max(0, placement.top - int(pad_y * 0.55))
    x1 = min(scene_w, placement.left + placement.width + pad_x)
    y1 = min(scene_h, placement.top + placement.height + int(pad_y * 0.95))
    return x0, y0, x1, y1


def scale_crop_for_edit(crop: Image.Image, max_side: int = INTERNAL_EDIT_MAX_SIDE) -> tuple[Image.Image, float]:
    w, h = crop.size
    longest = max(w, h)
    if longest <= max_side:
        return crop, 1.0
    scale = max_side / float(longest)
    resized = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return resized, scale


def make_local_edit_mask(crop_size: tuple[int, int], placement_in_crop: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = placement_in_crop
    mask = Image.new("L", crop_size, 0)
    draw = ImageDraw.Draw(mask)
    obj_w = x1 - x0
    obj_h = y1 - y0
    pad_x = max(16, int(obj_w * 0.35))
    pad_y_top = max(16, int(obj_h * 0.25))
    pad_y_bottom = max(18, int(obj_h * 0.5))
    draw.rounded_rectangle(
        [
            max(0, x0 - pad_x),
            max(0, y0 - pad_y_top),
            min(crop_size[0], x1 + pad_x),
            min(crop_size[1], y1 + pad_y_bottom),
        ],
        radius=max(12, min(crop_size) // 30),
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=LOCAL_MASK_BLUR_RADIUS))


def blend_local_edit(original_crop: Image.Image, refined_crop: Image.Image, mask: Image.Image) -> Image.Image:
    original_rgba = original_crop.convert("RGBA")
    refined_rgba = refined_crop.convert("RGBA")
    if refined_rgba.size != original_rgba.size:
        refined_rgba = refined_rgba.resize(original_rgba.size, Image.LANCZOS)
    return Image.composite(refined_rgba, original_rgba, mask)


def save_final_image(image: Image.Image, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, quality=100, subsampling=0, optimize=True)
    else:
        image.save(path)


def main() -> int:
    args = parse_args()
    args.object_scene_depth = max(1, min(10, int(args.object_scene_depth)))
    debug_run = DebugRun(args.output)
    client = build_client(args.provider)

    scene_full = load_rgba(args.scene)
    obj = load_rgba(args.object_image)
    scene_for_seg = resize_long_side(scene_full, INTERNAL_SEG_MAX_SCENE_SIDE)

    print(f"Run directory: {debug_run.run_dir}")
    print("[1/6] Extracting object matte...")
    obj_rgba = extract_object_rgba(obj, client, args.object_seg_model)
    object_mask = obj_rgba.getchannel("A")
    debug_run.save_image(scene_full, "01_scene_original.png", mode="RGB")
    debug_run.save_image(scene_for_seg, "02_scene_for_segmentation.png", mode="RGB")
    debug_run.save_image(obj, "03_object_input.png")
    debug_run.save_image(object_mask, "04_object_mask.png")
    debug_run.save_image(obj_rgba, "05_object_cutout.png")

    print("[2/6] Estimating monocular depth and finding support surfaces...")
    depth_small = estimate_monocular_depth(scene_for_seg.convert("RGB"))
    depth_preview_small = depth_to_preview(depth_small, scene_for_seg.size)
    debug_run.save_image(depth_preview_small, "06_depth_preview_scene_for_support.png")

    candidates_small, placement_debug_small = find_support_candidates(
        scene_for_seg.convert("RGB"),
        depth_small,
        depth_pref=args.object_scene_depth,
        client=client,
        object_rgb=obj_rgba.convert("RGB"),
        object_label=args.object_label,
        scene_seg_model=args.scene_seg_model,
        obj_rgba=obj_rgba,
    )

    candidates_full: list[SupportCandidate] = []
    scale_x = scene_full.width / float(scene_for_seg.width)
    scale_y = scene_full.height / float(scene_for_seg.height)

    def _scale_mask_to_full(mask_img: Optional[Image.Image]) -> Optional[Image.Image]:
        if mask_img is None:
            return None
        return mask_img.resize(scene_full.size, Image.LANCZOS)

    human_mask_full = _scale_mask_to_full(placement_debug_small.get("human_mask"))
    hard_mask_full = _scale_mask_to_full(placement_debug_small.get("hard_mask"))
    soft_mask_full = _scale_mask_to_full(placement_debug_small.get("soft_mask"))

    if human_mask_full is not None:
        debug_run.save_image(human_mask_full, "07_human_hard_block_mask.png")
    if hard_mask_full is not None:
        debug_run.save_image(hard_mask_full, "08_hard_block_mask.png")
    if soft_mask_full is not None:
        debug_run.save_image(soft_mask_full, "09_soft_block_mask.png")

    for idx, mask_img in enumerate(placement_debug_small.get("usable_masks", [])[:8], start=1):
        debug_run.save_image(_scale_mask_to_full(mask_img), f"10_usable_candidate_blob_{idx:02d}.png")

    for c in candidates_small:
        full_mask = c.mask.resize(scene_full.size, Image.LANCZOS)
        full_bbox = _mask_bbox(full_mask)
        if full_bbox is None:
            continue
        candidates_full.append(
            SupportCandidate(
                label=c.label,
                mask=full_mask,
                bbox=full_bbox,
                support_y=int(round(c.support_y * scale_y)),
                anchor_x=int(round(c.anchor_x * scale_x)),
                local_span_width=max(1, int(round(c.local_span_width * scale_x))),
                flatness=c.flatness,
                score=c.score,
                surface_type=c.surface_type,
                top_visibility=c.top_visibility,
                depth_value=c.depth_value,
                occlusion_penalty=c.occlusion_penalty,
                support_fraction=c.support_fraction,
                human_overlap_penalty=c.human_overlap_penalty,
                hard_overlap_penalty=c.hard_overlap_penalty,
                soft_overlap_penalty=c.soft_overlap_penalty,
            )
        )

    chosen_candidate = choose_support_candidate(candidates_full, scene_full.size, depth_pref=args.object_scene_depth)
    if chosen_candidate is not None:
        print(
            f"      Chosen support blob: {chosen_candidate.label} | "
            f"score={chosen_candidate.score:.1f} | span={chosen_candidate.local_span_width}px | "
            f"support_fraction={chosen_candidate.support_fraction:.2f}"
        )
    else:
        print("      No support surface found. Falling back to conservative center-lower placement.")

    candidate_preview = draw_support_candidates_preview(scene_full, candidates_full, chosen_candidate)
    debug_run.save_image(candidate_preview, "20_support_candidates_ranked.png")
    if chosen_candidate is not None:
        debug_run.save_image(chosen_candidate.mask, "21_chosen_support_mask_fullres.png")

    print("[3/6] Computing systematic placement and full-resolution precompositing...")
    placement = compute_auto_placement(
        scene_full,
        obj_rgba,
        chosen_candidate,
        args,
        hard_mask_img=hard_mask_full,
        soft_mask_img=soft_mask_full,
        human_mask_img=human_mask_full,
    )
    placement_preview = draw_placement_preview(scene_full, placement, chosen_candidate)
    precomp_full, resized_obj, shadow_full, object_layer_full = precompose(scene_full, obj_rgba, placement)
    debug_run.save_image(placement_preview, "22_placement_preview_fullres.png")
    debug_run.save_image(resized_obj, "23_object_resized.png")
    debug_run.save_image(shadow_full, "24_shadow_fullres.png")
    debug_run.save_image(object_layer_full, "25_object_layer_fullres.png")
    debug_run.save_image(precomp_full, "26_precomposite_fullres.png", mode="RGB")

    prompt = args.prompt.strip() if args.prompt and args.prompt.strip() else default_prompt(args.object_label, placement.support_label)
    crop_box = compute_local_crop_box(placement, scene_full.size)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    original_crop = scene_full.crop(crop_box)
    precomp_crop = precomp_full.crop(crop_box)
    placement_in_crop = (
        placement.left - crop_x0,
        placement.top - crop_y0,
        placement.left - crop_x0 + placement.width,
        placement.top - crop_y0 + placement.height,
    )
    local_mask_full = make_local_edit_mask(precomp_crop.size, placement_in_crop)
    precomp_crop_for_edit, crop_scale = scale_crop_for_edit(precomp_crop)
    debug_run.save_image(original_crop, "27_original_local_crop.png", mode="RGB")
    debug_run.save_image(precomp_crop, "28_precomposite_local_crop.png", mode="RGB")
    debug_run.save_image(local_mask_full, "29_local_blend_mask_fullres.png")
    debug_run.save_image(precomp_crop_for_edit, "30_crop_sent_to_edit_model.png", mode="RGB")

    if args.skip_refine:
        print("[4/6] Skipping HF refinement as requested.")
        refined_crop_full = precomp_crop.convert("RGB")
    else:
        print("[4/6] Refining only the local crop with HF image edit model...")
        print(f"      Model: {args.edit_model}")
        refined_crop_small = refine_with_hf(precomp_crop_for_edit, client, prompt, args.edit_model).convert("RGB")
        debug_run.save_image(refined_crop_small, "31_refined_local_crop_model_output.png", mode="RGB")
        if refined_crop_small.size != precomp_crop.size:
            refined_crop_full = refined_crop_small.resize(precomp_crop.size, Image.LANCZOS)
        else:
            refined_crop_full = refined_crop_small
        debug_run.save_image(refined_crop_full, "32_refined_local_crop_resized_to_fullres.png", mode="RGB")

    print("[5/6] Blending refined crop back into the original full-resolution scene...")
    blended_crop = blend_local_edit(original_crop, refined_crop_full, local_mask_full)
    final_full = scene_full.copy().convert("RGBA")
    final_full.paste(blended_crop, (crop_x0, crop_y0), blended_crop)
    debug_run.save_image(blended_crop, "33_blended_local_crop.png", mode="RGB")
    debug_run.save_image(final_full, "34_final_fullres_before_save.png", mode="RGB")

    metadata = {
        "scene": str(Path(args.scene).resolve()),
        "object_image": str(Path(args.object_image).resolve()),
        "requested_output": str(Path(args.output).resolve()),
        "final_output": str(debug_run.final_output.resolve()),
        "run_dir": str(debug_run.run_dir.resolve()),
        "prompt": prompt,
        "provider": args.provider,
        "edit_model": args.edit_model,
        "object_seg_model": args.object_seg_model,
        "scene_seg_model": args.scene_seg_model,
        "depth_model": DEFAULT_DEPTH_MODEL,
        "support_model": args.scene_seg_model,
        "placement": asdict(placement),
        "chosen_support": None if chosen_candidate is None else {
            "label": chosen_candidate.label,
            "bbox": list(chosen_candidate.bbox),
            "support_y": chosen_candidate.support_y,
            "anchor_x": chosen_candidate.anchor_x,
            "local_span_width": chosen_candidate.local_span_width,
            "flatness": chosen_candidate.flatness,
            "score": chosen_candidate.score,
            "surface_type": chosen_candidate.surface_type,
            "top_visibility": chosen_candidate.top_visibility,
            "depth_value": chosen_candidate.depth_value,
            "occlusion_penalty": chosen_candidate.occlusion_penalty,
            "support_fraction": chosen_candidate.support_fraction,
            "human_overlap_penalty": chosen_candidate.human_overlap_penalty,
            "hard_overlap_penalty": chosen_candidate.hard_overlap_penalty,
            "soft_overlap_penalty": chosen_candidate.soft_overlap_penalty,
        },
        "support_candidates": [
            {
                "label": c.label,
                "bbox": list(c.bbox),
                "support_y": c.support_y,
                "anchor_x": c.anchor_x,
                "local_span_width": c.local_span_width,
                "flatness": c.flatness,
                "score": c.score,
                "surface_type": c.surface_type,
                "top_visibility": c.top_visibility,
                "depth_value": c.depth_value,
                "occlusion_penalty": c.occlusion_penalty,
                "support_fraction": c.support_fraction,
                "human_overlap_penalty": c.human_overlap_penalty,
                "hard_overlap_penalty": c.hard_overlap_penalty,
                "soft_overlap_penalty": c.soft_overlap_penalty,
            }
            for c in candidates_full[:12]
        ],
        "crop_box": {"x0": crop_x0, "y0": crop_y0, "x1": crop_x1, "y1": crop_y1},
        "scene_original_size": {"width": scene_full.width, "height": scene_full.height},
        "scene_segmentation_size": {"width": scene_for_seg.width, "height": scene_for_seg.height},
        "local_crop_size": {"width": precomp_crop.width, "height": precomp_crop.height},
        "edit_input_size": {"width": precomp_crop_for_edit.width, "height": precomp_crop_for_edit.height},
        "crop_scale_sent_to_model": crop_scale,
        "skip_refine": args.skip_refine,
        "object_scene_depth": args.object_scene_depth,
        "internal_policy": {
            "depth_model": DEFAULT_DEPTH_MODEL,
            "support_model": args.scene_seg_model,
            "segmentation_max_side": INTERNAL_SEG_MAX_SCENE_SIDE,
            "edit_crop_max_side": INTERNAL_EDIT_MAX_SIDE,
            "min_local_padding_px": MIN_LOCAL_PADDING_PX,
            "max_local_padding_fraction": MAX_LOCAL_PADDING_FRACTION,
            "min_object_width_frac": MIN_OBJECT_WIDTH_FRAC,
            "default_object_width_frac": DEFAULT_OBJECT_WIDTH_FRAC,
            "max_object_width_frac": MAX_OBJECT_WIDTH_FRAC,
            "hard_blocker_margin_px": HARD_BLOCKER_MARGIN_PX,
            "soft_blocker_margin_px": SOFT_BLOCKER_MARGIN_PX,
            "support_edge_inset_px": SUPPORT_EDGE_INSET_PX,
        },
        "note": (
            "Support selection now subtracts human and scene-object blockers from support masks, "
            "splits remaining regions into smaller connected blobs, and chooses placement by sweeping "
            "the object footprint inside blob-shaped valid areas. Humans are treated as hard blockers."
        ),
    }

    print("[6/6] Saving output...")
    save_final_image(final_full, debug_run.final_output)
    debug_run.save_json(metadata)
    print(f"Saved final: {debug_run.final_output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
