import argparse
import base64
import io
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from dotenv import load_dotenv
from PIL import Image, ImageChops, ImageDraw, ImageFilter
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
# INTERNAL_EDIT_MAX_SIDE = 1408
INTERNAL_EDIT_MAX_SIDE = 2048
MIN_LOCAL_PADDING_PX = 80
MAX_LOCAL_PADDING_FRACTION = 0.48
LOCAL_MASK_BLUR_RADIUS = 12

# Conservative sizing policy. These are intentionally a bit smaller than before.
MIN_OBJECT_WIDTH_FRAC = 0.055
DEFAULT_OBJECT_WIDTH_FRAC = 0.085
MAX_OBJECT_WIDTH_FRAC = 0.135
ANCHOR_SEARCH_SIDE_FRAC = 0.22
MIN_SUPPORT_SPAN_FRAC = 0.10


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
    clearance_score: float = 0.0
    human_distance_score: float = 0.0
    center_penalty: float = 0.0
    forbidden_overlap: float = 0.0


_DEPTH_PIPELINE = None
_SCENE_SEG_PIPELINE = None


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


def _build_scene_seg_pipeline(scene_seg_model: str):
    global _SCENE_SEG_PIPELINE
    if _SCENE_SEG_PIPELINE is not None:
        return _SCENE_SEG_PIPELINE
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "Local scene segmentation requires transformers. Install it with: pip install transformers"
        ) from exc

    device = 0 if (torch is not None and torch.cuda.is_available()) else -1
    _SCENE_SEG_PIPELINE = pipeline("image-segmentation", model=scene_seg_model, device=device)
    return _SCENE_SEG_PIPELINE


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


def _normalize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return mask.convert("L")


def _mask_to_float(mask: Image.Image, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    if size is not None:
        mask = _normalize_mask(mask, size)
    return np.asarray(mask, dtype=np.float32) / 255.0


def _dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.convert("L")
    size = max(3, radius * 2 + 1)
    return mask.convert("L").filter(ImageFilter.MaxFilter(size=size))


def _erode_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.convert("L")
    size = max(3, radius * 2 + 1)
    return mask.convert("L").filter(ImageFilter.MinFilter(size=size))


def _is_person_label(label: str) -> bool:
    label = (label or "").strip().lower()
    return any(token in label for token in ["person", "people", "human", "man", "woman", "boy", "girl"])


def _is_support_like_label(label: str) -> bool:
    label = (label or "").strip().lower()
    support_terms = [
        "table", "desk", "counter", "countertop", "shelf", "bookcase", "cabinet",
        "stand", "dresser", "nightstand", "bench", "stool", "rack", "floor",
        "ground", "wall", "window sill", "windowsill", "ledge", "cupboard",
    ]
    return any(term in label for term in support_terms)


def _coerce_local_seg_outputs(raw_outputs, size: tuple[int, int]):
    class LocalSegOutput:
        def __init__(self, label: str, score: float, mask: Image.Image):
            self.label = label
            self.score = score
            self.mask = mask

    coerced = []
    for output in raw_outputs:
        label = str(output.get("label", "") or "")
        score = float(output.get("score", 0.0) or 0.0)
        mask = output.get("mask")
        if mask is None:
            continue
        if not isinstance(mask, Image.Image):
            try:
                mask = Image.fromarray(np.asarray(mask))
            except Exception:
                continue
        mask = _normalize_mask(mask, size)
        coerced.append(LocalSegOutput(label=label, score=score, mask=mask))
    return coerced


def build_scene_occupancy_masks(
    scene_rgb: Image.Image,
    client: Optional[InferenceClient],
    scene_seg_model: str,
) -> tuple[Image.Image, Image.Image]:
    del client
    size = scene_rgb.size
    empty = Image.new("L", size, 0)
    try:
        seg_pipe = _build_scene_seg_pipeline(scene_seg_model)
        raw_outputs = seg_pipe(scene_rgb.convert("RGB"))
        seg_outputs = _coerce_local_seg_outputs(raw_outputs, size)
    except Exception as exc:
        print(f"[masking] local scene segmentation failed: {exc}")
        return empty.copy(), empty.copy()

    print(f"[masking] got {len(seg_outputs)} segmentation outputs")
    for output in seg_outputs[:20]:
        print(
            "[masking]",
            "label=", getattr(output, "label", None),
            "score=", getattr(output, "score", None),
            "bbox=", _normalize_mask(output.mask, size).getbbox(),
        )

    person_mask = Image.new("L", size, 0)
    obstacle_mask = Image.new("L", size, 0)

    for output in seg_outputs:
        label = str(getattr(output, "label", "") or "")
        score = float(getattr(output, "score", 0.0) or 0.0)
        mask = _normalize_mask(output.mask, size)
        bbox = mask.getbbox()
        if bbox is None:
            continue
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if area <= 0:
            continue
        area_frac = area / float(max(1, size[0] * size[1]))
        if _is_person_label(label):
            pm = _dilate_mask(mask, max(14, int(min(size) * 0.02)))
            person_mask = ImageChops.lighter(person_mask, pm)
            obstacle_mask = ImageChops.lighter(obstacle_mask, pm)
            continue

        if _is_support_like_label(label):
            continue

        if score < 0.18 and area_frac < 0.003:
            continue

        dilated = _dilate_mask(mask, max(5, int(min(size) * 0.007)))
        obstacle_mask = ImageChops.lighter(obstacle_mask, dilated)

    return person_mask, obstacle_mask


def extract_object_rgba(obj_image: Image.Image, client: InferenceClient, seg_model: str) -> Image.Image:
    alpha = obj_image.getchannel("A")
    bbox = alpha_bbox(alpha)
    if bbox and ImageStatProxy.nonzero_fraction(alpha) > 0.01:
        return obj_image.crop(bbox)

    seg = client.image_segmentation(
        pil_to_png_bytes(obj_image.convert("RGB")),
        model=seg_model,
    )
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


DEFAULT_AFFORDANCE_MODEL = "CIDAS/clipseg-rd64-refined"
_CLIPSEG_PROCESSOR = None
_CLIPSEG_MODEL = None


def _build_affordance_pipeline():
    global _CLIPSEG_PROCESSOR, _CLIPSEG_MODEL
    if _CLIPSEG_PROCESSOR is not None and _CLIPSEG_MODEL is not None:
        return _CLIPSEG_PROCESSOR, _CLIPSEG_MODEL
    try:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    except Exception as exc:
        raise RuntimeError(
            "Direct affordance mask support detection requires transformers. Install it with: pip install transformers"
        ) from exc
    _CLIPSEG_PROCESSOR = CLIPSegProcessor.from_pretrained(DEFAULT_AFFORDANCE_MODEL)
    _CLIPSEG_MODEL = CLIPSegForImageSegmentation.from_pretrained(DEFAULT_AFFORDANCE_MODEL)
    if torch is not None:
        _CLIPSEG_MODEL.eval()
        if torch.cuda.is_available():
            _CLIPSEG_MODEL = _CLIPSEG_MODEL.to("cuda")
    return _CLIPSEG_PROCESSOR, _CLIPSEG_MODEL


def _build_affordance_prompt(object_label: str) -> str:
    return (
        f"the empty horizontal surface region in the scene where a {object_label} should be placed naturally, "
        "such as free tabletop, countertop, shelf top, or desk surface"
    )


def _largest_component_bbox(mask_arr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask_arr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _heatmap_to_candidate(scene_rgb: Image.Image, depth: np.ndarray, heatmap: np.ndarray, depth_pref: int) -> Optional[SupportCandidate]:
    scene_w, scene_h = scene_rgb.size
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap -= float(heatmap.min())
    denom = float(heatmap.max())
    if denom > 1e-6:
        heatmap /= denom
    support = heatmap.copy()
    support *= np.linspace(0.55, 1.0, scene_h, dtype=np.float32)[:, None]
    thresh = max(0.42, float(np.quantile(support, 0.92)))
    binary = support >= thresh
    bbox = _largest_component_bbox(binary)
    if bbox is None:
        thresh = max(0.30, float(np.quantile(support, 0.84)))
        bbox = _largest_component_bbox(support >= thresh)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    col_energy = support[y0:y1, x0:x1].mean(axis=0)
    col_thresh = max(float(np.mean(col_energy) * 0.85), 0.15)
    spans = _iter_spans(col_energy > col_thresh, max(16, int(scene_w * MIN_SUPPORT_SPAN_FRAC * 0.7)))
    if spans:
        sx0, sx1 = max(spans, key=lambda s: s[1] - s[0])
        x0 = x0 + sx0
        x1 = x0 + (sx1 - sx0 + 1)
    center_x = int(round((x0 + x1) / 2.0))
    support_rows = support[y0:y1, x0:x1].mean(axis=1)
    support_y = y0 + int(np.argmax(support_rows))
    probe_h = max(18, int(scene_h * 0.16))
    depth_value = float(depth[min(depth.shape[0]-1, support_y), min(depth.shape[1]-1, center_x)])
    confidence = float(np.mean(support[y0:y1, x0:x1]))
    occlusion_penalty = _compute_rgb_occlusion_penalty(scene_rgb, x0, x1, support_y, probe_h)
    mask_img = Image.fromarray(np.clip(support * 255.0, 0, 255).astype(np.uint8), mode="L")
    return SupportCandidate(
        label="affordance",
        mask=mask_img,
        bbox=(x0, y0, x1, y1),
        support_y=support_y,
        anchor_x=center_x,
        local_span_width=max(1, x1 - x0),
        flatness=0.9,
        score=float(confidence * 130.0 + (x1 - x0) * 0.08 - occlusion_penalty * 30.0 + _depth_match_score(depth_value, depth_pref) * 35.0),
        surface_type="placement_heatmap",
        top_visibility=0.95,
        depth_value=depth_value,
        occlusion_penalty=occlusion_penalty,
    )


def _connected_components(binary: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    binary = np.asarray(binary, dtype=bool)
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, int, int, int, np.ndarray]] = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords = []
            minx = maxx = x
            miny = maxy = y
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                for ny in range(max(0, cy - 1), min(h, cy + 2)):
                    for nx in range(max(0, cx - 1), min(w, cx + 2)):
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            mask = np.zeros((maxy - miny + 1, maxx - minx + 1), dtype=bool)
            for cy, cx in coords:
                mask[cy - miny, cx - minx] = True
            comps.append((minx, miny, maxx + 1, maxy + 1, mask))
    return comps


def _heatmap_to_candidates(
    scene_rgb: Image.Image,
    depth: np.ndarray,
    heatmap: np.ndarray,
    depth_pref: int,
    forbidden_mask: Optional[Image.Image] = None,
    person_mask: Optional[Image.Image] = None,
) -> list[SupportCandidate]:
    scene_w, scene_h = scene_rgb.size
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap -= float(heatmap.min())
    denom = float(heatmap.max())
    if denom > 1e-6:
        heatmap /= denom
    support = heatmap.copy()
    support *= np.linspace(0.45, 1.0, scene_h, dtype=np.float32)[:, None]

    forbidden_arr = _mask_to_float(forbidden_mask, scene_rgb.size) if forbidden_mask is not None else np.zeros((scene_h, scene_w), dtype=np.float32)
    person_arr = _mask_to_float(person_mask, scene_rgb.size) if person_mask is not None else np.zeros((scene_h, scene_w), dtype=np.float32)
    support = np.clip(support * (1.0 - np.clip(forbidden_arr * 1.15, 0.0, 0.97)), 0.0, 1.0)

    thresholds = []
    for q in (0.97, 0.94, 0.90, 0.86, 0.82):
        try:
            thresholds.append(max(0.18, float(np.quantile(support, q))))
        except Exception:
            pass
    thresholds = sorted(set(round(t, 4) for t in thresholds), reverse=True)

    min_comp_area = max(24, int(scene_w * scene_h * 0.0012))
    min_span = max(16, int(scene_w * MIN_SUPPORT_SPAN_FRAC * 0.45))
    candidates: list[SupportCandidate] = []
    seen = set()

    for thresh in thresholds:
        binary = support >= thresh
        comps = _connected_components(binary)
        for x0, y0, x1, y1, comp_mask in comps:
            if (x1 - x0) * (y1 - y0) < min_comp_area:
                continue
            comp_support = support[y0:y1, x0:x1].copy()
            comp_support *= comp_mask.astype(np.float32)
            if comp_support.size == 0:
                continue
            col_energy = comp_support.mean(axis=0)
            col_thresh = max(0.10, float(np.mean(col_energy) * 0.78))
            spans = _iter_spans(col_energy > col_thresh, min_span)
            if not spans:
                spans = [(0, comp_support.shape[1] - 1)]

            for sx0, sx1 in spans:
                px0 = x0 + sx0
                px1 = x0 + sx1 + 1
                if px1 - px0 < min_span:
                    continue
                local = support[y0:y1, px0:px1]
                if local.size == 0:
                    continue
                row_energy = local.mean(axis=1)
                if row_energy.size == 0:
                    continue
                support_y = y0 + int(np.argmax(row_energy))
                peak_cols = local[max(0, support_y - y0 - 3): min(local.shape[0], support_y - y0 + 4), :].mean(axis=0)
                anchor_x = px0 + int(np.argmax(peak_cols))
                depth_value = float(depth[min(depth.shape[0] - 1, support_y), min(depth.shape[1] - 1, anchor_x)])
                occlusion_penalty = _compute_rgb_occlusion_penalty(scene_rgb, px0, px1, support_y, max(18, int(scene_h * 0.14)))
                local_forbidden = forbidden_arr[y0:y1, px0:px1]
                local_person = person_arr[y0:y1, px0:px1]
                forbidden_overlap = float(np.mean(local_forbidden > 0.10)) if local_forbidden.size else 0.0
                human_overlap = float(np.mean(local_person > 0.10)) if local_person.size else 0.0
                if human_overlap > 0.0005 or forbidden_overlap > 0.22:
                    continue
                conf = float(np.mean(local))
                center_bias = 1.0 - min(1.0, abs((anchor_x / float(max(1, scene_w - 1))) - 0.5) / 0.5)
                human_distance = 1.0 - min(1.0, human_overlap * 6.0)
                score = (
                    conf * 135.0
                    + (px1 - px0) * 0.06
                    + _depth_match_score(depth_value, depth_pref) * 42.0
                    + human_distance * 14.0
                    - occlusion_penalty * 34.0
                    - forbidden_overlap * 90.0
                    - center_bias * 8.0
                )
                key = (int(anchor_x / 8), int(support_y / 8), int((px1 - px0) / 8))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SupportCandidate(
                        label='affordance',
                        mask=Image.fromarray(np.clip((support >= thresh).astype(np.uint8) * 255, 0, 255), mode='L'),
                        bbox=(px0, y0, px1, y1),
                        support_y=support_y,
                        anchor_x=anchor_x,
                        local_span_width=max(1, px1 - px0),
                        flatness=0.88,
                        score=float(score),
                        surface_type='placement_heatmap',
                        top_visibility=0.95,
                        depth_value=depth_value,
                        occlusion_penalty=occlusion_penalty,
                        clearance_score=max(0.0, 1.0 - forbidden_overlap),
                        human_distance_score=human_distance,
                        center_penalty=center_bias,
                        forbidden_overlap=forbidden_overlap,
                    )
                )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:36]


def _fallback_affordance_candidate(
    scene_rgb: Image.Image,
    depth: np.ndarray,
    depth_pref: int,
    forbidden_mask: Optional[Image.Image] = None,
    person_mask: Optional[Image.Image] = None,
) -> SupportCandidate:
    scene_w, scene_h = scene_rgb.size
    target_row = int(scene_h * (0.50 + 0.26 * _depth_preference_to_target(depth_pref)))
    forbidden_arr = _mask_to_float(forbidden_mask, scene_rgb.size) if forbidden_mask is not None else np.zeros((scene_h, scene_w), dtype=np.float32)
    person_arr = _mask_to_float(person_mask, scene_rgb.size) if person_mask is not None else np.zeros((scene_h, scene_w), dtype=np.float32)

    candidate_centers = [int(scene_w * frac) for frac in (0.14, 0.28, 0.42, 0.58, 0.72, 0.86)]
    band_w = max(32, int(scene_w * 0.16))
    best = None
    best_score = -1e18
    for cx in candidate_centers:
        x0 = max(0, cx - band_w // 2)
        x1 = min(scene_w, cx + band_w // 2)
        y0 = max(0, target_row - int(scene_h * 0.08))
        y1 = min(scene_h, target_row + int(scene_h * 0.07))
        if x1 <= x0 or y1 <= y0:
            continue
        local_forbidden = forbidden_arr[y0:y1, x0:x1]
        local_person = person_arr[y0:y1, x0:x1]
        forbidden_overlap = float(np.mean(local_forbidden > 0.10)) if local_forbidden.size else 0.0
        human_overlap = float(np.mean(local_person > 0.10)) if local_person.size else 0.0
        if human_overlap > 0.0005:
            continue
        depth_value = float(depth[min(depth.shape[0]-1, target_row), min(depth.shape[1]-1, cx)])
        center_bias = 1.0 - min(1.0, abs((cx / float(max(1, scene_w - 1))) - 0.5) / 0.5)
        score = _depth_match_score(depth_value, depth_pref) * 32.0 - forbidden_overlap * 120.0 - center_bias * 8.0
        if score > best_score:
            best_score = score
            best = (x0, y0, x1, y1, cx, depth_value, forbidden_overlap, center_bias)

    if best is None:
        cx = int(scene_w * 0.18)
        x0 = max(0, cx - band_w // 2)
        x1 = min(scene_w, cx + band_w // 2)
        y0 = max(0, target_row - int(scene_h * 0.08))
        y1 = min(scene_h, target_row + int(scene_h * 0.07))
        depth_value = float(depth[min(depth.shape[0]-1, target_row), min(depth.shape[1]-1, cx)])
        best = (x0, y0, x1, y1, cx, depth_value, 0.0, 0.0)

    x0, y0, x1, y1, cx, depth_value, forbidden_overlap, center_bias = best
    mask = _make_rect_mask((scene_w, scene_h), (x0, y0, x1, y1))
    return SupportCandidate(
        label='affordance',
        mask=mask,
        bbox=(x0, y0, x1, y1),
        support_y=target_row,
        anchor_x=cx,
        local_span_width=max(1, x1 - x0),
        flatness=0.7,
        score=100.0 - forbidden_overlap * 120.0 - center_bias * 8.0,
        surface_type='placement_heatmap',
        top_visibility=0.9,
        depth_value=depth_value,
        occlusion_penalty=0.2,
        clearance_score=max(0.0, 1.0 - forbidden_overlap),
        human_distance_score=1.0,
        center_penalty=center_bias,
        forbidden_overlap=forbidden_overlap,
    )


def find_support_candidates(
    scene_rgb: Image.Image,
    depth: np.ndarray,
    depth_pref: int = 6,
    client: Optional[InferenceClient] = None,
    object_rgb: Optional[Image.Image] = None,
    object_label: str = "object",
    scene_seg_model: str = DEFAULT_SCENE_SEG_MODEL,
) -> tuple[list[SupportCandidate], Image.Image, Image.Image]:
    del object_rgb
    person_mask, obstacle_mask = build_scene_occupancy_masks(scene_rgb, client, scene_seg_model)
    forbidden_mask = ImageChops.lighter(person_mask, obstacle_mask)
    try:
        processor, model = _build_affordance_pipeline()
        prompt = _build_affordance_prompt(object_label)
        inputs = processor(text=[prompt], images=[scene_rgb.convert("RGB")], padding=True, return_tensors="pt")
        if torch is not None and torch.cuda.is_available():
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad() if torch is not None else _nullcontext():
            outputs = model(**inputs)
        logits = outputs.logits[0]
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        heatmap = logits
        if heatmap.shape[::-1] != scene_rgb.size:
            heatmap = np.asarray(Image.fromarray(heatmap).resize(scene_rgb.size, Image.BILINEAR), dtype=np.float32)
        candidates = _heatmap_to_candidates(
            scene_rgb,
            depth,
            heatmap,
            depth_pref,
            forbidden_mask=forbidden_mask,
            person_mask=person_mask,
        )
        if candidates:
            return candidates, person_mask, forbidden_mask
    except Exception:
        pass
    return [_fallback_affordance_candidate(scene_rgb, depth, depth_pref, forbidden_mask=forbidden_mask, person_mask=person_mask)], person_mask, forbidden_mask


def choose_support_candidate(
    candidates: list[SupportCandidate],
    scene: Image.Image,
    obj_rgba: Image.Image,
    args: argparse.Namespace,
    forbidden_mask: Optional[Image.Image] = None,
    person_mask: Optional[Image.Image] = None,
) -> Optional[SupportCandidate]:
    if not candidates:
        return None

    scene_w, scene_h = scene.size
    forbidden_arr = _mask_to_float(forbidden_mask, scene.size) if forbidden_mask is not None else None
    person_arr = _mask_to_float(person_mask, scene.size) if person_mask is not None else None

    best = None
    best_score = -1e18
    for candidate in candidates:
        target_width = compute_target_width(scene, obj_rgba, candidate, args)
        target_height = max(16, int(target_width * (obj_rgba.height / float(max(1, obj_rgba.width)))))
        span_margin = max(3, int(target_width * 0.05))
        min_center = candidate.bbox[0] + span_margin + target_width // 2
        max_center = candidate.bbox[2] - span_margin - target_width // 2
        center_x = candidate.anchor_x
        if min_center <= max_center:
            center_x = max(min_center, min(max_center, center_x))

        lift = max(2, int(scene_h * 0.004))
        target_bottom = candidate.support_y + lift
        left = int(center_x - target_width / 2)
        top = int(target_bottom - target_height)
        left = max(0, min(scene_w - target_width, left))
        top = max(0, min(scene_h - target_height, top))
        right = min(scene_w, left + target_width)
        bottom = min(scene_h, top + target_height)

        fx0 = max(0, left - max(4, int(target_width * 0.14)))
        fx1 = min(scene_w, right + max(4, int(target_width * 0.14)))
        fy0 = max(0, top - max(4, int(target_height * 0.10)))
        fy1 = min(scene_h, bottom + max(6, int(target_height * 0.18)))

        forbidden_overlap = 0.0
        hard_person_overlap = 0.0
        if forbidden_arr is not None and fy1 > fy0 and fx1 > fx0:
            roi = forbidden_arr[fy0:fy1, fx0:fx1]
            forbidden_overlap = float(np.mean(roi > 0.10))
        if person_arr is not None and fy1 > fy0 and fx1 > fx0:
            proi = person_arr[fy0:fy1, fx0:fx1]
            hard_person_overlap = float(np.mean(proi > 0.10))
        if hard_person_overlap > 0.001:
            continue
        if forbidden_overlap > 0.08:
            continue

        support_band_y0 = max(0, bottom - max(4, int(scene_h * 0.008)))
        support_band_y1 = min(scene_h, bottom + max(4, int(scene_h * 0.010)))
        support_clearance = 1.0
        if forbidden_arr is not None and support_band_y1 > support_band_y0:
            sroi = forbidden_arr[support_band_y0:support_band_y1, left:right]
            if sroi.size > 0:
                support_clearance = max(0.0, 1.0 - float(np.mean(sroi > 0.10)) * 2.0)

        frame_margin = min(left, top, scene_w - right, scene_h - bottom)
        edge_score = min(1.0, frame_margin / float(max(12, scene_w * 0.08)))

        revised_score = (
            candidate.score
            + support_clearance * 28.0
            + edge_score * 10.0
            - forbidden_overlap * 160.0
            - candidate.center_penalty * 8.0
        )
        candidate.clearance_score = max(candidate.clearance_score, support_clearance)
        candidate.forbidden_overlap = max(candidate.forbidden_overlap, forbidden_overlap)
        if revised_score > best_score:
            best_score = revised_score
            best = candidate

    return best if best is not None else candidates[0]


class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def compute_target_width(scene: Image.Image, obj_rgba: Image.Image, candidate: Optional[SupportCandidate], args: argparse.Namespace) -> int:
    scene_w, scene_h = scene.size
    obj_w, obj_h = obj_rgba.size
    obj_aspect = obj_w / float(max(1, obj_h))

    target_height = estimate_target_height(scene, candidate, args.object_label, obj_rgba)
    target = int(target_height * obj_aspect)

    if candidate is not None:
        span_cap_ratio = 0.46 if candidate.surface_type == "top_surface" else 0.34
        support_cap_ratio = 0.28 if candidate.surface_type == "top_surface" else 0.22
        span_cap = int(candidate.local_span_width * span_cap_ratio)
        support_cap = int((candidate.bbox[2] - candidate.bbox[0]) * support_cap_ratio)
        target = min(target, span_cap, support_cap)
        if candidate.flatness < 0.35:
            target = int(target * 0.9)
        if candidate.local_span_width < int(scene_w * 0.18):
            target = int(target * 0.9)
        target *= 0.94 + 0.18 * _depth_match_score(candidate.depth_value, args.object_scene_depth)
        if candidate.occlusion_penalty > 0.44:
            target *= 0.9

    target = max(int(scene_w * MIN_OBJECT_WIDTH_FRAC), target)
    target = min(int(scene_w * MAX_OBJECT_WIDTH_FRAC), target)

    target = int(target * (0.90 + 0.22 * _depth_preference_to_target(args.object_scene_depth)))

    if args.scale is not None:
        target = max(16, int(target * args.scale))

    return max(16, target)


def compute_auto_placement(scene: Image.Image, obj_rgba: Image.Image, chosen_candidate: Optional[SupportCandidate], args: argparse.Namespace) -> Placement:
    scene_w, scene_h = scene.size
    obj_w, obj_h = obj_rgba.size

    aspect = obj_h / float(max(1, obj_w))
    target_width = compute_target_width(scene, obj_rgba, chosen_candidate, args)
    target_height = max(16, int(target_width * aspect))

    if args.x is not None:
        center_x = int(scene_w * args.x)
    elif chosen_candidate is not None:
        span_margin = max(4, int(target_width * 0.08))
        min_center = chosen_candidate.bbox[0] + span_margin + target_width // 2
        max_center = chosen_candidate.bbox[2] - span_margin - target_width // 2
        if min_center <= max_center:
            center_x = max(min_center, min(max_center, chosen_candidate.anchor_x))
        else:
            center_x = chosen_candidate.anchor_x
    else:
        center_x = scene_w // 2

    if args.y is not None:
        target_bottom = int(scene_h * args.y)
    elif chosen_candidate is not None:
        lift = max(2, int(scene_h * 0.004))
        if chosen_candidate.surface_type == "ledge_surface":
            lift = max(1, int(scene_h * 0.0025))
        target_bottom = chosen_candidate.support_y + lift
    else:
        target_bottom = int(scene_h * 0.80)

    target_bottom = min(scene_h - 2, max(target_height + 2, target_bottom))
    left = int(center_x - target_width / 2)
    top = int(target_bottom - target_height)
    left = max(0, min(scene_w - target_width, left))
    top = max(0, min(scene_h - target_height, top))

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
    return client.image_to_image(
        pil_to_png_bytes(precomp.convert("RGB")),
        prompt=prompt,
        model=edit_model,
    )


def draw_support_candidates_preview(scene: Image.Image, candidates: list[SupportCandidate], chosen: Optional[SupportCandidate]) -> Image.Image:
    preview = scene.convert("RGBA").copy()
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, c in enumerate(candidates[:5]):
        is_chosen = chosen is not None and c.anchor_x == chosen.anchor_x and c.support_y == chosen.support_y and c.label == chosen.label
        mask_rgba = Image.new("RGBA", preview.size, (0, 200, 255, 0) if is_chosen else (160, 100, 255, 0))
        soft_mask = c.mask.filter(ImageFilter.GaussianBlur(radius=2))
        mask_rgba.putalpha(soft_mask.point(lambda p: min(95 if is_chosen else 60, p)))
        overlay = Image.alpha_composite(overlay, mask_rgba)

        x0, y0, x1, y1 = c.bbox
        color = (255, 120, 80, 255) if is_chosen else (180, 120, 255, 220)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3 if is_chosen else 2)
        draw.line((x0, c.support_y, x1, c.support_y), fill=(255, 220, 0, 255) if is_chosen else (210, 180, 255, 200), width=3)
        draw.ellipse((c.anchor_x - 6, c.support_y - 6, c.anchor_x + 6, c.support_y + 6), fill=color)
        label = f"{idx+1}:{c.surface_type} s={int(c.score)} tv={c.top_visibility:.2f}"
        tx = max(0, min(preview.size[0] - 180, x0))
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
    pad_x = max(MIN_LOCAL_PADDING_PX, int(placement.width * 1.30))
    pad_y = max(MIN_LOCAL_PADDING_PX, int(placement.height * 1.30))

    pad_x = min(pad_x, int(scene_w * MAX_LOCAL_PADDING_FRACTION))
    pad_y = min(pad_y, int(scene_h * MAX_LOCAL_PADDING_FRACTION))

    x0 = max(0, placement.left - pad_x)
    y0 = max(0, placement.top - int(pad_y * 1.10))
    x1 = min(scene_w, placement.left + placement.width + pad_x)
    y1 = min(scene_h, placement.top + placement.height + int(pad_y * 1.65))
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

    candidates_small, person_mask_small, forbidden_mask_small = find_support_candidates(
        scene_for_seg.convert("RGB"),
        depth_small,
        depth_pref=args.object_scene_depth,
        client=client,
        object_label=args.object_label,
        scene_seg_model=args.scene_seg_model,
    )
    candidates_full: list[SupportCandidate] = []
    person_mask_full = person_mask_small.resize(scene_full.size, Image.NEAREST)
    forbidden_mask_full = forbidden_mask_small.resize(scene_full.size, Image.NEAREST)
    scale_x = scene_full.width / float(scene_for_seg.width)
    scale_y = scene_full.height / float(scene_for_seg.height)
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
                local_span_width=int(round(c.local_span_width * scale_x)),
                flatness=c.flatness,
                score=c.score,
                surface_type=c.surface_type,
                top_visibility=c.top_visibility,
                depth_value=c.depth_value,
                occlusion_penalty=c.occlusion_penalty,
                clearance_score=c.clearance_score,
                human_distance_score=c.human_distance_score,
                center_penalty=c.center_penalty,
                forbidden_overlap=c.forbidden_overlap,
            )
        )
    chosen_candidate = choose_support_candidate(
        candidates_full,
        scene_full,
        obj_rgba,
        args,
        forbidden_mask=forbidden_mask_full,
        person_mask=person_mask_full,
    )
    if chosen_candidate is not None:
        print(f"      Chosen support: {chosen_candidate.label} | score={chosen_candidate.score:.1f} | span={chosen_candidate.local_span_width}px | candidates={len(candidates_full)}")
    else:
        print("      No support surface found. Falling back to conservative center-lower placement.")

    candidate_preview = draw_support_candidates_preview(scene_full, candidates_full, chosen_candidate)
    debug_run.save_image(candidate_preview, "06_support_candidates_ranked.png")
    debug_run.save_image(person_mask_full, "07_people_forbidden_mask_fullres.png")
    debug_run.save_image(forbidden_mask_full, "08_forbidden_occupancy_mask_fullres.png")
    if chosen_candidate is not None:
        debug_run.save_image(chosen_candidate.mask, "09_chosen_support_mask_fullres.png")

    print("[3/6] Computing systematic placement and full-resolution precompositing...")
    placement = compute_auto_placement(scene_full, obj_rgba, chosen_candidate, args)
    placement_preview = draw_placement_preview(scene_full, placement, chosen_candidate)
    precomp_full, resized_obj, shadow_full, object_layer_full = precompose(scene_full, obj_rgba, placement)
    debug_run.save_image(placement_preview, "10_placement_preview_fullres.png")
    debug_run.save_image(resized_obj, "11_object_resized.png")
    debug_run.save_image(shadow_full, "12_shadow_fullres.png")
    debug_run.save_image(object_layer_full, "13_object_layer_fullres.png")
    debug_run.save_image(precomp_full, "14_precomposite_fullres.png", mode="RGB")

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
    debug_run.save_image(original_crop, "15_original_local_crop.png", mode="RGB")
    debug_run.save_image(precomp_crop, "16_precomposite_local_crop.png", mode="RGB")
    debug_run.save_image(local_mask_full, "17_local_blend_mask_fullres.png")
    debug_run.save_image(precomp_crop_for_edit, "18_crop_sent_to_edit_model.png", mode="RGB")

    if args.skip_refine:
        print("[4/6] Skipping HF refinement as requested.")
        refined_crop_full = precomp_crop.convert("RGB")
    else:
        print("[4/6] Refining only the local crop with HF image edit model...")
        print(f"      Model: {args.edit_model}")
        refined_crop_small = refine_with_hf(precomp_crop_for_edit, client, prompt, args.edit_model).convert("RGB")
        debug_run.save_image(refined_crop_small, "19_refined_local_crop_model_output.png", mode="RGB")
        if refined_crop_small.size != precomp_crop.size:
            refined_crop_full = refined_crop_small.resize(precomp_crop.size, Image.LANCZOS)
        else:
            refined_crop_full = refined_crop_small
        debug_run.save_image(refined_crop_full, "20_refined_local_crop_resized_to_fullres.png", mode="RGB")

    print("[5/6] Blending refined crop back into the original full-resolution scene...")
    blended_crop = blend_local_edit(original_crop, refined_crop_full, local_mask_full)
    final_full = scene_full.copy().convert("RGBA")
    final_full.paste(blended_crop, (crop_x0, crop_y0), blended_crop)
    debug_run.save_image(blended_crop, "21_blended_local_crop.png", mode="RGB")
    debug_run.save_image(final_full, "22_final_fullres_before_save.png", mode="RGB")

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
        "support_model": DEFAULT_AFFORDANCE_MODEL,
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
            "clearance_score": chosen_candidate.clearance_score,
            "human_distance_score": chosen_candidate.human_distance_score,
            "center_penalty": chosen_candidate.center_penalty,
            "forbidden_overlap": chosen_candidate.forbidden_overlap,
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
                "clearance_score": c.clearance_score,
                "human_distance_score": c.human_distance_score,
                "center_penalty": c.center_penalty,
                "forbidden_overlap": c.forbidden_overlap,
            }
            for c in candidates_full[:8]
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
            "support_model": DEFAULT_AFFORDANCE_MODEL,
            "segmentation_max_side": INTERNAL_SEG_MAX_SCENE_SIDE,
            "edit_crop_max_side": INTERNAL_EDIT_MAX_SIDE,
            "min_local_padding_px": MIN_LOCAL_PADDING_PX,
            "max_local_padding_fraction": MAX_LOCAL_PADDING_FRACTION,
            "min_object_width_frac": MIN_OBJECT_WIDTH_FRAC,
            "default_object_width_frac": DEFAULT_OBJECT_WIDTH_FRAC,
            "max_object_width_frac": MAX_OBJECT_WIDTH_FRAC,
        },
        "note": "Support selection now uses granular affordance candidates, hard person avoidance, and scene occupancy filtering to prefer smaller valid background placements away from clutter and humans.",
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
