from __future__ import annotations

import argparse
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFilter
from dotenv import load_dotenv
from rembg import remove
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, pipeline

try:
    from diffusers import FluxFillPipeline  # type: ignore
    _FLUX_FILL_AVAILABLE = True
except Exception:
    FluxFillPipeline = None  # type: ignore
    _FLUX_FILL_AVAILABLE = False

try:
    from transformers import Sam2Model, Sam2Processor  # type: ignore
    _SAM_AVAILABLE = True
except Exception:
    Sam2Model = None  # type: ignore
    Sam2Processor = None  # type: ignore
    _SAM_AVAILABLE = False

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "models": {
        "detector_id": "IDEA-Research/grounding-dino-tiny",
        "sam_id": "facebook/sam2.1-hiera-tiny",
        "depth_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "fill_id": "black-forest-labs/FLUX.1-Fill-dev",
    },
    "runtime": {
        "max_side": 1024,
        "dtype": "bf16",
    },
    "detection": {
        "surface_labels": ["countertop", "kitchen counter", "table", "desk", "island", "shelf"],
        "obstacle_labels": [
            "bowl", "plate", "cup", "mug", "glass", "bottle", "jar", "vase", "plant",
            "phone", "book", "box", "basket", "fruit", "banana", "apple", "orange", "lemon",
            "sink", "stove", "toaster", "kettle", "microwave", "coffee maker",
        ],
        "object_threshold": 0.28,
        "object_text_threshold": 0.20,
        "surface_threshold": 0.26,
        "surface_text_threshold": 0.20,
        "obstacle_threshold": 0.25,
        "obstacle_text_threshold": 0.20,
    },
    "extraction": {
        "use_rembg": True,
        "rembg_alpha_threshold": 20,
        "use_sam": True,
        "sam_box_padding": 8,
        "matte_blur": 0.8,
        "keep_largest_component": True,
        "grabcut_refine": True,
        "grabcut_iters": 2,
    },
    "placement": {
        "max_supports": 6,
        "edge_margin_ratio": 0.06,
        "candidate_step_divisor": 6,
        "min_object_height_ratio": 0.07,
        "max_object_height_ratio": 0.33,
        "prefer_center_weight": 0.25,
        "avoid_overlap_weight": 5.5,
        "depth_std_weight": 1.8,
        "support_band_weight": 1.4,
        "attempt_index": 0,
    },
    "paper_style": {
        "hf_radius_ratio": 0.08,
        "conditioning_ref_size": 256,
        "conditioning_gap": 16,
        "mask_feather_px": 7,
        "mask_expand_px": 14,
        "multi_seed": [0, 1, 2],
        "num_steps": 32,
        "guidance_scale": 17.0,
        "max_sequence_length": 256,
        "detail_refine": True,
        "detail_refine_steps": 18,
        "detail_refine_guidance": 12.0,
    },
    "output": {
        "save_conditioning_canvas": False,
        "save_mask": False,
    },
}

# -----------------------------------------------------------------------------
# Lazy model caches
# -----------------------------------------------------------------------------
_DET_PROCESSOR = None
_DET_MODEL = None
_SAM_PROCESSOR = None
_SAM_MODEL = None
_DEPTH_PIPE = None
_FILL_PIPE = None


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

    def as_int(self) -> Tuple[int, int, int, int]:
        return int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1))


@dataclass
class ExtractedObject:
    rgba: Image.Image
    mask: Image.Image
    label: str


@dataclass
class PlacementCandidate:
    box: BoundingBox
    support: BoundingBox
    score: float
    reason: str


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
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
    p = Path(path)
    if not p.exists():
        return DEFAULT_CONFIG
    with p.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_torch_dtype(cfg: dict, device: torch.device):
    dtype_name = str(cfg.get("runtime", {}).get("dtype", "bf16")).lower()
    if device.type != "cuda":
        return torch.float32
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


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
    if not _SAM_AVAILABLE or not cfg["extraction"].get("use_sam", True):
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
        raise RuntimeError("FluxFillPipeline is not available in this environment.")
    if _FILL_PIPE is None:
        dtype = get_torch_dtype(cfg, device)
        _FILL_PIPE = FluxFillPipeline.from_pretrained(
            cfg["models"]["fill_id"],
            torch_dtype=dtype,
            token=HF_TOKEN,
        )
        if device.type == "cuda":
            if hasattr(_FILL_PIPE, "enable_model_cpu_offload"):
                _FILL_PIPE.enable_model_cpu_offload()
            else:
                _FILL_PIPE.to(device)
        if hasattr(_FILL_PIPE, "vae") and hasattr(_FILL_PIPE.vae, "enable_slicing"):
            _FILL_PIPE.vae.enable_slicing()
        if hasattr(_FILL_PIPE, "vae") and hasattr(_FILL_PIPE.vae, "enable_tiling"):
            _FILL_PIPE.vae.enable_tiling()
    return _FILL_PIPE


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def maybe_resize(image: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale >= 0.999:
        return image, 1.0
    nw = max(32, int(round(w * scale / 8) * 8))
    nh = max(32, int(round(h * scale / 8) * 8))
    return image.resize((nw, nh), Image.LANCZOS), scale


def scale_box(box: BoundingBox, inv_scale: float) -> BoundingBox:
    return BoundingBox(
        x0=box.x0 * inv_scale,
        y0=box.y0 * inv_scale,
        x1=box.x1 * inv_scale,
        y1=box.y1 * inv_scale,
        score=box.score,
        label=box.label,
    )


def detect_boxes(
    image: Image.Image,
    labels: Iterable[str],
    device: torch.device,
    cfg: dict,
    threshold: float,
    text_threshold: float,
) -> List[BoundingBox]:
    labels = [x.strip() for x in labels if x and x.strip()]
    if not labels:
        return []
    proc, model = get_detector(device, cfg)
    resized, scale = maybe_resize(image, int(cfg["runtime"].get("max_side", 1024)))
    inputs = proc(images=resized, text=[labels], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[resized.size[::-1]],
    )[0]
    inv = 1.0 / scale
    out: List[BoundingBox] = []
    for i, box in enumerate(result.get("boxes", [])):
        score = float(result.get("scores", [])[i].item()) if i < len(result.get("scores", [])) else 1.0
        label = str(result.get("labels", [])[i]) if i < len(result.get("labels", [])) else ""
        b = BoundingBox(*[float(v) for v in box.tolist()], score=score, label=label)
        out.append(scale_box(b, inv).clamp(*image.size))
    return out


def choose_best_box(boxes: List[BoundingBox]) -> Optional[BoundingBox]:
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b.score, b.area()))


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == best, 255, 0).astype(np.uint8)


def refine_mask_with_sam(image: Image.Image, box: BoundingBox, device: torch.device, cfg: dict) -> Optional[Image.Image]:
    proc, model = get_sam(device, cfg)
    if proc is None or model is None:
        return None
    pad = int(cfg["extraction"].get("sam_box_padding", 8))
    input_boxes = [[[max(0.0, box.x0 - pad), max(0.0, box.y0 - pad), min(image.width, box.x1 + pad), min(image.height, box.y1 + pad)]]]
    try:
        sam_inputs = proc(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**sam_inputs)
        masks = proc.post_process_masks(outputs.pred_masks.cpu(), sam_inputs["original_sizes"], sam_inputs["reshaped_input_sizes"])
        if not masks or len(masks[0]) == 0:
            return None
        best_mask = None
        best_score = -1.0
        for m in masks[0][0]:
            mm = (m.numpy() > 0).astype(np.uint8) * 255
            ys, xs = np.where(mm > 0)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            area = float((mm > 0).sum())
            bbox_area = float(max(1, (x1 - x0 + 1) * (y1 - y0 + 1)))
            fill = area / bbox_area
            border = float((mm[0, :] > 0).sum() + (mm[-1, :] > 0).sum() + (mm[:, 0] > 0).sum() + (mm[:, -1] > 0).sum())
            score = area * fill - border * 15.0
            if 0.05 <= fill <= 0.98 and score > best_score:
                best_score = score
                best_mask = mm
        if best_mask is None:
            return None
        return Image.fromarray(best_mask, mode="L")
    except Exception:
        return None


def mask_from_rembg(image: Image.Image, alpha_threshold: int) -> Optional[Image.Image]:
    try:
        buf = io.BytesIO()
        image.convert("RGBA").save(buf, format="PNG")
        out = remove(buf.getvalue())
        rgba = Image.open(io.BytesIO(out)).convert("RGBA")
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        alpha = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(alpha, mode="L")
    except Exception:
        return None


def refine_mask_grabcut(image: Image.Image, mask_img: Image.Image, iters: int) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    mask = np.array(mask_img, dtype=np.uint8)
    gc = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc[mask > 220] = cv2.GC_PR_FGD
    fg = cv2.erode(((mask > 220).astype(np.uint8) * 255), np.ones((5, 5), np.uint8), iterations=1)
    bg = cv2.dilate(((mask > 0).astype(np.uint8) * 255), np.ones((9, 9), np.uint8), iterations=1) == 0
    gc[fg > 0] = cv2.GC_FGD
    gc[bg] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def crop_to_alpha(rgba: Image.Image) -> Tuple[Image.Image, Image.Image]:
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return rgba, alpha
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - 2)
    y0 = max(0, y0 - 2)
    x1 = min(rgba.width, x1 + 2)
    y1 = min(rgba.height, y1 + 2)
    cropped = rgba.crop((x0, y0, x1, y1))
    return cropped, cropped.getchannel("A")


def extract_reference_object(image: Image.Image, label: str, device: torch.device, cfg: dict) -> ExtractedObject:
    boxes = detect_boxes(
        image,
        [label],
        device,
        cfg,
        cfg["detection"]["object_threshold"],
        cfg["detection"]["object_text_threshold"],
    )
    best = choose_best_box(boxes)
    if best is None:
        raise RuntimeError(f"Could not detect '{label}' in the reference image.")

    rembg_mask = None
    if cfg["extraction"].get("use_rembg", True):
        rembg_mask = mask_from_rembg(image, int(cfg["extraction"].get("rembg_alpha_threshold", 20)))
    sam_mask = refine_mask_with_sam(image, best, device, cfg)

    if rembg_mask is not None and sam_mask is not None:
        a = np.array(rembg_mask, dtype=np.uint8) > 0
        b = np.array(sam_mask, dtype=np.uint8) > 0
        inter = a & b
        use = inter if inter.sum() > 0.55 * min(max(1, a.sum()), max(1, b.sum())) else (a | b)
        mask = Image.fromarray((use.astype(np.uint8) * 255), mode="L")
    elif rembg_mask is not None:
        mask = rembg_mask
    elif sam_mask is not None:
        mask = sam_mask
    else:
        # fallback to detected box
        arr = np.zeros((image.height, image.width), dtype=np.uint8)
        x0, y0, x1, y1 = best.as_int()
        arr[y0:y1, x0:x1] = 255
        mask = Image.fromarray(arr, mode="L")

    if cfg["extraction"].get("grabcut_refine", True):
        try:
            mask = refine_mask_grabcut(image, mask, int(cfg["extraction"].get("grabcut_iters", 2)))
        except Exception:
            pass

    mask_np = np.array(mask, dtype=np.uint8)
    if cfg["extraction"].get("keep_largest_component", True):
        mask_np = largest_component(mask_np)
    blur = float(cfg["extraction"].get("matte_blur", 0.8))
    if blur > 0:
        mask_np = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=blur, sigmaY=blur)
    rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    rgba[:, :, 3] = mask_np
    rgba[mask_np == 0, :3] = 0
    cropped_rgba, cropped_mask = crop_to_alpha(Image.fromarray(rgba, mode="RGBA"))
    return ExtractedObject(rgba=cropped_rgba, mask=cropped_mask, label=label)


# -----------------------------------------------------------------------------
# Depth and placement helpers
# -----------------------------------------------------------------------------
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
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    depth = (depth - dmin) / (dmax - dmin)
    # try to orient so bottom is generally nearer / larger depth value
    top = float(np.median(depth[: max(1, depth.shape[0] // 5)]))
    bottom = float(np.median(depth[-max(1, depth.shape[0] // 5):]))
    if bottom < top:
        depth = 1.0 - depth
    return depth


def filter_supports(boxes: List[BoundingBox], scene_size: Tuple[int, int]) -> List[BoundingBox]:
    w, h = scene_size
    out: List[BoundingBox] = []
    for b in boxes:
        if b.area() < w * h * 0.02:
            continue
        if b.width() < w * 0.22:
            continue
        if b.height() < h * 0.025:
            continue
        if b.height() > h * 0.24:
            continue
        if b.y0 > h * 0.82:
            continue
        out.append(b)
    out.sort(key=lambda x: (x.score, x.area()), reverse=True)
    return out


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.area() + b.area() - inter
    return 0.0 if union <= 0 else inter / union


def local_depth_std(depth: np.ndarray, box: BoundingBox) -> float:
    h, w = depth.shape
    x0, y0, x1, y1 = box.as_int()
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    patch = depth[y0:y1, x0:x1]
    if patch.size == 0:
        return 1.0
    return float(np.std(patch))


def choose_object_size(scene: Image.Image, obj: ExtractedObject, support: BoundingBox, depth: np.ndarray, cfg: dict) -> Tuple[int, int]:
    sw, sh = scene.size
    aspect = obj.rgba.width / max(1.0, obj.rgba.height)
    support_ratio = np.clip(support.width() / max(1.0, sw), 0.18, 0.8)
    support_target_h = sh * (0.10 + 0.20 * support_ratio)
    y0, y1 = max(0, int(round(support.y0))), min(sh, int(round(support.y1)))
    depth_val = float(np.median(depth[y0:y1, max(0, int(round(support.x0))): min(sw, int(round(support.x1)))])) if y1 > y0 else 0.5
    perspective_scale = float(np.interp(depth_val, [0.0, 1.0], [0.76, 1.18]))
    target_h = support_target_h * perspective_scale
    target_h = int(round(np.clip(target_h, sh * cfg["placement"]["min_object_height_ratio"], sh * cfg["placement"]["max_object_height_ratio"])))
    target_w = int(round(target_h * aspect))
    return max(24, target_w), max(24, target_h)


def propose_placements(
    scene: Image.Image,
    obj: ExtractedObject,
    supports: List[BoundingBox],
    obstacles: List[BoundingBox],
    depth: np.ndarray,
    cfg: dict,
) -> List[PlacementCandidate]:
    sw, sh = scene.size
    out: List[PlacementCandidate] = []
    max_supports = int(cfg["placement"].get("max_supports", 6))
    for support in supports[:max_supports]:
        target_w, target_h = choose_object_size(scene, obj, support, depth, cfg)
        edge_margin = int(round(support.width() * float(cfg["placement"].get("edge_margin_ratio", 0.06))))
        usable_x0 = max(0, int(round(support.x0)) + edge_margin)
        usable_x1 = min(sw, int(round(support.x1)) - edge_margin)
        foot_y = int(round(support.y0 + support.height() * 0.18))
        step = max(8, target_w // max(2, int(cfg["placement"].get("candidate_step_divisor", 6))))
        xs = list(range(usable_x0, max(usable_x0 + 1, usable_x1 - target_w + 1), step))
        if usable_x1 - target_w >= usable_x0:
            xs.append(usable_x1 - target_w)
        xs = sorted(set(xs))

        for x in xs:
            y = foot_y - target_h
            candidate = BoundingBox(x, y, x + target_w, y + target_h)
            if candidate.x0 < 0 or candidate.y0 < 0 or candidate.x1 > sw or candidate.y1 > sh:
                continue
            max_overlap = max([iou(candidate, o) for o in obstacles], default=0.0)
            center_offset = abs(candidate.centre()[0] - support.centre()[0]) / max(1.0, support.width())
            dst = local_depth_std(depth, BoundingBox(candidate.x0 + candidate.width() * 0.12, candidate.y1 - candidate.height() * 0.22, candidate.x1 - candidate.width() * 0.12, candidate.y1))
            band_pref = abs((candidate.y1 - support.y0) / max(1.0, support.height()) - 0.22)
            score = (
                max_overlap * float(cfg["placement"].get("avoid_overlap_weight", 5.5))
                + center_offset * float(cfg["placement"].get("prefer_center_weight", 0.25))
                + dst * float(cfg["placement"].get("depth_std_weight", 1.8))
                + band_pref * float(cfg["placement"].get("support_band_weight", 1.4))
                - support.score * 0.25
            )
            out.append(PlacementCandidate(candidate, support, float(score), f"support={support.label} overlap={max_overlap:.3f} depth_std={dst:.3f}"))
    out.sort(key=lambda c: c.score)
    return out


# -----------------------------------------------------------------------------
# Paper-style conditioning and scoring
# -----------------------------------------------------------------------------
def high_frequency_map(image: Image.Image, radius_ratio: float = 0.08) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    chans = []
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    radius = max(2, int(round(min(h, w) * radius_ratio)))
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = np.ones((h, w), dtype=np.float32)
    mask[dist2 <= radius * radius] = 0.0
    for c in range(3):
        F = np.fft.fft2(rgb[:, :, c])
        Fc = np.fft.fftshift(F)
        Fh = Fc * mask
        out = np.fft.ifft2(np.fft.ifftshift(Fh))
        mag = np.abs(out)
        chans.append(mag)
    hf = np.stack(chans, axis=2)
    hf -= hf.min()
    hf /= max(1e-6, hf.max())
    hf = (hf * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(hf, mode="RGB")


def feather_mask(box: BoundingBox, image_size: Tuple[int, int], expand_px: int, blur_px: int) -> Image.Image:
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = box.as_int()
    x0 = max(0, x0 - expand_px)
    y0 = max(0, y0 - expand_px)
    x1 = min(w, x1 + expand_px)
    y1 = min(h, y1 + expand_px)
    mask[y0:y1, x0:x1] = 255
    if blur_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_px, sigmaY=blur_px)
    return Image.fromarray(mask, mode="L")


def pad_to_square(image: Image.Image, size: int, bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (size, size), bg)
    scale = min(size / image.width, size / image.height)
    nw = max(1, int(round(image.width * scale)))
    nh = max(1, int(round(image.height * scale)))
    resized = image.resize((nw, nh), Image.LANCZOS)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def build_conditioning_canvas(
    scene: Image.Image,
    ref_rgb: Image.Image,
    ref_hf: Image.Image,
    mask: Image.Image,
    cfg: dict,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    ps = cfg["paper_style"]
    ref_size = int(ps.get("conditioning_ref_size", 256))
    gap = int(ps.get("conditioning_gap", 16))
    left = pad_to_square(ref_rgb, ref_size)
    middle = pad_to_square(ref_hf, ref_size, bg=(0, 0, 0))
    scene_panel = scene.convert("RGB")

    W = left.width + gap + middle.width + gap + scene_panel.width
    H = max(left.height, middle.height, scene_panel.height)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    x_ref = 0
    x_hf = left.width + gap
    x_scene = x_hf + middle.width + gap
    y_small = (H - left.height) // 2
    y_scene = (H - scene_panel.height) // 2
    canvas.paste(left, (x_ref, y_small))
    canvas.paste(middle, (x_hf, y_small))
    canvas.paste(scene_panel, (x_scene, y_scene))

    full_mask = Image.new("L", canvas.size, 0)
    full_mask.paste(mask, (x_scene, y_scene))
    scene_rect = (x_scene, y_scene, x_scene + scene_panel.width, y_scene + scene_panel.height)
    return canvas, full_mask, scene_rect


def crop_scene_from_canvas(canvas_img: Image.Image, scene_rect: Tuple[int, int, int, int]) -> Image.Image:
    return canvas_img.crop(scene_rect)


def masked_region_bbox(mask: Image.Image) -> Optional[BoundingBox]:
    arr = np.array(mask, dtype=np.uint8)
    ys, xs = np.where(arr > 8)
    if len(xs) == 0:
        return None
    return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def high_freq_similarity(a: Image.Image, b: Image.Image) -> float:
    aa = np.array(high_frequency_map(a, 0.08).convert("L"), dtype=np.float32) / 255.0
    bb = np.array(high_frequency_map(b, 0.08).convert("L"), dtype=np.float32) / 255.0
    if aa.shape != bb.shape:
        bb = cv2.resize(bb, (aa.shape[1], aa.shape[0]), interpolation=cv2.INTER_LINEAR)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-6:
        return 0.0
    return float(np.clip(np.sum(aa * bb) / denom, -1.0, 1.0))


def rgb_hist_similarity(a: Image.Image, b: Image.Image) -> float:
    aa = np.array(a.convert("RGB"), dtype=np.uint8)
    bb = np.array(b.convert("RGB"), dtype=np.uint8)
    if aa.shape != bb.shape:
        bb = np.array(Image.fromarray(bb).resize((aa.shape[1], aa.shape[0]), Image.LANCZOS), dtype=np.uint8)
    sims = []
    for i in range(3):
        ha = cv2.calcHist([aa], [i], None, [32], [0, 256])
        hb = cv2.calcHist([bb], [i], None, [32], [0, 256])
        ha = cv2.normalize(ha, ha).flatten()
        hb = cv2.normalize(hb, hb).flatten()
        sims.append(float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))
    return float(np.mean(sims))


def score_result(result_scene: Image.Image, placement: BoundingBox, ref: Image.Image, support: BoundingBox, depth: np.ndarray) -> float:
    x0, y0, x1, y1 = placement.as_int()
    crop = result_scene.crop((x0, y0, x1, y1)).convert("RGB")
    ref_resized = ref.resize(crop.size, Image.LANCZOS)
    hf = high_freq_similarity(crop, ref_resized)
    hist = rgb_hist_similarity(crop, ref_resized)
    patch_std = local_depth_std(depth, BoundingBox(x0, max(0, y1 - (y1 - y0) * 0.18), x1, y1))
    support_center_penalty = abs(placement.centre()[0] - support.centre()[0]) / max(1.0, support.width())
    return hf * 1.6 + hist * 0.7 - patch_std * 0.2 - support_center_penalty * 0.1


def make_prompt(label: str) -> str:
    return (
        f"Photorealistic product insertion. In the right panel, fill only the masked region with the exact same {label} "
        f"shown in the left reference panel. Preserve shape, color, material, logo, text, pattern, and branding. "
        f"Use the middle panel's detail map to preserve fine details and edges. Keep the result natural, coherent, "
        f"and physically integrated with the surrounding scene. Do not alter anything outside the masked region."
    )


def make_detail_prompt(label: str) -> str:
    return (
        f"Refine only the masked area of this {label}. Keep the global placement unchanged. Sharpen small text, logo, "
        f"label edges, fine patterns, and material micro-details while preserving the exact product identity and scene lighting."
    )


def run_fill(pipe, image: Image.Image, mask: Image.Image, prompt: str, seed: int, cfg: dict) -> Image.Image:
    steps = int(cfg["paper_style"].get("num_steps", 32))
    guidance = float(cfg["paper_style"].get("guidance_scale", 17.0))
    max_seq = int(cfg["paper_style"].get("max_sequence_length", 256))
    w, h = image.size
    out_w = max(16, int(round(w / 16) * 16))
    out_h = max(16, int(round(h / 16) * 16))
    image_r = image.resize((out_w, out_h), Image.LANCZOS)
    mask_r = mask.resize((out_w, out_h), Image.LANCZOS)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        image=image_r,
        mask_image=mask_r,
        height=out_h,
        width=out_w,
        guidance_scale=guidance,
        num_inference_steps=steps,
        max_sequence_length=max_seq,
        generator=generator,
    ).images[0].convert("RGB")
    return result.resize((w, h), Image.LANCZOS)


def run_detail_refine(pipe, canvas: Image.Image, full_mask: Image.Image, scene_rect: Tuple[int, int, int, int], label: str, cfg: dict) -> Image.Image:
    mask_bbox = masked_region_bbox(full_mask)
    if mask_bbox is None:
        return canvas
    x0, y0, x1, y1 = mask_bbox.as_int()
    pad = 48
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(canvas.width, x1 + pad)
    y1 = min(canvas.height, y1 + pad)
    crop = canvas.crop((x0, y0, x1, y1)).convert("RGB")
    crop_mask = full_mask.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=3.0))
    steps = int(cfg["paper_style"].get("detail_refine_steps", 18))
    guidance = float(cfg["paper_style"].get("detail_refine_guidance", 12.0))
    generator = torch.Generator(device="cpu").manual_seed(1234)
    out_w = max(16, int(round(crop.width / 16) * 16))
    out_h = max(16, int(round(crop.height / 16) * 16))
    result = pipe(
        prompt=make_detail_prompt(label),
        image=crop.resize((out_w, out_h), Image.LANCZOS),
        mask_image=crop_mask.resize((out_w, out_h), Image.LANCZOS),
        height=out_h,
        width=out_w,
        guidance_scale=guidance,
        num_inference_steps=steps,
        max_sequence_length=int(cfg["paper_style"].get("max_sequence_length", 256)),
        generator=generator,
    ).images[0].convert("RGB").resize(crop.size, Image.LANCZOS)
    merged = canvas.copy()
    soft = crop_mask.filter(ImageFilter.GaussianBlur(radius=4.0))
    merged_crop = Image.composite(result, crop, soft)
    merged.paste(merged_crop, (x0, y0))
    return merged


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def save_debug(scene: Image.Image, supports: List[BoundingBox], candidates: List[PlacementCandidate], chosen: PlacementCandidate, path: Path) -> None:
    debug = scene.convert("RGB").copy()
    draw = ImageDraw.Draw(debug)
    for s in supports:
        draw.rectangle(s.as_int(), outline=(0, 255, 0), width=3)
        if s.label:
            draw.text((s.x0 + 3, s.y0 + 3), s.label, fill=(0, 255, 0))
    for i, c in enumerate(candidates[:8]):
        color = (255, 200, 0) if i else (255, 0, 0)
        draw.rectangle(c.box.as_int(), outline=color, width=2)
    draw.rectangle(chosen.box.as_int(), outline=(255, 0, 0), width=4)
    debug.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned reference-based inpainting approximation for object insertion.")
    parser.add_argument("--scene", required=True, help="Path to scene image")
    parser.add_argument("--object-image", required=True, help="Path to reference product/object image")
    parser.add_argument("--object-label", required=True, help="Object/product label")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--config", default=None, help="Optional YAML config")
    parser.add_argument("--attempt", type=int, default=None, help="Candidate index override")
    parser.add_argument("--debug-overlay", default=None, help="Optional placement overlay path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.attempt is not None:
        cfg["placement"]["attempt_index"] = int(args.attempt)

    device = get_device(args.device)
    print(f"Using device: {device}")

    scene = open_rgb(args.scene)
    ref_img = open_rgb(args.object_image)

    print("Estimating depth...")
    depth = estimate_depth_map(scene, device, cfg)

    print("Extracting reference object...")
    extracted = extract_reference_object(ref_img, args.object_label, device, cfg)
    ref_rgb = Image.new("RGB", extracted.rgba.size, (255, 255, 255))
    ref_rgb.paste(extracted.rgba.convert("RGB"), mask=extracted.mask)
    ref_hf = high_frequency_map(ref_rgb, float(cfg["paper_style"].get("hf_radius_ratio", 0.08)))

    print("Detecting supports and obstacles...")
    supports = filter_supports(
        detect_boxes(scene, cfg["detection"]["surface_labels"], device, cfg, cfg["detection"]["surface_threshold"], cfg["detection"]["surface_text_threshold"]),
        scene.size,
    )
    if not supports:
        raise RuntimeError("No plausible support surfaces found.")
    obstacles = detect_boxes(scene, cfg["detection"]["obstacle_labels"] + [args.object_label], device, cfg, cfg["detection"]["obstacle_threshold"], cfg["detection"]["obstacle_text_threshold"])

    print("Ranking candidate regions...")
    candidates = propose_placements(scene, extracted, supports, obstacles, depth, cfg)
    if not candidates:
        raise RuntimeError("No plausible placement candidates found.")
    attempt_index = int(cfg["placement"].get("attempt_index", 0))
    chosen = candidates[min(max(0, attempt_index), len(candidates) - 1)]
    print(f"Chosen candidate: {chosen.reason}")

    mask = feather_mask(
        chosen.box,
        scene.size,
        expand_px=int(cfg["paper_style"].get("mask_expand_px", 14)),
        blur_px=int(cfg["paper_style"].get("mask_feather_px", 7)),
    )

    print("Building paper-style conditioning canvas...")
    conditioning_canvas, conditioning_mask, scene_rect = build_conditioning_canvas(scene, ref_rgb, ref_hf, mask, cfg)

    if cfg["output"].get("save_conditioning_canvas", False):
        cond_path = Path(args.output).with_name(Path(args.output).stem + "_conditioning.png")
        conditioning_canvas.save(cond_path)
        print(f"Saved conditioning canvas to {cond_path}")
    if cfg["output"].get("save_mask", False):
        mask_path = Path(args.output).with_name(Path(args.output).stem + "_mask.png")
        conditioning_mask.save(mask_path)
        print(f"Saved mask to {mask_path}")

    print("Running multi-seed reference-based inpainting...")
    pipe = get_fill_pipe(device, cfg)
    seeds = list(cfg["paper_style"].get("multi_seed", [0, 1, 2]))
    prompt = make_prompt(args.object_label)

    best_canvas = None
    best_score = -1e9
    for seed in seeds:
        generated_canvas = run_fill(pipe, conditioning_canvas, conditioning_mask, prompt, int(seed), cfg)
        if cfg["paper_style"].get("detail_refine", True):
            generated_canvas = run_detail_refine(pipe, generated_canvas, conditioning_mask, scene_rect, args.object_label, cfg)
        result_scene = crop_scene_from_canvas(generated_canvas, scene_rect)
        score = score_result(result_scene, chosen.box, ref_rgb, chosen.support, depth)
        print(f"Seed {seed}: score={score:.4f}")
        if score > best_score:
            best_score = score
            best_canvas = generated_canvas

    if best_canvas is None:
        raise RuntimeError("Inpainting failed for all seeds.")

    final_scene = crop_scene_from_canvas(best_canvas, scene_rect)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_scene.save(out_path)
    print(f"Saved output to {out_path}")

    if args.debug_overlay:
        dbg_path = Path(args.debug_overlay)
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug(scene, supports, candidates, chosen, dbg_path)
        print(f"Saved debug overlay to {dbg_path}")


if __name__ == "__main__":
    main()
