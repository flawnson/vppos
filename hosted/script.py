import argparse
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import InferenceClient


DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit"
DEFAULT_OBJECT_SEG_MODEL = "briaai/RMBG-2.0"
DEFAULT_SCENE_SEG_MODEL = "facebook/mask2former-swin-large-coco-panoptic"
DEFAULT_PROVIDER = "auto"

# Internal policy knobs.
INTERNAL_SEG_MAX_SCENE_SIDE = 1536
INTERNAL_EDIT_MAX_SIDE = 1408
MIN_LOCAL_PADDING_PX = 80
MAX_LOCAL_PADDING_FRACTION = 0.35
LOCAL_MASK_BLUR_RADIUS = 12

# Conservative sizing policy. These are intentionally a bit smaller than before.
MIN_OBJECT_WIDTH_FRAC = 0.055
DEFAULT_OBJECT_WIDTH_FRAC = 0.085
MAX_OBJECT_WIDTH_FRAC = 0.135
ANCHOR_SEARCH_SIDE_FRAC = 0.22
MIN_SUPPORT_SPAN_FRAC = 0.10

SUPPORT_LABEL_PRIORITY = {
    "countertop": 0,
    "counter": 1,
    "table": 2,
    "dining table": 3,
    "desk": 4,
    "bench": 5,
    "shelf": 6,
    "coffee table": 7,
}


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


def choose_best_mask(seg_outputs: Iterable) -> Optional[Image.Image]:
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


def _mask_bbox(mask: Image.Image) -> Optional[tuple[int, int, int, int]]:
    bbox = mask.getbbox()
    if bbox is None:
        return None
    return tuple(int(v) for v in bbox)


def _top_profile(mask: Image.Image, bbox: tuple[int, int, int, int]) -> list[Optional[int]]:
    x0, y0, x1, y1 = bbox
    profile: list[Optional[int]] = []
    for x in range(x0, x1):
        hit_y = None
        for y in range(y0, y1):
            if mask.getpixel((x, y)) > 30:
                hit_y = y
                break
        profile.append(hit_y)
    return profile


def _find_best_flat_span(profile: list[Optional[int]], x0: int, scene_h: int) -> Optional[tuple[int, int, int, float]]:
    min_span = max(18, int(len(profile) * MIN_SUPPORT_SPAN_FRAC))
    best = None
    best_score = -1e9
    start = 0
    while start < len(profile):
        while start < len(profile) and profile[start] is None:
            start += 1
        if start >= len(profile):
            break
        end = start
        ys = []
        while end < len(profile) and profile[end] is not None:
            ys.append(profile[end])
            end += 1
        if len(ys) >= min_span:
            local_min = min(ys)
            local_max = max(ys)
            y_variation = local_max - local_min
            span_width = end - start
            median_y = sorted(ys)[len(ys) // 2]
            flatness = 1.0 / (1.0 + y_variation)
            lower_bonus = max(0.0, (median_y / float(scene_h)) - 0.28)
            score = span_width * 2.2 + lower_bonus * 120.0 - y_variation * 7.0
            if score > best_score:
                best_score = score
                anchor_x = x0 + (start + end) // 2
                best = (anchor_x, median_y, span_width, flatness)
        start = end + 1
    return best


def _local_span_around_anchor(profile: list[Optional[int]], x0: int, anchor_x: int, support_y: int) -> tuple[int, int, int, float]:
    idx = max(0, min(len(profile) - 1, anchor_x - x0))
    tolerance = max(3, int(0.008 * max(1, len(profile))))
    left = idx
    while left - 1 >= 0 and profile[left - 1] is not None and abs(profile[left - 1] - support_y) <= tolerance:
        left -= 1
    right = idx
    while right + 1 < len(profile) and profile[right + 1] is not None and abs(profile[right + 1] - support_y) <= tolerance:
        right += 1
    span_width = right - left + 1
    flat_band = [p for p in profile[left:right + 1] if p is not None]
    y_variation = (max(flat_band) - min(flat_band)) if flat_band else 999
    flatness = 1.0 / (1.0 + y_variation)
    return x0 + left, x0 + right, span_width, flatness


def find_support_candidates(scene_rgb: Image.Image, client: InferenceClient, seg_model: str) -> list[SupportCandidate]:
    try:
        seg = client.image_segmentation(pil_to_png_bytes(scene_rgb), model=seg_model)
    except Exception:
        return []

    scene_w, scene_h = scene_rgb.size
    candidates: list[SupportCandidate] = []

    for output in seg:
        label = str(getattr(output, "label", "")).lower().strip()
        if label not in SUPPORT_LABEL_PRIORITY:
            continue

        mask = output.mask.convert("L")
        if mask.size != scene_rgb.size:
            mask = mask.resize(scene_rgb.size, Image.LANCZOS)
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        area = width * height
        if area <= 0:
            continue
        if y1 < int(scene_h * 0.35):
            continue
        if width < int(scene_w * 0.12):
            continue

        profile = _top_profile(mask, bbox)
        flat = _find_best_flat_span(profile, x0, scene_h)
        if flat is None:
            continue
        anchor_x, support_y, _, _ = flat
        span_left, span_right, local_span_width, flatness = _local_span_around_anchor(profile, x0, anchor_x, support_y)
        if local_span_width < int(scene_w * MIN_SUPPORT_SPAN_FRAC):
            continue

        center_x = (x0 + x1) / 2.0
        center_bias = 1.0 - min(1.0, abs(center_x - scene_w / 2.0) / (scene_w / 2.0))
        lower_bias = min(1.0, support_y / float(scene_h))
        width_frac = width / float(scene_w)
        priority_bonus = max(0, 10 - SUPPORT_LABEL_PRIORITY[label]) * 16.0
        score = (
            priority_bonus
            + local_span_width * 1.8
            + width_frac * 120.0
            + flatness * 180.0
            + center_bias * 35.0
            + lower_bias * 40.0
        )

        candidates.append(
            SupportCandidate(
                label=label,
                mask=mask,
                bbox=bbox,
                support_y=int(support_y),
                anchor_x=int(anchor_x),
                local_span_width=int(local_span_width),
                flatness=float(flatness),
                score=float(score),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def choose_support_candidate(candidates: list[SupportCandidate], scene_size: tuple[int, int]) -> Optional[SupportCandidate]:
    if not candidates:
        return None
    scene_w, _ = scene_size
    best = None
    best_score = -1e18
    for c in candidates:
        x0, _, x1, _ = c.bbox
        width_frac = (x1 - x0) / float(scene_w)
        conservative_surface_bonus = min(1.0, c.local_span_width / max(1.0, scene_w * 0.24)) * 40.0
        score = c.score + conservative_surface_bonus + width_frac * 30.0
        if score > best_score:
            best_score = score
            best = c
    return best


def compute_target_width(scene: Image.Image, obj_rgba: Image.Image, candidate: Optional[SupportCandidate], args: argparse.Namespace) -> int:
    scene_w, scene_h = scene.size
    obj_w, obj_h = obj_rgba.size
    _ = obj_w, obj_h

    width_from_scene = int(scene_w * DEFAULT_OBJECT_WIDTH_FRAC)
    width_from_height = int(scene_h * 0.16)
    target = min(width_from_scene, width_from_height)

    if candidate is not None:
        x0, _, x1, _ = candidate.bbox
        support_width = x1 - x0
        local_span = candidate.local_span_width
        # Conservative: fit to a modest fraction of the locally flat surface, not the full support bbox.
        width_from_local_span = int(local_span * 0.22)
        width_from_support = int(support_width * 0.12)
        width_from_scene = int(scene_w * DEFAULT_OBJECT_WIDTH_FRAC)
        target = min(max(width_from_local_span, int(scene_w * MIN_OBJECT_WIDTH_FRAC)), width_from_support, width_from_scene)
        # If the support is especially narrow or not that flat, shrink further.
        if candidate.flatness < 0.35:
            target = int(target * 0.88)
        if local_span < int(scene_w * 0.18):
            target = int(target * 0.88)

    target = max(int(scene_w * MIN_OBJECT_WIDTH_FRAC), target)
    target = min(int(scene_w * MAX_OBJECT_WIDTH_FRAC), target)

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
        center_x = chosen_candidate.anchor_x
    else:
        center_x = scene_w // 2

    if args.y is not None:
        target_bottom = int(scene_h * args.y)
    elif chosen_candidate is not None:
        target_bottom = chosen_candidate.support_y + max(2, int(scene_h * 0.004))
    else:
        target_bottom = int(scene_h * 0.80)

    # Keep the object comfortably inside the frame.
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
        label = f"{idx+1}:{c.label} s={int(c.score)}"
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
    pad_x = max(MIN_LOCAL_PADDING_PX, int(placement.width * 0.9))
    pad_y = max(MIN_LOCAL_PADDING_PX, int(placement.height * 0.9))

    pad_x = min(pad_x, int(scene_w * MAX_LOCAL_PADDING_FRACTION))
    pad_y = min(pad_y, int(scene_h * MAX_LOCAL_PADDING_FRACTION))

    x0 = max(0, placement.left - pad_x)
    y0 = max(0, placement.top - int(pad_y * 0.8))
    x1 = min(scene_w, placement.left + placement.width + pad_x)
    y1 = min(scene_h, placement.top + placement.height + int(pad_y * 1.35))
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

    print("[2/6] Finding and ranking support surfaces...")
    candidates_small = find_support_candidates(scene_for_seg.convert("RGB"), client, args.scene_seg_model)
    candidates_full: list[SupportCandidate] = []
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
            )
        )
    chosen_candidate = choose_support_candidate(candidates_full, scene_full.size)
    if chosen_candidate is not None:
        print(f"      Chosen support: {chosen_candidate.label} | score={chosen_candidate.score:.1f} | span={chosen_candidate.local_span_width}px")
    else:
        print("      No support surface found. Falling back to conservative center-lower placement.")

    candidate_preview = draw_support_candidates_preview(scene_full, candidates_full, chosen_candidate)
    debug_run.save_image(candidate_preview, "06_support_candidates_ranked.png")
    if chosen_candidate is not None:
        debug_run.save_image(chosen_candidate.mask, "07_chosen_support_mask_fullres.png")

    print("[3/6] Computing systematic placement and full-resolution precompositing...")
    placement = compute_auto_placement(scene_full, obj_rgba, chosen_candidate, args)
    placement_preview = draw_placement_preview(scene_full, placement, chosen_candidate)
    precomp_full, resized_obj, shadow_full, object_layer_full = precompose(scene_full, obj_rgba, placement)
    debug_run.save_image(placement_preview, "08_placement_preview_fullres.png")
    debug_run.save_image(resized_obj, "09_object_resized.png")
    debug_run.save_image(shadow_full, "10_shadow_fullres.png")
    debug_run.save_image(object_layer_full, "11_object_layer_fullres.png")
    debug_run.save_image(precomp_full, "12_precomposite_fullres.png", mode="RGB")

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
    debug_run.save_image(original_crop, "13_original_local_crop.png", mode="RGB")
    debug_run.save_image(precomp_crop, "14_precomposite_local_crop.png", mode="RGB")
    debug_run.save_image(local_mask_full, "15_local_blend_mask_fullres.png")
    debug_run.save_image(precomp_crop_for_edit, "16_crop_sent_to_edit_model.png", mode="RGB")

    if args.skip_refine:
        print("[4/6] Skipping HF refinement as requested.")
        refined_crop_full = precomp_crop.convert("RGB")
    else:
        print("[4/6] Refining only the local crop with HF image edit model...")
        print(f"      Model: {args.edit_model}")
        refined_crop_small = refine_with_hf(precomp_crop_for_edit, client, prompt, args.edit_model).convert("RGB")
        debug_run.save_image(refined_crop_small, "17_refined_local_crop_model_output.png", mode="RGB")
        if refined_crop_small.size != precomp_crop.size:
            refined_crop_full = refined_crop_small.resize(precomp_crop.size, Image.LANCZOS)
        else:
            refined_crop_full = refined_crop_small
        debug_run.save_image(refined_crop_full, "18_refined_local_crop_resized_to_fullres.png", mode="RGB")

    print("[5/6] Blending refined crop back into the original full-resolution scene...")
    blended_crop = blend_local_edit(original_crop, refined_crop_full, local_mask_full)
    final_full = scene_full.copy().convert("RGBA")
    final_full.paste(blended_crop, (crop_x0, crop_y0), blended_crop)
    debug_run.save_image(blended_crop, "19_blended_local_crop.png", mode="RGB")
    debug_run.save_image(final_full, "20_final_fullres_before_save.png", mode="RGB")

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
        "placement": asdict(placement),
        "chosen_support": None if chosen_candidate is None else {
            "label": chosen_candidate.label,
            "bbox": list(chosen_candidate.bbox),
            "support_y": chosen_candidate.support_y,
            "anchor_x": chosen_candidate.anchor_x,
            "local_span_width": chosen_candidate.local_span_width,
            "flatness": chosen_candidate.flatness,
            "score": chosen_candidate.score,
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
        "internal_policy": {
            "segmentation_max_side": INTERNAL_SEG_MAX_SCENE_SIDE,
            "edit_crop_max_side": INTERNAL_EDIT_MAX_SIDE,
            "min_local_padding_px": MIN_LOCAL_PADDING_PX,
            "max_local_padding_fraction": MAX_LOCAL_PADDING_FRACTION,
            "min_object_width_frac": MIN_OBJECT_WIDTH_FRAC,
            "default_object_width_frac": DEFAULT_OBJECT_WIDTH_FRAC,
            "max_object_width_frac": MAX_OBJECT_WIDTH_FRAC,
        },
        "note": "Placement now ranks multiple support candidates, anchors on the flattest local span, and sizes the product conservatively from local support geometry instead of scene width.",
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
