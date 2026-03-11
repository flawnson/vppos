import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoModelForImageSegmentation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    SamModel,
    SamProcessor,
    VitMatteForImageMatting,
    VitMatteImageProcessor,
    pipeline as hf_pipeline,
)

from qwen_vl_utils import process_vision_info


DEFAULT_SIZE_PROFILES = {
    "small": {
        "support_width_frac": 0.14,
        "height_frac_near": 0.13,
        "height_frac_far": 0.06,
    },
    "medium": {
        "support_width_frac": 0.22,
        "height_frac_near": 0.20,
        "height_frac_far": 0.10,
    },
    "large": {
        "support_width_frac": 0.34,
        "height_frac_near": 0.32,
        "height_frac_far": 0.16,
    },
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_uri(path: Path) -> str:
    return str(path.resolve())


def _to_pil_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _pil_rgba_to_np_rgba(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGBA"))


def _np_rgba_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _safe_json_extract(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def _make_trimap(alpha_u8: np.ndarray, fg_erode: int = 10, bg_dilate: int = 10) -> np.ndarray:
    a = alpha_u8
    fg = (a > 200).astype(np.uint8) * 255
    bg = (a < 5).astype(np.uint8) * 255

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_erode, fg_erode))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilate, bg_dilate))

    fg2 = cv2.erode(fg, k1, iterations=1)
    bg2 = cv2.dilate(bg, k2, iterations=1)

    trimap = np.full_like(a, 128, dtype=np.uint8)
    trimap[bg2 > 0] = 0
    trimap[fg2 > 0] = 255
    return trimap


def _normalize_depth_for_scale(depth_u8: np.ndarray) -> np.ndarray:
    d = depth_u8.astype(np.float32)
    h = d.shape[0]
    top = d[: h // 4].mean()
    bottom = d[3 * h // 4 :].mean()
    if bottom < top:
        d = d.max() - d
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return d


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def _alpha_bbox(alpha_u8: np.ndarray, thr: int = 10) -> Tuple[int, int, int, int]:
    return _bbox_from_mask((alpha_u8 > thr).astype(np.uint8))


def _crop_to_alpha(img_rgba: Image.Image, pad: int = 4) -> Image.Image:
    arr = _pil_rgba_to_np_rgba(img_rgba)
    alpha = arr[:, :, 3]
    x1, y1, x2, y2 = _alpha_bbox(alpha, thr=10)
    h, w = alpha.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    cropped = arr[y1 : y2 + 1, x1 : x2 + 1, :]
    return _np_rgba_to_pil(cropped)


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

    patch = out[y1:y2, x1:x2, :]
    obj_patch = obj_rgba[oy1:oy2, ox1:ox2, :]

    alpha = obj_patch[:, :, 3:4].astype(np.float32) / 255.0
    rgb = obj_patch[:, :, :3].astype(np.float32)
    patch_f = patch.astype(np.float32)

    blended = rgb * alpha + patch_f * (1.0 - alpha)
    out[y1:y2, x1:x2, :] = blended.astype(np.uint8)
    return out


def _render_contact_shadow(
    scene_rgb: np.ndarray,
    obj_alpha_u8: np.ndarray,
    x: int,
    y: int,
    direction: str,
    softness_px: int,
    opacity: float,
    squash_y: float,
    shear_x: float,
    offset_px: int,
) -> np.ndarray:
    h, w = obj_alpha_u8.shape[:2]

    yy = np.linspace(0, 1, h).reshape(h, 1)
    weight = np.clip((yy - 0.70) / 0.30, 0, 1)
    base = (obj_alpha_u8.astype(np.float32) / 255.0) * weight
    base = (base * 255).astype(np.uint8)

    dx, dy = 0, 0
    dir_l = (direction or "").lower().strip()
    if "left" in dir_l:
        dx = +offset_px
    if "right" in dir_l:
        dx = -offset_px
    if "top" in dir_l:
        dy = +offset_px
    if "bottom" in dir_l:
        dy = -offset_px

    M = np.array([[1.0, shear_x, 0.0], [0.0, squash_y, 0.0]], dtype=np.float32)
    shadow = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, softness_px / 6.0))

    shadow_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    shadow_rgba[:, :, 3] = np.clip(shadow.astype(np.float32) * opacity, 0, 255).astype(np.uint8)

    out2 = scene_rgb.copy()
    H2, W2 = out2.shape[:2]
    top_left_x = x + dx
    top_left_y = y + dy + int(h * 0.55)

    x1 = max(0, top_left_x)
    y1 = max(0, top_left_y)
    x2 = min(W2, top_left_x + w)
    y2 = min(H2, top_left_y + h)
    if x1 >= x2 or y1 >= y2:
        return out2

    sx1 = x1 - top_left_x
    sy1 = y1 - top_left_y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    alpha = shadow_rgba[sy1:sy2, sx1:sx2, 3:4].astype(np.float32) / 255.0
    patch = out2[y1:y2, x1:x2, :].astype(np.float32)
    darkened = patch * (1.0 - 0.55 * alpha)
    out2[y1:y2, x1:x2, :] = np.clip(darkened, 0, 255).astype(np.uint8)
    return out2


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    best_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == best_idx).astype(np.uint8)


def _mask_stats(mask: np.ndarray) -> Dict[str, float]:
    x1, y1, x2, y2 = _bbox_from_mask(mask)
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    area = float((mask > 0).sum())
    fill = area / float(w * h)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": w,
        "h": h,
        "area": area,
        "fill": fill,
    }


def _surface_name_score(surface_name: str) -> float:
    s = (surface_name or "").lower()
    if any(k in s for k in ["counter", "countertop", "island", "table", "desk"]):
        return 1.0
    if any(k in s for k in ["shelf", "mantel", "ledge"]):
        return 0.7
    if any(k in s for k in ["cabinet", "drawer", "door", "wall", "window", "fridge", "oven"]):
        return -1.0
    return 0.0


def _likely_vertical_surface(mask: np.ndarray, surface_name: str) -> bool:
    s = _mask_stats(mask)
    name = (surface_name or "").lower()

    if any(k in name for k in ["cabinet", "drawer", "door", "wall", "window", "fridge", "oven"]):
        return True

    if s["h"] > s["w"] * 1.15 and s["fill"] > 0.35:
        return True

    return False


def _surface_top_band(mask: np.ndarray, surface_name: str, band_px: int = 14) -> np.ndarray:
    mask = _largest_connected_component(mask)
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=np.uint8)

    cols = np.where(mask.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return out

    thickness = band_px
    if "shelf" in (surface_name or "").lower():
        thickness = max(thickness, 10)

    for x in cols:
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        y0 = int(ys.min())
        y1 = min(H, y0 + thickness)
        out[y0:y1, x] = 1

    out = (out & mask).astype(np.uint8)
    kx = max(7, band_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return out


def _iter_true_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(row.astype(bool)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return runs


def _support_candidates_from_band(
    support_band: np.ndarray,
    min_run_width: int,
    max_candidates: int = 48,
) -> List[Tuple[int, int, int]]:
    H, W = support_band.shape
    candidates: List[Tuple[int, int, int]] = []

    row_scores = []
    for y in range(H):
        row_sum = int(support_band[y].sum())
        if row_sum > 0:
            row_scores.append((row_sum, y))
    row_scores.sort(reverse=True)

    used = set()
    for _, y in row_scores[: min(30, len(row_scores))]:
        runs = _iter_true_runs(support_band[y])
        for x1, x2 in runs:
            run_w = x2 - x1 + 1
            if run_w < min_run_width:
                continue

            xs = [int((x1 + x2) / 2)]
            if run_w >= min_run_width * 2:
                xs.extend([x1 + run_w // 3, x1 + (2 * run_w) // 3])

            for xc in xs:
                key = (int(xc / 8), int(y / 4))
                if key in used:
                    continue
                used.add(key)
                candidates.append((int(xc), int(y), int(run_w)))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[:max_candidates]


def _contact_support_ratio(
    support_band: np.ndarray,
    alpha_mask: np.ndarray,
    top_left_x: int,
    top_left_y: int,
) -> float:
    H, W = support_band.shape
    x1 = max(0, top_left_x)
    y1 = max(0, top_left_y)
    x2 = min(W, top_left_x + alpha_mask.shape[1])
    y2 = min(H, top_left_y + alpha_mask.shape[0])
    if x1 >= x2 or y1 >= y2:
        return 0.0

    ox1 = x1 - top_left_x
    oy1 = y1 - top_left_y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    alpha_patch = alpha_mask[oy1:oy2, ox1:ox2]
    if alpha_patch.size == 0:
        return 0.0

    h = alpha_patch.shape[0]
    y_start = max(0, int(h * 0.88))
    contact = (alpha_patch[y_start:, :] > 10).astype(np.uint8)
    if contact.sum() == 0:
        return 0.0

    support = (support_band[y1:y2, x1:x2][y_start:, :] > 0).astype(np.uint8)
    overlap = (contact & support).sum()
    return float(overlap) / float(contact.sum() + 1e-6)


def _center_bias(x: int, width: int) -> float:
    mid = (width - 1) / 2.0
    return 1.0 - min(1.0, abs(x - mid) / max(1.0, mid))


def _make_qwen_scene_proxy(scene: Image.Image, debug_dir: Path) -> Path:
    proxy = scene.copy()
    proxy.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    out = debug_dir / "scene_for_qwen.jpg"
    proxy.save(out, quality=95)
    return out


@dataclass
class PlacementResult:
    surface_name: str
    x_center: int
    y_contact: int
    scale: float
    score: float
    support_width: int


class ContextualImageEditor:
    def __init__(self, cfg: Dict[str, Any], device: str) -> None:
        self.cfg = cfg
        self.device_str = device
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self) -> None:
        m = self.cfg["models"]

        self.cutout_model = AutoModelForImageSegmentation.from_pretrained(
            m["cutout_model"], trust_remote_code=True
        ).to(self.device)
        self.cutout_model.eval()

        self._cutout_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.vitmatte_processor = VitMatteImageProcessor.from_pretrained(m["matting_model"])
        self.vitmatte_model = VitMatteForImageMatting.from_pretrained(m["matting_model"]).to(self.device)
        self.vitmatte_model.eval()

        self.qwen_processor = AutoProcessor.from_pretrained(
            m["reasoning_vlm"],
            use_fast=False,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            m["reasoning_vlm"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        self.gd_processor = AutoProcessor.from_pretrained(m["grounding_dino"])
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(m["grounding_dino"]).to(self.device)
        self.gd_model.eval()

        self.sam_processor = SamProcessor.from_pretrained(m["sam"])
        self.sam_model = SamModel.from_pretrained(m["sam"]).to(self.device)
        self.sam_model.eval()

        self.depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=m["depth_model"],
            device=0 if self.device.type == "cuda" else -1,
        )

    def cutout_and_matte(self, obj_path: Path, debug_dir: Path) -> Tuple[Image.Image, Dict[str, Any]]:
        obj = Image.open(obj_path).convert("RGB")

        inp = self._cutout_transform(obj).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.cutout_model(inp)[-1].sigmoid().detach().cpu()

        pred = preds[0].squeeze().numpy()
        mask_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_u8, mode="L").resize(obj.size, Image.BILINEAR)

        rgba = obj.convert("RGBA")
        rgba.putalpha(mask_pil)

        rgba_cropped = _crop_to_alpha(rgba, pad=6)
        alpha0 = np.array(rgba_cropped.split()[-1])

        trimap = _make_trimap(alpha0, fg_erode=10, bg_dilate=12)
        trimap_pil = Image.fromarray(trimap, mode="L")
        rgb_cropped = rgba_cropped.convert("RGB")

        inputs = self.vitmatte_processor(images=rgb_cropped, trimaps=trimap_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            alphas = self.vitmatte_model(**inputs).alphas

        H, W = rgb_cropped.size[1], rgb_cropped.size[0]
        alpha_resized = torch.nn.functional.interpolate(alphas, size=(H, W), mode="bicubic", align_corners=False)[0, 0]
        alpha_u8 = torch.clamp(alpha_resized, 0, 1).mul(255).byte().cpu().numpy()

        rgba_refined = _pil_rgba_to_np_rgba(rgba_cropped)
        rgba_refined[:, :, 3] = alpha_u8
        rgba_refined_pil = _np_rgba_to_pil(rgba_refined)

        meta = {"cutout_mask_mean": float(mask_u8.mean())}

        if self.cfg.get("debug", {}).get("save_intermediates", True):
            _ensure_dir(debug_dir)
            rgba_refined_pil.save(debug_dir / "object_rgba_refined.png")
            Image.fromarray(trimap).save(debug_dir / "object_trimap.png")

        return rgba_refined_pil, meta

    def qwen_json(self, image_uri: str, prompt: str, max_new_tokens: int = 96) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        out_text = self.qwen_processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return _safe_json_extract(out_text[0] if out_text else "")

    def grounding_dino_detect(
        self,
        image: Image.Image,
        queries: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        q = " ".join([f"a {s.lower().strip().rstrip('.')}." for s in queries if s.strip()])
        if not q.strip():
            return []

        inputs = self.gd_processor(images=image, text=q, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)

        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        dets = []
        r0 = results[0]
        boxes = r0["boxes"].cpu().numpy() if hasattr(r0["boxes"], "cpu") else np.array(r0["boxes"])
        scores = r0["scores"].cpu().numpy() if hasattr(r0["scores"], "cpu") else np.array(r0["scores"])
        labels = r0.get("text_labels", r0.get("labels", []))

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            dets.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(scores[i]),
                    "label": str(labels[i]),
                }
            )
        return dets

    def sam_segment_boxes(self, image: Image.Image, boxes_xyxy: List[List[float]]) -> List[np.ndarray]:
        if not boxes_xyxy:
            return []

        input_boxes = [[[b[0], b[1], b[2], b[3]] for b in boxes_xyxy]]
        inputs = self.sam_processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=False)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]

        masks_np = []
        for m in masks:
            arr = np.squeeze(m.numpy())
            arr = (arr > 0).astype(np.uint8)
            masks_np.append(arr)

        return masks_np

    def estimate_depth(self, scene: Image.Image, debug_dir: Path) -> np.ndarray:
        depth_pil = self.depth_pipe(scene)["depth"]
        depth_u8 = np.array(depth_pil.convert("L"))
        closeness = _normalize_depth_for_scale(depth_u8)
        if self.cfg.get("debug", {}).get("save_intermediates", True):
            _ensure_dir(debug_dir)
            Image.fromarray((closeness * 255).astype(np.uint8)).save(debug_dir / "scene_depth_closeness.png")
        return closeness

    def choose_placement(
        self,
        scene: Image.Image,
        obj_rgba: Image.Image,
        surfaces: List[Tuple[str, np.ndarray]],
        obstacles_mask: np.ndarray,
        closeness: np.ndarray,
        object_size_class: str,
        preferred_height_band: str,
    ) -> Optional[PlacementResult]:
        W, H = scene.size
        obj_arr = _pil_rgba_to_np_rgba(obj_rgba)
        obj_alpha = obj_arr[:, :, 3]

        bx1, by1, bx2, by2 = _alpha_bbox(obj_alpha, thr=10)
        obj_h = max(1, by2 - by1 + 1)
        obj_w = max(1, bx2 - bx1 + 1)

        profiles = self.cfg.get("placement", {}).get("size_profiles", DEFAULT_SIZE_PROFILES)
        profile = profiles.get(object_size_class, profiles["medium"])

        clamp_cfg = self.cfg.get("placement", {}).get("clamp_scale", {})
        min_scale = float(clamp_cfg.get("min", 0.05))
        max_scale = float(clamp_cfg.get("max", 0.75))
        safety = int(self.cfg.get("placement", {}).get("safety_margin_px", 10))

        best: Optional[PlacementResult] = None

        for surface_name, raw_mask in surfaces:
            mask = _largest_connected_component(raw_mask)
            if mask.sum() < 1500:
                continue

            if _likely_vertical_surface(mask, surface_name):
                continue

            surface_stats = _mask_stats(mask)
            support_band = _surface_top_band(mask, surface_name, band_px=max(10, surface_stats["h"] // 18))

            obs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * safety + 1, 2 * safety + 1))
            inflated_obstacles = cv2.dilate((obstacles_mask > 0).astype(np.uint8), obs_kernel, iterations=1)
            free_support = (support_band > 0).astype(np.uint8)
            free_support[inflated_obstacles > 0] = 0

            if free_support.sum() < 200:
                continue

            min_run_width = max(28, int(W * 0.06))
            candidates = _support_candidates_from_band(free_support, min_run_width=min_run_width, max_candidates=40)
            if not candidates:
                continue

            for xc, yc, run_w in candidates:
                y_norm = yc / max(1, H - 1)
                perspective = 0.55 + 0.90 * y_norm

                support_width_target = max(20.0, run_w * float(profile["support_width_frac"]))
                height_frac = float(profile["height_frac_far"]) + (
                    float(profile["height_frac_near"]) - float(profile["height_frac_far"])
                ) * perspective
                target_height_px = H * height_frac

                scale_w = support_width_target / max(1.0, float(obj_w))
                scale_h = target_height_px / max(1.0, float(obj_h))

                obj_aspect = float(obj_h) / max(1.0, float(obj_w))
                if obj_aspect > 1.2:
                    scale = 0.35 * scale_w + 0.65 * scale_h
                else:
                    scale = 0.60 * scale_w + 0.40 * scale_h

                c_here = float(closeness[min(H - 1, yc), min(W - 1, xc)])
                c_surf = float(np.median(closeness[mask > 0])) if (mask > 0).any() else c_here
                depth_adjust = np.clip((c_here + 1e-4) / (c_surf + 1e-4), 0.82, 1.18)
                scale *= float(depth_adjust)
                scale = float(np.clip(scale, min_scale, max_scale))

                new_w = max(1, int(round(obj_arr.shape[1] * scale)))
                new_h = max(1, int(round(obj_arr.shape[0] * scale)))
                obj_scaled_alpha = cv2.resize(obj_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                sx1, sy1, sx2, sy2 = _alpha_bbox(obj_scaled_alpha, thr=10)
                if sx2 <= sx1 or sy2 <= sy1:
                    continue

                top_left_x = xc - int((sx1 + sx2) / 2)
                top_left_y = yc - sy2

                if top_left_x < 0 or top_left_y < 0:
                    continue
                if top_left_x + new_w > W or top_left_y + new_h > H:
                    continue

                x1 = top_left_x
                y1 = top_left_y
                x2 = top_left_x + new_w
                y2 = top_left_y + new_h

                alpha_patch = (obj_scaled_alpha > 10).astype(np.uint8)
                obs_patch = (obstacles_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
                if alpha_patch.shape != obs_patch.shape:
                    continue
                if int((alpha_patch & obs_patch).sum()) > 0:
                    continue

                support_ratio = _contact_support_ratio(free_support, obj_scaled_alpha, top_left_x, top_left_y)
                if support_ratio < 0.20:
                    continue

                band = (preferred_height_band or "").lower().strip()
                if band == "high":
                    band_score = 1.0 - y_norm
                elif band == "low":
                    band_score = y_norm
                else:
                    band_score = 1.0 - abs(y_norm - 0.55) * 2.0

                score = (
                    2.0 * _surface_name_score(surface_name)
                    + 1.8 * min(1.0, run_w / max(1.0, W * 0.35))
                    + 1.6 * support_ratio
                    + 0.9 * band_score
                    + 0.4 * _center_bias(xc, W)
                )

                if best is None or score > best.score:
                    best = PlacementResult(
                        surface_name=surface_name,
                        x_center=int(xc),
                        y_contact=int(yc),
                        scale=float(scale),
                        score=float(score),
                        support_width=int(run_w),
                    )

        return best

    def run(
        self,
        scene_path: Path,
        object_path: Path,
        output_dir: Path,
    ) -> Path:
        _ensure_dir(output_dir)
        debug_dir = output_dir / "debug"
        _ensure_dir(debug_dir)

        perf = self.cfg.get("performance", {})
        seed = int(perf.get("seed", 123))
        torch.manual_seed(seed)
        np.random.seed(seed)

        scene = Image.open(scene_path).convert("RGB")

        obj_rgba, _ = self.cutout_and_matte(object_path, debug_dir=debug_dir)
        obj_rgba.save(debug_dir / "object_rgba_final.png")

        obj_for_qwen_path = debug_dir / "object_for_qwen.png"
        obj_rgba.save(obj_for_qwen_path)
        obj_uri = _as_uri(obj_for_qwen_path)

        scene_for_qwen_path = _make_qwen_scene_proxy(scene, debug_dir)
        scene_uri = _as_uri(scene_for_qwen_path)

        obj_prompt = (
            "Return ONLY valid JSON.\n"
            "Describe this object for realistic placement in a photo scene.\n"
            "JSON schema:\n"
            "{\n"
            '  "object_name": string,\n'
            '  "size_class": "small"|"medium"|"large",\n'
            '  "preferred_surfaces": [string],\n'
            '  "must_be_upright": boolean\n'
            "}\n"
        )
        scene_prompt = (
            "Return ONLY valid JSON.\n"
            "You are helping place an object into this scene realistically.\n"
            "Infer likely top support surfaces that an object can rest on, not vertical faces.\n"
            "Avoid cabinet fronts, walls, and window areas.\n"
            "JSON schema:\n"
            "{\n"
            '  "scene_type": string,\n'
            '  "candidate_surfaces": [string],\n'
            '  "obstacles_to_avoid": [string],\n'
            '  "preferred_height_band": "low"|"mid"|"high",\n'
            '  "light_direction": string\n'
            "}\n"
        )

        obj_info = self.qwen_json(obj_uri, obj_prompt)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scene_info = self.qwen_json(scene_uri, scene_prompt)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del self.qwen_model
        self.qwen_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.cfg.get("debug", {}).get("save_intermediates", True):
            (debug_dir / "qwen_object.json").write_text(json.dumps(obj_info, indent=2), encoding="utf-8")
            (debug_dir / "qwen_scene.json").write_text(json.dumps(scene_info, indent=2), encoding="utf-8")

        size_class = str(obj_info.get("size_class", "medium")).lower().strip()
        light_dir = str(scene_info.get("light_direction", "top-left"))
        preferred_band = str(scene_info.get("preferred_height_band", "mid")).lower().strip()

        default_surfaces = self.cfg.get("placement", {}).get(
            "surface_queries_default",
            ["countertop", "kitchen island", "table", "shelf"],
        )
        llm_surfaces = scene_info.get("candidate_surfaces") or []
        if not isinstance(llm_surfaces, list):
            llm_surfaces = []

        candidate_surfaces = []
        seen = set()
        for s in (llm_surfaces + default_surfaces):
            s2 = str(s).strip().lower()
            if not s2 or s2 in seen:
                continue
            seen.add(s2)
            candidate_surfaces.append(s2)

        obstacles = scene_info.get("obstacles_to_avoid") or []
        if not isinstance(obstacles, list):
            obstacles = []
        obstacles = [str(x).strip().lower() for x in obstacles if str(x).strip()]
        obstacles += ["chair", "stool", "person", "window", "cabinet front", "drawer"]
        obstacles = list(dict.fromkeys(obstacles))

        closeness = self.estimate_depth(scene, debug_dir=debug_dir)

        surf_dets = self.grounding_dino_detect(scene, candidate_surfaces)
        surf_boxes = [d["box"] for d in sorted(surf_dets, key=lambda x: -x["score"])[:12]]
        surf_masks = self.sam_segment_boxes(scene, surf_boxes)

        surfaces: List[Tuple[str, np.ndarray]] = []
        for i, msk in enumerate(surf_masks):
            msk2 = np.squeeze(msk)
            if msk2.ndim != 2:
                continue
            if msk2.shape != (scene.size[1], scene.size[0]):
                continue
            name = candidate_surfaces[min(i, len(candidate_surfaces) - 1)]
            surfaces.append((name, msk2))

        obstacles_mask = np.zeros((scene.size[1], scene.size[0]), dtype=np.uint8)
        if obstacles:
            obs_dets = self.grounding_dino_detect(scene, obstacles, box_threshold=0.30, text_threshold=0.20)
            obs_boxes = [d["box"] for d in sorted(obs_dets, key=lambda x: -x["score"])[:20]]
            obs_masks = self.sam_segment_boxes(scene, obs_boxes)
            for msk in obs_masks:
                msk2 = np.squeeze(msk)
                if msk2.ndim != 2:
                    continue
                if msk2.shape != obstacles_mask.shape:
                    continue
                obstacles_mask = np.maximum(obstacles_mask, (msk2 > 0).astype(np.uint8))

        if self.cfg.get("debug", {}).get("save_intermediates", True):
            dbg_mask = np.squeeze(obstacles_mask).astype(np.uint8)
            if dbg_mask.ndim != 2:
                raise RuntimeError(f"obstacles_mask has bad shape: {dbg_mask.shape}")
            Image.fromarray(dbg_mask * 255, mode="L").save(debug_dir / "scene_obstacles_mask.png")

        place = self.choose_placement(
            scene=scene,
            obj_rgba=obj_rgba,
            surfaces=surfaces,
            obstacles_mask=obstacles_mask,
            closeness=closeness,
            object_size_class=size_class,
            preferred_height_band=preferred_band,
        )
        if place is None:
            raise RuntimeError("No valid top-support placement found. Try broader support queries in config.yaml.")

        scene_np = _pil_to_np_rgb(scene)

        obj_arr = _pil_rgba_to_np_rgba(obj_rgba)
        new_w = max(1, int(round(obj_arr.shape[1] * place.scale)))
        new_h = max(1, int(round(obj_arr.shape[0] * place.scale)))
        obj_scaled = cv2.resize(obj_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        alpha_s = obj_scaled[:, :, 3]
        sx1, sy1, sx2, sy2 = _alpha_bbox(alpha_s, thr=10)
        top_left_x = place.x_center - int((sx1 + sx2) / 2)
        top_left_y = place.y_contact - sy2

        if self.cfg.get("shadow", {}).get("enabled", True):
            sh = self.cfg.get("shadow", {})
            scene_np = _render_contact_shadow(
                scene_rgb=scene_np,
                obj_alpha_u8=alpha_s,
                x=top_left_x,
                y=top_left_y,
                direction=light_dir,
                softness_px=int(sh.get("softness_px", 28)),
                opacity=float(sh.get("opacity", 0.32)),
                squash_y=float(sh.get("squash_y", 0.18)),
                shear_x=float(sh.get("shear_x", 0.12)),
                offset_px=int(sh.get("offset_px", 12)),
            )

        composed = _place_rgba_over_rgb(scene_np, obj_scaled, top_left_x, top_left_y)

        out_path = output_dir / "composite.png"
        _np_rgb_to_pil(composed).save(out_path)

        if bool(self.cfg.get("performance", {}).get("enable_inpaint", False)):
            try:
                from diffusers import AutoPipelineForInpainting
            except Exception as e:
                raise RuntimeError("diffusers not available but enable_inpaint=true") from e

            inpaint_model_id = self.cfg["models"]["inpaint_model"]
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = AutoPipelineForInpainting.from_pretrained(
                inpaint_model_id,
                torch_dtype=dtype,
                variant="fp16" if torch.cuda.is_available() else None,
            )
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            base_img = _np_rgb_to_pil(composed).resize((1024, 1024))
            mask = np.zeros((scene.size[1], scene.size[0]), dtype=np.uint8)

            x1 = max(0, top_left_x)
            y1 = max(0, top_left_y)
            x2 = min(scene.size[0], top_left_x + new_w)
            y2 = min(scene.size[1], top_left_y + new_h)

            if x2 > x1 and y2 > y1:
                ring = np.zeros_like(mask)
                ring[y1:y2, x1:x2] = (
                    alpha_s[(y1 - top_left_y):(y2 - top_left_y), (x1 - top_left_x):(x2 - top_left_x)] > 10
                ).astype(np.uint8) * 255
                ring = cv2.dilate(ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), 1)
                ring2 = cv2.erode(ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), 1)
                ring = cv2.subtract(ring, ring2)

                below = np.zeros_like(mask)
                by1 = min(scene.size[1] - 1, y2)
                by2 = min(scene.size[1], y2 + int(new_h * 0.40))
                bx1 = max(0, x1 - int(new_w * 0.25))
                bx2 = min(scene.size[0], x2 + int(new_w * 0.25))
                below[by1:by2, bx1:bx2] = 255

                mask = np.maximum(ring, below)

            mask_pil = Image.fromarray(mask).resize((1024, 1024))

            steps = int(self.cfg.get("performance", {}).get("inpaint_steps", 20))
            gs = float(self.cfg.get("performance", {}).get("inpaint_guidance_scale", 7.5))
            strength = float(self.cfg.get("performance", {}).get("inpaint_strength", 0.65))

            refined = pipe(
                prompt="photorealistic, natural lighting, realistic shadow, seamless composite",
                image=base_img,
                mask_image=mask_pil,
                num_inference_steps=steps,
                guidance_scale=gs,
                strength=strength,
            ).images[0]

            refined = refined.resize(scene.size)
            refined.save(output_dir / "composite_refined.png")

        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual object insertion with geometry-aware top-surface placement.")
    parser.add_argument("--scene", required=True, type=str, help="Path to the scene image.")
    parser.add_argument("--object-image", required=True, type=str, help="Path to the object image.")
    parser.add_argument("--output", required=True, type=str, help="Output directory.")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device preference.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.config))
    editor = ContextualImageEditor(cfg=cfg, device=args.device)

    out_path = editor.run(
        scene_path=Path(args.scene),
        object_path=Path(args.object_image),
        output_dir=Path(args.output),
    )
    print(str(out_path))


if __name__ == "__main__":
    main()