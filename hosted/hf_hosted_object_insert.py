from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image, ImageFilter


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


# Try one of:
# "Qwen/Qwen-Image-Edit-2511"   # provider fal
# "Qwen/Qwen-Image-Edit-2509"   # provider WaveSpeed
# "FireRedTeam/FireRed-Image-Edit-1.0"  # provider fal
# "black-forest-labs/FLUX.1-Kontext-dev"  # provider fal
DEFAULT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_PROVIDER = "fal-ai"
DEFAULT_OUTPUT = "outputs/output.png"


@dataclass
class Placement:
    x: int
    y: int
    scale: float


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def open_rgba(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def fit_object(
    obj: Image.Image,
    scene_size: Tuple[int, int],
    scale: float,
    max_fraction_of_scene_width: float = 0.28,
) -> Image.Image:
    scene_w, scene_h = scene_size
    max_w = int(scene_w * max_fraction_of_scene_width * scale)

    ow, oh = obj.size
    if ow <= 0 or oh <= 0:
        raise ValueError("Invalid object image dimensions.")

    ratio = max_w / ow
    new_w = max(1, int(round(ow * ratio)))
    new_h = max(1, int(round(oh * ratio)))

    if new_h > int(scene_h * 0.8):
        ratio = (scene_h * 0.8) / oh
        new_w = max(1, int(round(ow * ratio)))
        new_h = max(1, int(round(oh * ratio)))

    return obj.resize((new_w, new_h), Image.LANCZOS)


def alpha_bbox(alpha: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    return alpha.getbbox()


def crop_to_alpha(rgba: Image.Image, pad: int = 0) -> Image.Image:
    alpha = rgba.getchannel("A")
    bbox = alpha_bbox(alpha)
    if bbox is None:
        return rgba

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgba.width, x1 + pad)
    y1 = min(rgba.height, y1 + pad)
    return rgba.crop((x0, y0, x1, y1))


def soften_alpha_edges(rgba: Image.Image, blur_radius: float = 0.6) -> Image.Image:
    rgba = rgba.copy()
    alpha = rgba.getchannel("A").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    rgba.putalpha(alpha)
    return rgba


def default_placement(scene: Image.Image, obj: Image.Image) -> Placement:
    x = (scene.width - obj.width) // 2
    y = int(scene.height * 0.68) - obj.height
    x = max(0, min(scene.width - obj.width, x))
    y = max(0, min(scene.height - obj.height, y))
    return Placement(x=x, y=y, scale=1.0)


def composite_object(
    scene: Image.Image,
    obj: Image.Image,
    placement: Placement,
) -> tuple[Image.Image, Image.Image]:
    scene_rgba = scene.convert("RGBA")
    composed = scene_rgba.copy()

    composed.alpha_composite(obj, (placement.x, placement.y))

    mask = Image.new("L", scene.size, 0)
    mask.paste(obj.getchannel("A"), (placement.x, placement.y))

    return composed.convert("RGB"), mask


def save_debug_images(
    composed: Image.Image,
    mask: Image.Image,
    debug_dir: str | Path,
) -> None:
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    composed.save(debug_dir / "debug_composed.png")
    mask.save(debug_dir / "debug_mask.png")


def build_prompt(
    object_label: str,
    user_prompt: Optional[str],
    preserve_scene: bool,
) -> str:
    base = (
        f"Blend the inserted {object_label} naturally into the scene. "
        f"Preserve the {object_label} exactly, including shape, branding, logo, text, colors, "
        f"surface texture, and fine details. "
    )

    if preserve_scene:
        base += (
            "Keep the rest of the scene unchanged. "
            "Do not alter background geometry, furniture, walls, counters, lighting setup, "
            "or add/remove unrelated objects. "
        )

    if user_prompt and user_prompt.strip():
        base += user_prompt.strip()

    return base


def make_client(provider: str) -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in your environment.")
    return InferenceClient(provider=provider, api_key=HF_TOKEN)


def run_hosted_edit(
    image: Image.Image,
    model: str,
    provider: str,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
) -> Image.Image:
    client = make_client(provider)

    image_bytes = image_to_png_bytes(image)

    result = client.image_to_image(
        image=image_bytes,
        prompt=prompt,
        model=model,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    if not isinstance(result, Image.Image):
        raise RuntimeError("Hosted model did not return a PIL image.")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert an object into a scene using a hosted Hugging Face image-edit model."
    )

    parser.add_argument("--scene", required=True, help="Path to the scene image")
    parser.add_argument("--object-image", required=True, help="Path to the object image (prefer transparent PNG)")
    parser.add_argument("--object-label", required=True, help="Object label, e.g. 'ginger ale can'")

    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--debug-dir", default="outputs/debug", help="Directory for debug images")

    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="HF inference provider")

    parser.add_argument("--prompt", default="", help="Optional extra prompt")
    parser.add_argument("--no-preserve-scene", action="store_true", help="Allow broader scene edits")

    parser.add_argument("--x", type=int, default=None, help="Top-left x placement in scene pixels")
    parser.add_argument("--y", type=int, default=None, help="Top-left y placement in scene pixels")
    parser.add_argument("--scale", type=float, default=1.0, help="Object scale multiplier")

    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")

    parser.add_argument(
        "--alpha-blur",
        type=float,
        default=0.6,
        help="Light blur on object alpha edges before compositing",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene = open_rgb(args.scene)
    obj = open_rgba(args.object_image)
    obj = crop_to_alpha(obj, pad=1)
    obj = fit_object(obj, scene.size, scale=args.scale)
    obj = soften_alpha_edges(obj, blur_radius=args.alpha_blur)

    if args.x is not None and args.y is not None:
        placement = Placement(
            x=max(0, min(scene.width - obj.width, args.x)),
            y=max(0, min(scene.height - obj.height, args.y)),
            scale=args.scale,
        )
    else:
        placement = default_placement(scene, obj)

    composed, mask = composite_object(scene, obj, placement)
    save_debug_images(composed, mask, args.debug_dir)

    prompt = build_prompt(
        object_label=args.object_label,
        user_prompt=args.prompt,
        preserve_scene=not args.no_preserve_scene,
    )

    result = run_hosted_edit(
        image=composed,
        model=args.model,
        provider=args.provider,
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
    )

    ensure_dir(args.output)
    result.save(args.output)

    print(f"Saved result to: {args.output}")
    print(f"Saved debug images to: {args.debug_dir}")
    print(f"Model: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Placement: x={placement.x}, y={placement.y}, scale={placement.scale}")
    print(f"Prompt: {prompt}")


if __name__ == "__main__":
    main()