from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"Required module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)

    import sys
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_torch() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _clear_module_cuda_state(module: ModuleType) -> None:
    cleanup_fn = getattr(module, "cleanup_models", None)
    if callable(cleanup_fn):
        try:
            cleanup_fn()
        except Exception:
            pass

    for name in dir(module):
        lowered = name.lower()
        if not name.startswith("_"):
            continue
        if not any(
            token in lowered
            for token in ("model", "processor", "pipe", "pipeline", "detector", "segment", "depth", "vlm")
        ):
            continue
        try:
            setattr(module, name, None)
        except Exception:
            pass

    _cleanup_torch()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end object compositing pipeline: object understanding -> extraction -> scene understanding -> placement."
    )
    parser.add_argument("--scene", required=True, type=str, help="Path to the scene image.")
    parser.add_argument("--object-image", required=True, type=str, help="Path to the object/source image.")
    parser.add_argument("--object-label", required=True, type=str, help="Object label, e.g. 'vase with flowers'.")
    parser.add_argument("--output-root", required=True, type=str, help="Root output directory for all pipeline stages.")

    parser.add_argument("--object-understanding-config", required=True, type=str, help="Path to object understanding YAML config.")
    parser.add_argument("--extractor-config", required=True, type=str, help="Path to extractor YAML config.")
    parser.add_argument("--scene-understanding-config", required=True, type=str, help="Path to scene understanding YAML config.")
    parser.add_argument("--placement-config", required=True, type=str, help="Path to placement YAML config.")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], type=str, help="Device preference.")

    parser.add_argument(
        "--object-understanding-script",
        default="../object-understanding/object_understanding.py",
        type=str,
        help="Path to object_understanding.py",
    )
    parser.add_argument(
        "--extractor-script",
        default="../extractor/object_extractor.py",
        type=str,
        help="Path to object_extractor.py",
    )
    parser.add_argument(
        "--scene-understanding-script",
        default="../scene-understanding/scene_understanding.py",
        type=str,
        help="Path to scene_understanding.py",
    )
    parser.add_argument(
        "--placement-script",
        default="../placement/placement_reasoning.py",
        type=str,
        help="Path to placement_reasoning.py",
    )

    args = parser.parse_args()

    scene_path = Path(args.scene)
    object_image_path = Path(args.object_image)
    output_root = Path(args.output_root)

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene image not found: {scene_path}")
    if not object_image_path.exists():
        raise FileNotFoundError(f"Object image not found: {object_image_path}")

    _ensure_dir(output_root)

    object_understanding_out = output_root / "object_understanding"
    extracted_out = output_root / "extracted"
    scene_understanding_out = output_root / "scene_understanding"
    placement_out = output_root / "placement"

    for p in [object_understanding_out, extracted_out, scene_understanding_out, placement_out]:
        _ensure_dir(p)

    root_dir = Path(__file__).resolve().parent

    object_understanding_mod = _load_module(
        "object_understanding_mod",
        (root_dir / args.object_understanding_script).resolve(),
    )
    extractor_mod = _load_module(
        "object_extractor_mod",
        (root_dir / args.extractor_script).resolve(),
    )
    scene_understanding_mod = _load_module(
        "scene_understanding_mod",
        (root_dir / args.scene_understanding_script).resolve(),
    )
    placement_mod = _load_module(
        "placement_reasoning_mod",
        (root_dir / args.placement_script).resolve(),
    )

    # ------------------------------------------------------------------
    # Stage 1: Object understanding
    # ------------------------------------------------------------------
    object_understanding_cfg = object_understanding_mod._read_yaml(Path(args.object_understanding_config))
    object_understanding_result = object_understanding_mod.understand_object_from_path(
        image_path=object_image_path,
        object_label=args.object_label,
        cfg=object_understanding_cfg,
        device=args.device,
        output_dir=object_understanding_out,
    )

    object_understanding_json = (
        Path(object_understanding_result.json_path)
        if object_understanding_result.json_path is not None
        else object_understanding_out / "object_understanding.json"
    )
    object_understanding_payload = _load_json_if_exists(object_understanding_json)

    del object_understanding_result
    del object_understanding_cfg
    _clear_module_cuda_state(object_understanding_mod)

    # ------------------------------------------------------------------
    # Stage 2: Object extraction
    # ------------------------------------------------------------------
    extractor_cfg = extractor_mod._read_yaml(Path(args.extractor_config))
    extracted = extractor_mod.extract_object_from_path(
        image_path=object_image_path,
        object_label=args.object_label,
        cfg=extractor_cfg,
        device=args.device,
        debug_dir=extracted_out / "debug",
        object_understanding=object_understanding_payload,
    )

    object_rgba_path = extracted_out / "object_rgba.png"
    object_alpha_path = extracted_out / "object_alpha.png"
    object_mask_path = extracted_out / "object_mask.png"
    object_meta_path = extracted_out / "meta.yaml"

    extracted.rgba.save(object_rgba_path)
    extracted.alpha.save(object_alpha_path)
    extracted.mask_binary.save(object_mask_path)

    import yaml  # local import to keep dependency surface minimal at top

    with object_meta_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "method": extracted.method,
                "label": extracted.label,
                "bbox_xyxy": extracted.bbox_xyxy,
                "debug": extracted.debug,
            },
            f,
            sort_keys=False,
        )

    del extracted
    del extractor_cfg
    _clear_module_cuda_state(extractor_mod)

    # ------------------------------------------------------------------
    # Stage 3: Scene understanding
    # ------------------------------------------------------------------
    scene_understanding_cfg = scene_understanding_mod._read_yaml(Path(args.scene_understanding_config))
    scene_engine = scene_understanding_mod.SceneUnderstandingEngine(
        cfg=scene_understanding_cfg,
        device=args.device,
    )
    scene_result = scene_engine.analyze(
        scene_path=scene_path,
        output_dir=scene_understanding_out,
    )

    if scene_result.support_json_path is None:
        raise RuntimeError("Scene understanding did not produce scene_understanding.json")

    scene_understanding_json = Path(scene_result.support_json_path)

    del scene_engine
    del scene_result
    del scene_understanding_cfg
    _clear_module_cuda_state(scene_understanding_mod)

    # ------------------------------------------------------------------
    # Stage 4: Placement
    # ------------------------------------------------------------------
    placement_cfg = placement_mod._read_yaml(Path(args.placement_config))
    placement_result = placement_mod.place_object_in_scene(
        scene_path=scene_path,
        scene_understanding_json=scene_understanding_json,
        object_rgba_path=object_rgba_path,
        output_dir=placement_out,
        cfg=placement_cfg,
        object_understanding=object_understanding_payload,
    )

    del placement_cfg
    _clear_module_cuda_state(placement_mod)

    final_payload = {
        "composite_path": placement_result.composite_path,
        "placement_json_path": placement_result.placement_json_path,
        "debug_dir": placement_result.debug_dir,
        "object_understanding_json": str(object_understanding_json),
        "object_rgba_path": str(object_rgba_path),
        "scene_understanding_json": str(scene_understanding_json),
        "stage_output_dirs": {
            "object_understanding": str(object_understanding_out),
            "extracted": str(extracted_out),
            "scene_understanding": str(scene_understanding_out),
            "placement": str(placement_out),
        },
    }

    with (output_root / "pipeline_result.json").open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    if placement_result.composite_path is None:
        raise RuntimeError("Pipeline finished, but no composite output path was produced.")

    print(placement_result.composite_path)


if __name__ == "__main__":
    main()