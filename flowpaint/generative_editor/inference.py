#!/usr/bin/env python3
"""FlowPaint Generative Editor inference (paper Sec. 3.4).

Runs instruction-guided editing with the fine-tuned SDXL InstructPix2Pix U-Net,
using the Dual Classifier-Free Guidance strategy the paper describes:

    * DDIM sampler, 10 inference steps;
    * Text Guidance Scale  = 7.5  (adherence to the evasion prompt);
    * Image Guidance Scale = 1.5  (adherence to the original flow structure).

Two modes:

  1. Single flow image -> single repainted image:
        python inference.py --checkpoint <ckpt> --input flow.png --prompt "..." --output out.png

  2. A JSONL dataset of {original_image, edited_image, edit_prompt} records
     (as produced by dataset/prepare_instruct_dataset.py); optional PSNR/SSIM
     against the target images when scikit-image is installed:
        python inference.py --checkpoint <ckpt> --dataset-json test.jsonl \\
            --dataset-root <root> --output-dir validation_results/flowpaint_sdxl

``--checkpoint`` accepts either a Hugging Face repo id or a local directory. If a
local directory contains a ``unet/`` subfolder (an accelerate/diffusers training
checkpoint such as ``checkpoint-<step>``), only that U-Net is loaded on top of the
SDXL InstructPix2Pix base defined by ``--base-model``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    StableDiffusionXLInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from PIL import Image

DEFAULT_BASE_MODEL = "diffusers/sdxl-instructpix2pix-768"


def build_pipeline(
    checkpoint: str,
    base_model: str,
    device: str,
) -> StableDiffusionXLInstructPix2PixPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    ckpt_path = Path(checkpoint)

    if ckpt_path.exists() and (ckpt_path / "unet").exists():
        # Local accelerate/diffusers checkpoint: load only the fine-tuned U-Net
        # on top of the SDXL InstructPix2Pix base.
        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            base_model, torch_dtype=dtype
        )
        pipeline.unet = UNet2DConditionModel.from_pretrained(
            ckpt_path / "unet", torch_dtype=dtype
        )
    else:
        # A fully self-contained pipeline directory or hub repo id.
        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            checkpoint, torch_dtype=dtype
        )

    # Paper uses a DDIM sampler with 10 steps.
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.eval()
    return pipeline


def repaint(
    pipeline: StableDiffusionXLInstructPix2PixPipeline,
    image: Image.Image,
    prompt: str,
    *,
    num_steps: int,
    text_guidance_scale: float,
    image_guidance_scale: float,
    generator: torch.Generator,
) -> Image.Image:
    result = pipeline(
        prompt=prompt,
        image=image.convert("RGB"),
        num_inference_steps=num_steps,
        guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        generator=generator,
    )
    return result.images[0]


def _try_metrics(target: Image.Image, generated: Image.Image) -> Optional[dict]:
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError:
        return None
    t = np.array(target.convert("RGB"))
    g = np.array(generated.convert("RGB"))
    return {
        "psnr": float(peak_signal_noise_ratio(t, g, data_range=255)),
        "ssim": float(structural_similarity(t, g, channel_axis=2, data_range=255)),
    }


def run_single(args: argparse.Namespace, pipeline, generator) -> None:
    image = Image.open(args.input)
    out = repaint(
        pipeline, image, args.prompt,
        num_steps=args.num_steps,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        generator=generator,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"[ok] wrote {out_path}")


def run_dataset(args: argparse.Namespace, pipeline, generator) -> None:
    from tqdm import tqdm

    root = args.dataset_root
    records: List[dict] = []
    with args.dataset_json.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_index"] = idx
            records.append(rec)
    if args.max_samples:
        records = records[: args.max_samples]

    out_dir = args.output_dir.resolve()
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    psnr_scores, ssim_scores = [], []
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as mfile:
        for pos, rec in enumerate(tqdm(records, desc="Evaluating", unit="sample")):
            prompt = rec.get("edit_prompt") or rec.get("edit_instruction") or ""
            original = Image.open(root / rec["original_image"])
            generated = repaint(
                pipeline, original, prompt,
                num_steps=args.num_steps,
                text_guidance_scale=args.text_guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                generator=generator,
            )
            stem = f"{pos:05d}_{rec['_index']:05d}"
            generated.save(gen_dir / f"{stem}.png")

            entry = {"index": rec["_index"], "generated_image": str(gen_dir / f"{stem}.png")}
            if "edited_image" in rec and (root / rec["edited_image"]).exists():
                metrics = _try_metrics(Image.open(root / rec["edited_image"]), generated)
                if metrics:
                    entry.update(metrics)
                    psnr_scores.append(metrics["psnr"])
                    ssim_scores.append(metrics["ssim"])
            mfile.write(json.dumps(entry) + "\n")

    summary = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "samples_evaluated": len(records),
        "text_guidance_scale": args.text_guidance_scale,
        "image_guidance_scale": args.image_guidance_scale,
        "steps": args.num_steps,
        "seed": args.seed,
        "mean_psnr": float(np.mean(psnr_scores)) if psnr_scores else None,
        "mean_ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {out_dir/'summary.json'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Fine-tuned checkpoint: hub repo id, pipeline dir, or a dir containing unet/.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help="SDXL InstructPix2Pix base used when loading a unet-only checkpoint.")
    p.add_argument("--device", choices=["cuda", "cpu"],
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Single-image mode
    p.add_argument("--input", type=Path, help="Single input flow PNG.")
    p.add_argument("--prompt", type=str, help="Evasion instruction for single-image mode.")
    p.add_argument("--output", type=Path, help="Output PNG for single-image mode.")

    # Dataset mode
    p.add_argument("--dataset-json", type=Path, help="JSONL of original/edited/prompt records.")
    p.add_argument("--dataset-root", type=Path, help="Root directory the JSONL paths are relative to.")
    p.add_argument("--output-dir", type=Path, default=Path("validation_results/flowpaint_sdxl"))
    p.add_argument("--max-samples", type=int, help="Optional cap on dataset-mode samples.")

    # Sampling (paper defaults)
    p.add_argument("--num-steps", type=int, default=10, help="DDIM inference steps (paper: 10).")
    p.add_argument("--text-guidance-scale", type=float, default=7.5, help="Text CFG scale (paper: 7.5).")
    p.add_argument("--image-guidance-scale", type=float, default=1.5, help="Image CFG scale (paper: 1.5).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args.checkpoint, args.base_model, args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    if args.dataset_json is not None:
        if args.dataset_root is None:
            raise SystemExit("--dataset-root is required in dataset mode")
        run_dataset(args, pipeline, generator)
    elif args.input is not None:
        if not (args.prompt and args.output):
            raise SystemExit("single-image mode requires --input, --prompt and --output")
        run_single(args, pipeline, generator)
    else:
        raise SystemExit("provide either --input (single image) or --dataset-json (batch)")


if __name__ == "__main__":
    main()
