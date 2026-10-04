#!/usr/bin/env python3
"""Supervised fine-tuning entry point for the FlowPaint Generative Editor (paper Sec. 3.4).

FlowPaint fine-tunes the U-Net of an SDXL InstructPix2Pix backbone
(``diffusers/sdxl-instructpix2pix-768``) on flow-image triplets
(original flow image, target evasive flow image, user instruction) while keeping
the VAE and text encoders frozen. This module is a thin, config-driven wrapper
around the official Hugging Face ``diffusers`` example script
``train_instruct_pix2pix_sdxl.py``: it converts the JSONL produced by
``dataset/prepare_instruct_dataset.py`` into a local Hugging Face dataset,
translates a single YAML file into the CLI flags that script expects, and
launches it in-process.

The training script itself is *not* vendored here (it is version-coupled to your
installed ``diffusers`` release and carries its own Apache-2.0 license). Obtain it
from the diffusers repository:

    https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py

and either drop it next to this file, or point ``IP2P_SDXL_TRAIN_SCRIPT`` at it:

    export IP2P_SDXL_TRAIN_SCRIPT=/path/to/train_instruct_pix2pix_sdxl.py

Example:

    accelerate launch flowpaint/generative_editor/fine_tune.py \\
        --config flowpaint/generative_editor/train_config_example.yaml \\
        --name flowpaint_run --output-dir ./models --logdir ./logs
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent


def _resolve_train_script() -> Path:
    """Locate the diffusers SDXL InstructPix2Pix training script."""
    env = os.environ.get("IP2P_SDXL_TRAIN_SCRIPT")
    if env:
        return Path(env).expanduser().resolve()
    return HERE / "train_instruct_pix2pix_sdxl.py"


def _resolve_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _required(section: Dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section or section[key] is None:
        raise ValueError(f"Missing `{section_name}.{key}` in training config")
    return section[key]


def _add_arg(cli: List[str], flag: str, value: Optional[Any]) -> None:
    if value is None:
        return
    cli.extend([flag, str(value)])


def _add_flag(cli: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cli.append(flag)


def _run_python_script(script_path: Path, cli_args: List[str]) -> None:
    if not script_path.exists():
        raise FileNotFoundError(
            f"SDXL training script not found at {script_path}.\n"
            "Download it from the diffusers repository and set IP2P_SDXL_TRAIN_SCRIPT, "
            "or place it next to fine_tune.py. See this file's docstring for the URL."
        )
    script_dir = script_path.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    sys.argv = [script_path.name, *cli_args]
    runpy.run_path(str(script_path), run_name="__main__")


def prepare_hf_dataset(
    jsonl_path: Path,
    image_root: Path,
    out_dir: Path,
    columns: tuple[str, str, str],
) -> Path:
    """Convert the FlowPaint JSONL into a parquet dataset the diffusers script can load.

    ``columns`` names the original image, edited image and prompt fields. Image
    paths in the JSONL are resolved relative to ``image_root``.
    """
    from datasets import Dataset, Image as ImageFeature

    original_col, edited_col, prompt_col = columns
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            records.append({
                original_col: str((image_root / entry[original_col]).resolve()),
                edited_col: str((image_root / entry[edited_col]).resolve()),
                prompt_col: entry[prompt_col],
            })
    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column(original_col, ImageFeature()).cast_column(edited_col, ImageFeature())
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(data_dir / "train-00000-of-00001.parquet"))
    return out_dir


def load_training_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def compute_run_dirs(args: argparse.Namespace, config: Dict[str, Any]) -> tuple[Path, Path]:
    paths_cfg: Dict[str, Any] = config.get("paths", {})
    output_base = paths_cfg.get("output_dir") or args.output_dir
    output_dir = _resolve_path(output_base) / args.name

    if args.logdir is not None:
        logging_dir = _resolve_path(args.logdir) / args.name
    else:
        logging_subdir = paths_cfg.get("logging_subdir") or "runs"
        logging_dir = output_dir / logging_subdir

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, logging_dir


def build_sdxl_cli_args(
    args: argparse.Namespace,
    config: Dict[str, Any],
    output_dir: Path,
    logging_dir: Path,
) -> List[str]:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    pretrained_model = _required(model_cfg, "model", "pretrained_model_name_or_path")
    dataset_path = _resolve_path(_required(data_cfg, "data", "dataset_path"))
    image_root = _resolve_path(data_cfg.get("image_root") or dataset_path.parent)
    columns = (
        data_cfg.get("original_image_column", "original_image"),
        data_cfg.get("edited_image_column", "edited_image"),
        data_cfg.get("edit_prompt_column", "edit_prompt"),
    )
    hf_dataset = prepare_hf_dataset(dataset_path, image_root, output_dir / "hf_dataset", columns)

    cli: List[str] = [
        "--pretrained_model_name_or_path", str(pretrained_model),
        "--dataset_name", str(hf_dataset),
        "--original_image_column", columns[0],
        "--edited_image_column", columns[1],
        "--edit_prompt_column", columns[2],
        "--output_dir", str(output_dir),
        "--logging_dir", str(logging_dir),
        "--train_batch_size", str(_required(data_cfg, "data", "train_batch_size")),
        "--seed", str(args.seed),
    ]

    _add_arg(cli, "--pretrained_vae_model_name_or_path", model_cfg.get("pretrained_vae_model_name_or_path"))
    _add_arg(cli, "--revision", model_cfg.get("revision"))
    _add_arg(cli, "--non_ema_revision", model_cfg.get("non_ema_revision"))
    _add_arg(cli, "--cache_dir", model_cfg.get("cache_dir"))

    _add_arg(cli, "--max_train_samples", data_cfg.get("max_train_samples"))
    _add_arg(cli, "--resolution", data_cfg.get("resolution"))
    _add_arg(cli, "--dataloader_num_workers", data_cfg.get("dataloader_num_workers"))
    _add_flag(cli, "--center_crop", bool(data_cfg.get("center_crop", False)))
    _add_flag(cli, "--random_flip", bool(data_cfg.get("random_flip", False)))

    _add_arg(cli, "--num_train_epochs", training_cfg.get("num_train_epochs"))
    max_steps = args.max_steps or training_cfg.get("max_train_steps")
    _add_arg(cli, "--max_train_steps", max_steps)

    grad_accum = args.accumulate_grad_batches or training_cfg.get("gradient_accumulation_steps")
    _add_arg(cli, "--gradient_accumulation_steps", grad_accum)
    _add_flag(cli, "--gradient_checkpointing", bool(training_cfg.get("gradient_checkpointing", False)))

    _add_arg(cli, "--learning_rate", training_cfg.get("learning_rate"))
    _add_arg(cli, "--lr_scheduler", training_cfg.get("lr_scheduler"))
    _add_arg(cli, "--lr_warmup_steps", training_cfg.get("lr_warmup_steps"))
    _add_arg(cli, "--conditioning_dropout_prob", training_cfg.get("conditioning_dropout_prob"))

    precision = args.precision or training_cfg.get("mixed_precision")
    _add_arg(cli, "--mixed_precision", precision)

    scale_lr = args.scale_lr or training_cfg.get("scale_lr", False)
    _add_flag(cli, "--scale_lr", bool(scale_lr))

    _add_flag(cli, "--use_8bit_adam", bool(training_cfg.get("use_8bit_adam", False)))
    _add_flag(cli, "--allow_tf32", bool(training_cfg.get("allow_tf32", False)))
    _add_flag(cli, "--use_ema", bool(model_cfg.get("use_ema", False)))
    _add_flag(
        cli,
        "--enable_xformers_memory_efficient_attention",
        bool(model_cfg.get("enable_xformers_memory_efficient_attention", False)),
    )

    _add_arg(cli, "--adam_beta1", training_cfg.get("adam_beta1"))
    _add_arg(cli, "--adam_beta2", training_cfg.get("adam_beta2"))
    _add_arg(cli, "--adam_weight_decay", training_cfg.get("adam_weight_decay"))
    _add_arg(cli, "--adam_epsilon", training_cfg.get("adam_epsilon"))
    _add_arg(cli, "--max_grad_norm", training_cfg.get("max_grad_norm"))

    resume = args.resume or training_cfg.get("resume_from_checkpoint")
    _add_arg(cli, "--resume_from_checkpoint", resume)

    _add_arg(cli, "--report_to", training_cfg.get("report_to"))
    _add_arg(cli, "--checkpointing_steps", training_cfg.get("checkpointing_steps"))
    _add_arg(cli, "--checkpoints_total_limit", training_cfg.get("checkpoints_total_limit"))

    _add_arg(cli, "--validation_prompt", training_cfg.get("validation_prompt"))
    _add_arg(cli, "--val_image_url_or_path", training_cfg.get("val_image_url_or_path"))
    _add_arg(cli, "--num_validation_images", training_cfg.get("num_validation_images"))
    _add_arg(cli, "--validation_steps", training_cfg.get("validation_steps"))

    return cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the FlowPaint Generative Editor (SDXL InstructPix2Pix U-Net).",
    )
    parser.add_argument("--config", type=Path, default=HERE / "train_config_example.yaml",
                        help="Path to the training YAML config.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"),
                        help="Directory root where checkpoints will be written.")
    parser.add_argument("--logdir", type=Path,
                        help="Optional root directory for logs (default <output_dir>/<logging_subdir>).")
    parser.add_argument("--name", type=str, default="flowpaint-sdxl",
                        help="Run name appended to the output/log directories.")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU list exposed via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--seed", type=int, default=23, help="Global training seed.")
    parser.add_argument("--resume", type=Path, help="Resume from this checkpoint (overrides config).")
    parser.add_argument("--precision", type=str, choices=["no", "fp16", "bf16"],
                        help="Override the accelerate mixed-precision setting.")
    parser.add_argument("--scale-lr", action="store_true", help="Force LR scaling regardless of config.")
    parser.add_argument("--accumulate-grad-batches", type=int, help="Override gradient accumulation steps.")
    parser.add_argument("--max-steps", type=int, help="Stop training after this many optimizer steps.")

    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output_dir = _resolve_path(args.output_dir)
    if args.logdir is not None:
        args.logdir = _resolve_path(args.logdir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    config = load_training_config(args.config)
    output_dir, logging_dir = compute_run_dirs(args, config)
    cli_args = build_sdxl_cli_args(args, config, output_dir, logging_dir)
    _run_python_script(_resolve_train_script(), cli_args)


if __name__ == "__main__":
    main()
