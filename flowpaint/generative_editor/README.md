# Generative Editor (paper §3.4)

The editor is an SDXL InstructPix2Pix model
([`diffusers/sdxl-instructpix2pix-768`](https://huggingface.co/diffusers/sdxl-instructpix2pix-768))
whose U-Net is fine-tuned on flow-image triplets (original flow image, target evasive
flow image, evasion instruction). The VAE and the text encoders stay frozen.

**Model weights are not released.** Train the editor with `fine_tune.py` and pass
the resulting checkpoint to `inference.py` or `scripts/run_pipeline.py`.

## Fine-tuning

`fine_tune.py` wraps the official diffusers example
[`train_instruct_pix2pix_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py).
It converts the JSONL from [`dataset/`](../../dataset/README.md) into a local Hugging
Face dataset and translates `train_config_example.yaml` into the script's flags.
Download the script once, matching your installed diffusers version, and either put
it next to `fine_tune.py` or point `IP2P_SDXL_TRAIN_SCRIPT` at it.

```bash
export IP2P_SDXL_TRAIN_SCRIPT=/path/to/train_instruct_pix2pix_sdxl.py
accelerate launch flowpaint/generative_editor/fine_tune.py \
  --config flowpaint/generative_editor/train_config_example.yaml \
  --name flowpaint_run --output-dir ./models
```

`train_config_example.yaml` reproduces the paper's training configuration:

| Setting | Value |
|---|---|
| Trained module | U-Net only (VAE and text encoders frozen) |
| Epochs | 10 |
| Learning rate | 1e-5, constant schedule, 500 warmup steps |
| Batch | global batch size 4 with gradient accumulation 4 (batch 1 × accumulation 4 on one GPU) |
| Optimizer and precision | 8-bit Adam, BF16 |
| Conditioning dropout | 0.1 |
| Resolution | 768 |

## Inference

`inference.py` edits flow images with dual classifier-free guidance, using the
paper's defaults: a DDIM sampler with 10 steps, text guidance scale 7.5 and image
guidance scale 1.5.

```bash
# one image
python flowpaint/generative_editor/inference.py \
  --checkpoint ./models/flowpaint_run/checkpoint-<step> \
  --input flow.png --output repainted.png \
  --prompt "Transform the HTTPS network traffic visualization to bypass the GFW"

# a JSONL dataset, with PSNR/SSIM against the targets when scikit-image is installed
python flowpaint/generative_editor/inference.py \
  --checkpoint ./models/flowpaint_run/checkpoint-<step> \
  --dataset-json <dataset>/val.jsonl --dataset-root <dataset> \
  --output-dir validation_results/flowpaint
```

`--checkpoint` accepts a training checkpoint directory that contains `unet/`, which
is loaded on top of `--base-model`, or a full pipeline directory or hub id.
