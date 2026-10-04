# Dataset Construction (paper §4)

The Generative Editor is fine-tuned on flow-image triplets: an original flow image, a
target evasive flow image and an evasion instruction. In the paper, benign traffic
from [Stratosphere IPS](https://www.stratosphereips.org/datasets-overview) is paired
with evasive counterparts from two tools, giving 6,712 paired samples:

- **Geneva**: the strategy
  `[TCP:flags:PA]-duplicate(tamper{TCP:dataofs:replace:10}(tamper{TCP:chksum:corrupt},),)-|`,
  which reaches a 98% bypass rate against the GFW. `geneva_strategy_pa_processor.py`
  applies it to benign PCAPs.
- **UPGen**: the six public Protocol Specification Files (PSFs). These traces come
  from the UPGen release and are not generated here.

## 1. Synthesize evasive targets

```bash
python dataset/geneva_strategy_pa_processor.py --input benign_pcaps/ --output geneva_pcaps/
```

Each `<name>.pcap` produces `<name>_obfuscated.pcap`.

## 2. Encode both sides as flow images

```bash
python -m flowpaint.traffic_encoder.pcap2nprint --pcap-dir benign_pcaps/ -o nprint/
python -m flowpaint.traffic_encoder.pcap2nprint --pcap-dir geneva_pcaps/ -o nprint/
python -m flowpaint.traffic_encoder.nprint_to_png -i nprint/ -o png/
```

Every evasive image must be named `<base>_obfuscated.png` next to its original
`<base>.png`. Rename the blocks if your file names differ.

## 3. Build the InstructPix2Pix JSONL

```bash
python dataset/prepare_instruct_dataset.py --image-dir png/ --output-dir flowpaint_dataset/ --target gfw
```

This writes `train/`, `val/`, `train.jsonl` and `val.jsonl` with
`{original_image, edited_image, edit_prompt, ...}` records. `--target` selects the
paper's prompt for every pair:

| `--target` | Prompt |
|---|---|
| `gfw` (Geneva) | `Transform the HTTPS network traffic visualization to bypass the GFW` |
| `classifier` (UPGen) | `Transform the HTTPS network traffic visualization to evade Deep learning/Machine learning classifiers` |

`--instruction-mode varied` or `augmented` replaces the fixed prompt with paraphrases
from `augment_instructions.py`.

Set `data.dataset_path` in
[`train_config_example.yaml`](../flowpaint/generative_editor/train_config_example.yaml)
to `flowpaint_dataset/train.jsonl` and `data.image_root` to `flowpaint_dataset/`.
[`sample_data/fine_tune_pairs/`](../sample_data/fine_tune_pairs) holds a one-record
example of the format.
