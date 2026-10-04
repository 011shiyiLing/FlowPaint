"""Build the InstructPix2Pix training set from paired flow images (paper Sec. 4).

Each record pairs an original flow image ``<base>.png`` with its evasive target
``<base>_obfuscated.png`` and an evasion instruction. By default every record uses
the paper's prompt for the chosen target:

  * ``gfw``        (Geneva targets): "Transform the HTTPS network traffic visualization to bypass the GFW"
  * ``classifier`` (UPGen targets):  "Transform the HTTPS network traffic visualization to evade
                                       Deep learning/Machine learning classifiers"

Example:

    python dataset/prepare_instruct_dataset.py --image-dir <png_dir> --output-dir <dataset_dir> --target gfw
"""

import argparse
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random

PAPER_PROMPTS = {
    "gfw": "Transform the HTTPS network traffic visualization to bypass the GFW",
    "classifier": "Transform the HTTPS network traffic visualization to evade Deep learning/Machine learning classifiers",
}


def parse_obfuscation_strategy(strategy: str) -> str:
    """
    Generate effect-based instruction for the Geneva obfuscation strategy.
    Strategy: "[TCP:flags:PA]-duplicate(tamper{TCP:dataofs:replace:10}(tamper{TCP:chksum:corrupt},),)-|"

    This describes the desired censorship bypass effect rather than technical implementation.
    Based on Geneva CCS'19 paper - achieves 98% bypass rate for China's GFW, 100% for Kazakhstan.
    """

    # Effect-based instructions focusing on censorship bypass capabilities
    effect_based_instructions = [
        "Transform traffic to bypass TCP-based censorship systems against Chinese and Kazakh firewalls",
        "Generate network flow that evades Great Firewall detection",
        "Create censorship-resistant traffic that bypasses Kazakhstan's DPI with 100% effectiveness",
        "Obfuscate packets to defeat TCP Control Block tracking used by Chinese and Kazakhstani censors",
        "Apply obfuscated strategy for high bypass rate against China and Kazakhstan censorship",
        "Transform flow to evade state-level censorship through TCP header manipulation bypass technique",
        "Generate traffic resistant to GFW pattern matching and connection tracking mechanisms",
    ]

    return random.choice(effect_based_instructions)

def find_image_pairs(image_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Find all paired images (normal and obfuscated) in the dataset.
    Returns list of tuples: (base_name, normal_path, obfuscated_path)
    """
    pairs = []
    processed = set()

    for file in sorted(image_dir.iterdir()):
        if file.suffix != '.png':
            continue

        filename = file.stem

        if '_obfuscated' in filename:
            base_name = filename.replace('_obfuscated', '')
            if base_name in processed:
                continue

            normal_file = image_dir / f"{base_name}.png"
            obfuscated_file = file

            if normal_file.exists():
                pairs.append((base_name, str(normal_file), str(obfuscated_file)))
                processed.add(base_name)

    return pairs

def create_instruct_pix2pix_dataset(
    image_dir: str,
    output_dir: str,
    train_split: float = 0.9,
    obfuscation_strategy: str = "[TCP:flags:PA]-duplicate(tamper{TCP:dataofs:replace:10}(tamper{TCP:chksum:corrupt},),)-|",
    instruction_mode: str = "consistent",  # Options: "consistent", "varied", "augmented"
    target: str = "gfw",
):
    """
    Create dataset in InstructPix2Pix format with proper directory structure and metadata.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    print(f"Scanning for image pairs in {image_dir}")
    pairs = find_image_pairs(image_dir)
    print(f"Found {len(pairs)} image pairs")

    random.shuffle(pairs)

    train_size = int(len(pairs) * train_split)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")

    train_metadata = []
    val_metadata = []

    def process_split(pairs_list, split_dir, metadata_list, split_name):
        for idx, (base_name, normal_path, obfuscated_path) in enumerate(pairs_list):
            if idx % 100 == 0:
                print(f"Processing {split_name} pair {idx}/{len(pairs_list)}")

            new_normal_name = f"{idx:05d}_original.png"
            new_obfuscated_name = f"{idx:05d}_edited.png"

            shutil.copy2(normal_path, split_dir / new_normal_name)
            shutil.copy2(obfuscated_path, split_dir / new_obfuscated_name)

            input_caption = "HTTPS network traffic visualization"

            if instruction_mode == "consistent":
                # The paper's prompt for this target, shared by all pairs
                edit_instruction = PAPER_PROMPTS[target]
            elif instruction_mode == "varied":
                # Use varied instructions for better generalization
                edit_instruction = parse_obfuscation_strategy(obfuscation_strategy)
            elif instruction_mode == "augmented":
                # For maximum generalization: create multiple entries with different instructions
                # This would require modifying the loop to duplicate entries
                # For now, use highly varied instructions
                from augment_instructions import generate_diverse_instructions
                diverse_instructions = generate_diverse_instructions()
                edit_instruction = random.choice(diverse_instructions)
            else:
                raise ValueError(f"Unknown instruction_mode {instruction_mode!r}")

            # Output caption describes the achieved effect
            output_caption = f"{input_caption} {'bypass GFW' if target == 'gfw' else 'evade ML classifiers'}."

            metadata_entry = {
                "image_id": idx,
                "original_image": new_normal_name,
                "edited_image": new_obfuscated_name,
                "input_caption": input_caption,
                "edit_instruction": edit_instruction,
                "output_caption": output_caption,
                "original_filename": base_name,
                "obfuscation_strategy": obfuscation_strategy
            }

            metadata_list.append(metadata_entry)

    print("\nProcessing training set...")
    process_split(train_pairs, train_dir, train_metadata, "train")

    print("\nProcessing validation set...")
    process_split(val_pairs, val_dir, val_metadata, "val")

    with open(output_dir / "train_metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)

    with open(output_dir / "val_metadata.json", 'w') as f:
        json.dump(val_metadata, f, indent=2)

    dataset_info = {
        "dataset_name": "Network Traffic Obfuscation Dataset",
        "total_pairs": len(pairs),
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "obfuscation_strategy": obfuscation_strategy,
        "description": "Paired dataset of normal and obfuscated network traffic visualizations for InstructPix2Pix training"
    }

    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nDataset created successfully at {output_dir}")
    print(f"Train metadata: {output_dir / 'train_metadata.json'}")
    print(f"Val metadata: {output_dir / 'val_metadata.json'}")
    print(f"Dataset info: {output_dir / 'dataset_info.json'}")

def create_simplified_jsonl_format(output_dir: str, jsonl_output_dir: str):
    """
    Create JSONL format files that are commonly used for InstructPix2Pix training.
    """
    output_dir = Path(output_dir)
    jsonl_output_dir = Path(jsonl_output_dir)
    jsonl_output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        metadata_file = output_dir / f"{split}_metadata.json"
        if not metadata_file.exists():
            print(f"Warning: {metadata_file} not found")
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        jsonl_file = jsonl_output_dir / f"{split}.jsonl"

        with open(jsonl_file, 'w') as f:
            for entry in metadata:
                jsonl_entry = {
                    "original_image": f"{split}/{entry['original_image']}",
                    "edited_image": f"{split}/{entry['edited_image']}",
                    "edit_prompt": entry['edit_instruction'],
                    "input_caption": entry['input_caption'],
                    "output_caption": entry['output_caption']
                }
                f.write(json.dumps(jsonl_entry) + '\n')

        print(f"Created {jsonl_file} with {len(metadata)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", required=True, help="Directory with <base>.png and <base>_obfuscated.png pairs.")
    parser.add_argument("--output-dir", required=True, help="Directory for the dataset (images, metadata, JSONL).")
    parser.add_argument("--target", choices=sorted(PAPER_PROMPTS), default="gfw",
                        help="Evasion target, selects the paper prompt (default: gfw).")
    parser.add_argument("--instruction-mode", choices=["consistent", "varied", "augmented"], default="consistent",
                        help="consistent = paper prompt for every pair; varied/augmented = paraphrased instructions.")
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    create_instruct_pix2pix_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        instruction_mode=args.instruction_mode,
        target=args.target,
    )
    create_simplified_jsonl_format(args.output_dir, args.output_dir)
    print("\nDataset preparation complete!")
