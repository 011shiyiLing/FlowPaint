"""
Instruction augmentation for better generalization during InstructPix2Pix training.
This script generates diverse instructions that express the same censorship bypass intent.
"""

import random
from typing import List

def generate_diverse_instructions() -> List[str]:
    """
    Generate a comprehensive set of instructions with various phrasings
    to improve model's understanding of user intent.
    """

    # Core concepts that users might express
    censorship_terms = ["censorship", "firewall", "GFW", "Great Firewall", "blocking", "filtering", "DPI", "deep packet inspection"]
    bypass_terms = ["bypass", "evade", "circumvent", "avoid", "defeat", "resist", "penetrate", "overcome"]
    target_terms = ["Chinese", "China", "China's", "Kazakhstan", "Kazakhstani", "state-level", "government"]

    # Template-based generation
    templates = [
        "{{bypass}} {{target}} {{censorship}}",
        "Make traffic {{bypass}} {{censorship}}",
        "Transform to {{bypass}} {{target}} {{censorship}}",
        "Generate flow that can {{bypass}} {{censorship}}",
        "Create packets to {{bypass}} {{target}} network {{censorship}}",
        "Apply obfuscation to {{bypass}} {{censorship}}",
        "Modify traffic for {{censorship}} {{bypass}}",
        "Enable {{censorship}} {{bypass}} for {{target}} networks",
    ]

    # Specific variations expressing the same intent
    specific_instructions = [
        # Technical descriptions
        "Apply TCP desynchronization for censorship evasion",
        "Use packet manipulation to bypass network filters",
        "Implement anti-censorship obfuscation",
        "Deploy evasion techniques against DPI",

        # Effect-focused descriptions
        "Make this traffic invisible to censors",
        "Hide traffic from government monitoring",
        "Prevent detection by national firewalls",
        "Ensure packets reach destination despite blocking",

        # Casual user expressions
        "Help me access blocked content",
        "Get around the firewall",
        "Make my traffic unblockable",
        "Bypass internet restrictions",

        # Location-specific
        "Evade Chinese internet censorship",
        "Bypass Kazakhstan's network filters",
        "Circumvent GFW blocking",
        "Defeat Great Firewall detection",

        # Success-rate focused
        "Apply high success rate bypass technique",
        "Use proven anti-censorship method",
        "Implement 98% effective evasion",
        "Deploy reliable firewall bypass",

        # Strategy-specific but user-friendly
        "Use Geneva's first strategy",
        "Apply Geneva censorship bypass",
        "Implement Geneva evasion technique",
        "Deploy Geneva's TCP manipulation",
    ]

    # Generate template-based variations
    generated = []
    for template in templates:
        for _ in range(3):  # Generate 3 variations per template
            instruction = template
            if "{{bypass}}" in instruction:
                instruction = instruction.replace("{{bypass}}", random.choice(bypass_terms))
            if "{{target}}" in instruction:
                instruction = instruction.replace("{{target}}", random.choice(target_terms))
            if "{{censorship}}" in instruction:
                instruction = instruction.replace("{{censorship}}", random.choice(censorship_terms))
            generated.append(instruction)

    # Combine all instructions
    all_instructions = specific_instructions + generated

    # Remove duplicates while preserving order
    seen = set()
    unique_instructions = []
    for inst in all_instructions:
        if inst.lower() not in seen:
            seen.add(inst.lower())
            unique_instructions.append(inst)

    return unique_instructions

def augment_dataset_with_diverse_instructions(
    original_instruction: str,
    augmentation_factor: int = 5
) -> List[str]:
    """
    For each training sample, generate multiple instruction variations.
    This is for data augmentation during training.

    Args:
        original_instruction: The base instruction
        augmentation_factor: How many variations to generate

    Returns:
        List of augmented instructions including the original
    """
    all_instructions = generate_diverse_instructions()

    # Always include the original
    augmented = [original_instruction]

    # Randomly sample additional instructions
    samples = random.sample(all_instructions, min(augmentation_factor - 1, len(all_instructions) - 1))
    augmented.extend(samples)

    return augmented

def test_instruction_similarity():
    """
    Test how different user inputs might be interpreted.
    """
    test_inputs = [
        "bypass china firewall",
        "evade GFW",
        "bypass firewall"
        "get around blocking",
        "make traffic undetectable",
        "circumvent Chinese censorship",
        "help me access blocked sites",
        "TCP obfuscation for China",
        "anti-censorship transformation",
        "defeat deep packet inspection",
    ]

    print("Testing instruction variations that should map to the same effect:\n")
    print("Training instruction: 'Transform traffic to bypass TCP-based censorship systems against Chinese firewall'\n")
    print("User might input:")
    for inp in test_inputs:
        print(f"  - {inp}")

    print("\n" + "="*80)
    print("Recommendation: Use diverse instructions during training for better generalization")

if __name__ == "__main__":
    print("Generating diverse instruction set for training...\n")
    instructions = generate_diverse_instructions()

    print(f"Generated {len(instructions)} unique instruction variations:")
    for i, inst in enumerate(instructions[:10], 1):
        print(f"{i:2d}. {inst}")

    print(f"\n... and {len(instructions) - 10} more variations")

    print("\n" + "="*80)
    test_instruction_similarity()

    print("\n" + "="*80)
    print("Usage in training:")
    print("1. For better generalization, randomly select from these instructions during training")
    print("2. Or use data augmentation: duplicate images with different instructions")
    print("3. This helps the model understand that different phrasings mean the same transformation")