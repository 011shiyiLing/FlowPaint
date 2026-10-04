#!/usr/bin/env python3
"""End-to-end FlowPaint pipeline for one PCAP batch.

    PCAP --(Traffic Encoder)--> nPrint --> PNG
         --(Generative Editor)--> repainted PNG
         --(Protocol Decoder)--> nPrint --> repaired nPrint --> replayable PCAP
                                 (original payloads reattached, checksums recomputed)

Each stage maps to a paper section:
  * Traffic Encoder    (Sec. 3.3): flowpaint/traffic_encoder/*
  * Generative Editor  (Sec. 3.4): flowpaint/generative_editor/inference.py
  * Protocol Decoder   (Sec. 3.5): flowpaint/protocol_decoder/*

Example:
  python scripts/run_pipeline.py \\
    --pcap flow.pcap \\
    --prompt "Transform the HTTPS network traffic visualization to bypass the GFW" \\
    --checkpoint /path/to/your/checkpoint \\
    --work-root ./jobs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flowpaint.traffic_encoder import pcap2nprint, nprint_to_png  # noqa: E402
from flowpaint.protocol_decoder import image_to_nprint, reassembly  # noqa: E402


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", required=True, type=Path, help="Input PCAP (segmented into 1024-packet chunks).")
    ap.add_argument("--prompt", required=True, help="Evasion instruction fed to the Generative Editor.")
    ap.add_argument("--checkpoint", default="", help="Fine-tuned SDXL checkpoint (dir with unet/ or hub id). "
                    "If empty, the encoder PNGs are passed through unedited (dry run).")
    ap.add_argument("--work-root", default=Path("./jobs"), type=Path, help="Per-job working directory root.")
    ap.add_argument("--template", default=REPO_ROOT / "flowpaint" / "protocol_decoder" / "correct_format.nprint",
                    type=Path, help="Reference nPrint template for post-processing.")
    ap.add_argument("--replay-iface", default="", help="If set, tcpreplay the rebuilt pcaps on this interface.")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--text-guidance-scale", type=float, default=7.5)
    ap.add_argument("--image-guidance-scale", type=float, default=1.5)
    args = ap.parse_args()

    pcap_path = args.pcap.resolve()
    job_dir = ensure_dir(args.work_root.resolve() / pcap_path.stem)
    chunk_pcap_dir = ensure_dir(job_dir / "00_pcap_chunks")
    nprint_dir = ensure_dir(job_dir / "01_nprint")
    src_png_dir = ensure_dir(job_dir / "02_png_src")
    gen_png_dir = ensure_dir(job_dir / "03_png_generated")
    gen_nprint_dir = ensure_dir(job_dir / "04_nprint_generated")
    post_nprint_dir = ensure_dir(job_dir / "05_nprint_postprocessed")
    rebuilt_pcap_dir = ensure_dir(job_dir / "06_pcap_rebuilt")

    pp = REPO_ROOT / "flowpaint" / "protocol_decoder" / "packet_postprocess.py"
    infer = REPO_ROOT / "flowpaint" / "generative_editor" / "inference.py"

    # 1) PCAP -> nPrint (1024-packet chunks)
    print("[stage] Traffic Encoder: PCAP -> nPrint")
    pcap2nprint.split_and_convert_pcap_to_nprint(pcap_path, nprint_dir, chunk_pcap_dir=chunk_pcap_dir)

    # 2) nPrint -> PNG (selective masking + ternary color encoding)
    print("[stage] Traffic Encoder: nPrint -> PNG")
    nprint_to_png.convert_directory(nprint_dir, src_png_dir)

    # 3) Generative Editor: PNG -> repainted PNG
    print("[stage] Generative Editor: repaint")
    if args.checkpoint:
        for png in sorted(src_png_dir.glob("*.png")):
            run_cmd([
                sys.executable, str(infer),
                "--checkpoint", str(args.checkpoint),
                "--input", str(png),
                "--prompt", args.prompt,
                "--output", str(gen_png_dir / png.name),
                "--num-steps", str(args.num_steps),
                "--text-guidance-scale", str(args.text_guidance_scale),
                "--image-guidance-scale", str(args.image_guidance_scale),
            ])
    else:
        print("[warn] no --checkpoint; passing encoder PNGs through unedited (dry run).")
        for png in src_png_dir.glob("*.png"):
            (gen_png_dir / png.name).write_bytes(png.read_bytes())

    # 4) Protocol Decoder: repainted PNG -> nPrint (inverse visual decoding)
    print("[stage] Protocol Decoder: PNG -> nPrint")
    for png in sorted(gen_png_dir.glob("*.png")):
        ref = nprint_dir / f"{png.stem}.nprint"
        if not ref.exists():
            cands = list(nprint_dir.glob(f"{png.stem}*.nprint"))
            if not cands:
                raise SystemExit(f"Cannot find reference nPrint for {png.stem}")
            ref = cands[0]
        image_to_nprint.reconstruct_nprint(ref, png, gen_nprint_dir / f"{png.stem}.nprint")

    # 5) Protocol Decoder: intra/inter-packet repair, then flow reassembly to PCAP
    print("[stage] Protocol Decoder: repair + reassemble")
    for gen_csv in sorted(gen_nprint_dir.glob("*.nprint")):
        base = gen_csv.stem
        post = post_nprint_dir / f"{base}_post.nprint"
        run_cmd([
            sys.executable, str(pp),
            "--generated", str(gen_csv),
            "--output", str(post),
            "--template", str(args.template),
        ])
        out_pcap = rebuilt_pcap_dir / f"{base}_rebuilt.pcap"
        final_nprint = post_nprint_dir / f"{base}_final.nprint"
        stats = reassembly.reassemble(post, out_pcap, chunk_pcap_dir / f"{base}.pcap", final_nprint)
        print(f"[reassembly] {base}: {stats.packets} packets, {stats.payload_bytes} payload bytes reattached, "
              f"{stats.padded_bytes} padded, {stats.leftover_bytes} original bytes not carried")

    # 6) Optional replay
    if args.replay_iface:
        print(f"[stage] tcpreplay -> {args.replay_iface}")
        for pcap in sorted(rebuilt_pcap_dir.glob("*.pcap")):
            run_cmd(["tcpreplay", f"--intf1={args.replay_iface}", str(pcap)])

    print(f"[done] Job dir: {job_dir}")


if __name__ == "__main__":
    main()
