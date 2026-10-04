# Scripts

## `run_pipeline.py`

Runs the full FlowPaint pipeline on one PCAP.

```bash
python scripts/run_pipeline.py \
  --pcap flow.pcap \
  --prompt "Transform the HTTPS network traffic visualization to bypass the GFW" \
  --checkpoint /path/to/your/checkpoint \
  --work-root ./jobs
```

Each run writes its intermediate results to `<work-root>/<pcap name>/`:

| Directory | Content |
|---|---|
| `00_pcap_chunks/` | 1024-packet PCAP blocks of the input |
| `01_nprint/` | nPrint encoding of each block |
| `02_png_src/` | Original flow images |
| `03_png_generated/` | Flow images repainted by the Generative Editor |
| `04_nprint_generated/` | Decoded nPrint of the repainted images |
| `05_nprint_postprocessed/` | Repaired nPrint (`*_post`) and nPrint of the rebuilt PCAP (`*_final`) |
| `06_pcap_rebuilt/` | Replayable PCAPs with original payloads and valid checksums |

Options:

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint` | none | Fine-tuned editor. Without it the images pass through unedited (dry run). |
| `--num-steps` | 10 | DDIM steps |
| `--text-guidance-scale` | 7.5 | Text classifier-free guidance |
| `--image-guidance-scale` | 1.5 | Image classifier-free guidance |
| `--template` | `flowpaint/protocol_decoder/correct_format.nprint` | Reference nPrint for the repair passes |
| `--replay-iface` | none | Replay the rebuilt PCAPs on this interface with `tcpreplay` |

## `install_nprint.sh`

Builds nPrint v1.2.1 into `third_party/nprint-1.2.1/`. Afterwards set
`NPRINT_BIN=third_party/nprint-1.2.1/nprint` or put the binary on `PATH`.
