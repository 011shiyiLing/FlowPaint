# Protocol Decoder (paper §3.5)

Turns an edited flow image back into a valid, replayable PCAP. The repair passes
reuse NetDiffusion's protocol-repair pipeline, applied here to edited flows.

| Step | File | What it does |
|---|---|---|
| Inverse visual decoding | `image_to_nprint.py` | Resizes the image back to the encoder geometry if needed, drops padding rows, and maps every pixel to the nPrint value of its nearest palette color (orange → 1, purple → 0, yellow → −1). The matrix is reindexed to the reference nPrint column order. |
| Intra-packet repair | `packet_postprocess.py` (passes in `reconstruction.py`) | IP version 4, majority transport protocol, TTL > 0, IPv4 options removed, remaining placeholder bits filled randomly, IPv4 header length and TCP data offset recomputed, IPv4 total length kept between the header size and 1500, UDP length kept between 8 and 1500, TCP options snapped to the nearest valid kind. |
| Inter-packet restoration | `packet_postprocess.py` (passes in `reconstruction.py`) | Direction sequence sampled from a Markov chain fitted on reference traffic, consistent addresses and ports, per-direction IPv4 identification, TCP sequence and acknowledgement numbers, three-way handshake flags, fragmentation disabled. |
| Flow reassembly | `reassembly.py` | Converts the repaired nPrint to packets with `nprint -N`, reattaches the original application payloads, and recomputes the IPv4, TCP, UDP and ICMP checksums. |

Payload reattachment reads the original flow's payload per direction as one byte
stream and re-segments it, in order, to fill the payload length that each
reconstructed packet declares. If the edited flow declares more payload room than the
original carried, the rest is zero padded. If it declares less, the leftover bytes are
reported.

`correct_format.nprint` is the reference nPrint used for column order and for fitting
the direction Markov chain. Its addresses are replaced with documentation addresses
(192.0.2.10 and 198.51.100.10). The repair passes clear all IPv6 fields to −1, since
the scope is IPv4.

## Usage

```bash
# 1. image -> nPrint
python -m flowpaint.protocol_decoder.image_to_nprint \
  --org_nprint nprint/flow_0000.nprint --input_dir repainted_png/ --output_dir decoded/

# 2. intra-packet repair and inter-packet restoration
python flowpaint/protocol_decoder/packet_postprocess.py \
  --generated decoded/flow_0000.nprint --output repaired.nprint

# 3. flow reassembly
python flowpaint/protocol_decoder/reassembly.py \
  --nprint repaired.nprint --original-pcap pcap_chunks/flow_0000.pcap \
  --output rebuilt.pcap --final-nprint rebuilt.nprint
```

`reconstruction.py` can also run steps 2 and 3 in one call (`--generated_nprint_path`,
`--formatted_nprint_path`, `--output`, `--nprint`, `--original_pcap`).
