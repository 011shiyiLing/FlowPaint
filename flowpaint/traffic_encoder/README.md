# Traffic Encoder (paper §3.3)

Turns a raw PCAP into flow images that the Generative Editor can edit.

1. **Segmentation and nPrint encoding** (`pcap2nprint.py`). A trace is split into
   blocks of 1024 packets and each block is encoded with
   [nPrint](https://github.com/nprint/nprint) over the IPv4, IPv6, TCP, UDP and ICMP
   headers (`-4 -6 -t -u -i -p 0`). Every packet becomes a row of bits with values in
   {1, 0, −1}, where −1 marks a field that is absent. Payload bytes are not encoded.
2. **Address masking and ternary color encoding** (`nprint_to_png.py`). Source and
   destination IPv4 and IPv6 address bits are removed, so the model edits statistical
   structure rather than endpoints. This leaves 1088 header bits per packet: the
   IPv4, TCP, UDP and ICMP fields plus the 64 non-address bits of the IPv6 block,
   which are always −1 for IPv4 traffic. Each remaining bit becomes one pixel:
   **orange = 1**, **purple = 0**, **yellow = −1**. Blocks shorter than 1024
   packets are padded with blue rows that the decoder later drops. The result is a
   1088 × 1024 RGBA PNG (width × height).

## Usage

```bash
# PCAP -> nPrint blocks (keep the PCAP blocks too, the decoder needs them to reattach payloads)
python -m flowpaint.traffic_encoder.pcap2nprint \
  --pcap flow.pcap --output-dir nprint/ --chunk-pcap-dir pcap_chunks/

# nPrint blocks -> flow images
python -m flowpaint.traffic_encoder.nprint_to_png -i nprint/ -o png/
```

`pcap2nprint` locates the nPrint binary through `NPRINT_BIN`, then `PATH`, then
`third_party/nprint-1.2.1/nprint` (see [`scripts/install_nprint.sh`](../../scripts/install_nprint.sh)).

| Environment variable | Default | Meaning |
|---|---|---|
| `NPRINT_BIN` | auto | Path to the nPrint v1.2.1 binary |
| `PACKETS_PER_CHUNK` | 1024 | Packets per block |
| `PNG_PADDED_HEIGHT` | 1024 | Image height (rows) |
