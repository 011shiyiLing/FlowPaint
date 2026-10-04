#!/usr/bin/env python3
"""Flow Reassembly (paper Sec. 3.5): repaired nPrint -> replayable PCAP.

This is the last stage of the Protocol Decoder. It

1. converts the repaired nPrint matrix back to packets with ``nprint -N``;
2. reattaches the original application payloads. The nPrint matrix never encodes
   payload bytes, so the edit only changes headers. The payload of the original
   flow is read per direction (client to server, server to client) as one byte
   stream and re-segmented in order to fill the payload length that each
   reconstructed packet declares (IPv4 total length minus the IPv4 and TCP
   header lengths, or UDP length minus 8). Bytes therefore keep their order
   within each direction. If the edited flow declares more payload room than the
   original carried, the remainder is zero padded; if it declares less, the
   leftover bytes are reported;
3. recomputes the IPv4, TCP, UDP and ICMP checksums so that every packet is valid.

Example:

    python flowpaint/protocol_decoder/reassembly.py \\
        --nprint repaired.nprint --original-pcap chunk_0000.pcap \\
        --output rebuilt.pcap --final-nprint final.nprint
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from scapy.all import ICMP, IP, TCP, UDP, Raw, rdpcap
from scapy.utils import PcapWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flowpaint.traffic_encoder.pcap2nprint import NPRINT_PATH  # noqa: E402

# Same header layers as the Traffic Encoder: IPv4, ICMP, IPv6, TCP, UDP, no payload.
NPRINT_LAYER_FLAGS = ["-4", "-i", "-6", "-t", "-u", "-p", "0"]
DLT_RAW = 101


@dataclass
class ReassemblyStats:
    packets: int = 0
    payload_bytes: int = 0
    padded_bytes: int = 0
    leftover_bytes: int = 0


def nprint_to_pcap(nprint_path: Path, pcap_path: Path) -> None:
    subprocess.run([str(NPRINT_PATH), "-N", str(nprint_path), "-W", str(pcap_path)], check=True)


def pcap_to_nprint(pcap_path: Path, nprint_path: Path, max_packets: int = 1024) -> None:
    subprocess.run(
        [str(NPRINT_PATH), "-F", "-1", "-P", str(pcap_path), *NPRINT_LAYER_FLAGS,
         "-c", str(max_packets), "-W", str(nprint_path)],
        check=True,
    )


def _transport_payload(pkt: IP) -> bytes:
    if pkt.haslayer(TCP):
        return bytes(pkt[TCP].payload)
    if pkt.haslayer(UDP):
        return bytes(pkt[UDP].payload)
    return b""


def original_payload_streams(original_pcap: Path) -> Dict[str, bytearray]:
    """Return the original TCP/UDP payload bytes per direction, in packet order."""
    streams = {"c2s": bytearray(), "s2c": bytearray()}
    client: Optional[str] = None
    for pkt in rdpcap(str(original_pcap)):
        if not pkt.haslayer(IP):
            continue
        ip = pkt[IP]
        if client is None:
            client = ip.src
        direction = "c2s" if ip.src == client else "s2c"
        streams[direction].extend(_transport_payload(ip))
    return streams


def _declared_payload_len(ip: IP) -> int:
    if ip.haslayer(TCP):
        return ip.len - ip.ihl * 4 - ip[TCP].dataofs * 4
    if ip.haslayer(UDP):
        return ip[UDP].len - 8
    return 0


def reattach_payloads(packets: List[IP], streams: Dict[str, bytearray], stats: ReassemblyStats) -> None:
    """Fill each packet's declared payload length from the original byte streams."""
    offsets = {"c2s": 0, "s2c": 0}
    client = packets[0].src if packets else None
    for ip in packets:
        layer = TCP if ip.haslayer(TCP) else UDP if ip.haslayer(UDP) else None
        if layer is None:
            continue
        need = max(0, _declared_payload_len(ip))
        direction = "c2s" if ip.src == client else "s2c"
        start = offsets[direction]
        chunk = bytes(streams[direction][start:start + need])
        offsets[direction] = start + len(chunk)
        stats.payload_bytes += len(chunk)
        stats.padded_bytes += need - len(chunk)
        ip[layer].remove_payload()
        if need:
            ip[layer].add_payload(Raw(chunk + b"\x00" * (need - len(chunk))))
    stats.leftover_bytes = sum(len(streams[d]) - offsets[d] for d in streams)


def recompute_checksums(ip: IP) -> IP:
    """Drop every checksum so scapy recomputes them when the packet is serialized."""
    del ip.chksum
    for layer in (TCP, UDP, ICMP):
        if ip.haslayer(layer):
            del ip[layer].chksum
    return IP(bytes(ip))


def reassemble(
    nprint_path: Path,
    output_pcap: Path,
    original_pcap: Optional[Path] = None,
    final_nprint: Optional[Path] = None,
) -> ReassemblyStats:
    output_pcap.parent.mkdir(parents=True, exist_ok=True)
    headers_pcap = output_pcap.with_suffix(".headers.pcap")
    nprint_to_pcap(nprint_path, headers_pcap)

    packets = [pkt[IP] for pkt in rdpcap(str(headers_pcap)) if pkt.haslayer(IP)]
    stats = ReassemblyStats(packets=len(packets))
    if original_pcap is not None:
        reattach_payloads(packets, original_payload_streams(original_pcap), stats)

    writer = PcapWriter(str(output_pcap), linktype=DLT_RAW, sync=True)
    for ip in packets:
        writer.write(bytes(recompute_checksums(ip)))
    writer.close()
    headers_pcap.unlink(missing_ok=True)

    if final_nprint is not None:
        pcap_to_nprint(output_pcap, final_nprint, max_packets=max(1, len(packets)))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nprint", required=True, type=Path, help="Repaired nPrint from packet_postprocess.py.")
    parser.add_argument("--output", required=True, type=Path, help="Path of the rebuilt PCAP.")
    parser.add_argument("--original-pcap", type=Path,
                        help="PCAP of the original (unedited) flow chunk whose payloads are reattached.")
    parser.add_argument("--final-nprint", type=Path, help="Optionally re-encode the rebuilt PCAP to this nPrint.")
    args = parser.parse_args()

    stats = reassemble(args.nprint, args.output, args.original_pcap, args.final_nprint)
    print(
        f"[reassembly] {stats.packets} packets, {stats.payload_bytes} payload bytes reattached, "
        f"{stats.padded_bytes} padded, {stats.leftover_bytes} original bytes not carried"
    )


if __name__ == "__main__":
    main()
