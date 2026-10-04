"""Convert PCAP traces to nPrint CSV chunks without relying on Scapy.

Each trace is split into blocks of ``PACKETS_PER_CHUNK`` (1024) packets and every
block is encoded with nPrint over the IPv4, IPv6, TCP, UDP and ICMP headers
(``-4 -6 -t -u -i``). After the Traffic Encoder masks the IPv4 and IPv6 address
bits, 1088 header bits remain per packet; the 64 non-address IPv6 bits are always
-1 for IPv4 traffic. Payload bytes are not encoded (``-p 0``); pass ``chunk_pcap_dir`` to keep each block as a PCAP so the
Protocol Decoder can reattach the original payloads after editing.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _resolve_nprint_binary() -> str:
    """Locate the nPrint v1.2.1 binary.

    Resolution order:
      1. ``NPRINT_BIN`` environment variable (explicit path);
      2. an ``nprint`` binary on ``$PATH`` (e.g. after ``scripts/install_nprint.sh``);
      3. a local ``third_party/nprint-1.2.1/nprint`` build inside the repo.
    """
    env_bin = os.environ.get("NPRINT_BIN")
    if env_bin:
        return env_bin
    which = shutil.which("nprint")
    if which:
        return which
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "third_party" / "nprint-1.2.1" / "nprint")


NPRINT_PATH = _resolve_nprint_binary()
PACKETS_PER_CHUNK = int(os.environ.get("PACKETS_PER_CHUNK", "1024"))
NPRINT_OUTPUT_SUBDIR = os.environ.get("NPRINT_OUTPUT_DIR", "nprint_traffic")


def _write_chunk(
    global_header: bytes,
    records: Sequence[Tuple[bytes, bytes]],
    base_name: str,
    chunk_index: int,
    output_dir: Path,
    obfuscated: bool,
    max_packets: int,
    chunk_pcap_dir: Optional[Path] = None,
) -> None:
    suffix = "_obfuscated" if obfuscated else ""
    chunk_name = f"{base_name}_{chunk_index:04d}{suffix}.nprint"
    chunk_output = output_dir / chunk_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(global_header)
        for rec_header, packet in records:
            tmp.write(rec_header)
            tmp.write(packet)
        temp_path = Path(tmp.name)

    cmd = [
        str(NPRINT_PATH),
        "-F",
        "-1",
        "-P",
        str(temp_path),
        "-4",
        "-i",
        "-6",
        "-t",
        "-u",
        "-p",
        "0",
        "-c",
        str(max_packets),
        "-W",
        str(chunk_output),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"nprint failed for {chunk_name}: {exc.stderr}")
        raise
    finally:
        if chunk_pcap_dir is not None:
            chunk_pcap_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), chunk_pcap_dir / Path(chunk_name).with_suffix(".pcap"))
        else:
            temp_path.unlink(missing_ok=True)


def split_and_convert_pcap_to_nprint(
    pcap_file_path: Path,
    output_dir: Path,
    obfuscated: bool = False,
    chunk_pcap_dir: Optional[Path] = None,
) -> None:
    with pcap_file_path.open("rb") as handle:
        global_header = handle.read(24)
        if len(global_header) != 24:
            raise ValueError(f"Invalid pcap header in {pcap_file_path}")

        magic = global_header[:4]
        if magic == b"\xd4\xc3\xb2\xa1":
            endian = "<"
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian = ">"
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian = "<"
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian = ">"
        elif magic == b"\n\r\r\n":
            print(f"Skipping pcap-ng file (no conversion): {pcap_file_path.name}")
            return
        else:
            raise ValueError(f"Unsupported pcap magic {magic!r} in {pcap_file_path}")

        base_name = pcap_file_path.stem
        records: List[Tuple[bytes, bytes]] = []
        chunk_index = 0

        while True:
            record_header = handle.read(16)
            if len(record_header) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", record_header)
            packet = handle.read(incl_len)
            if len(packet) < incl_len:
                break

            records.append((record_header, packet))

            if len(records) == PACKETS_PER_CHUNK:
                _write_chunk(
                    global_header,
                    records,
                    base_name,
                    chunk_index,
                    output_dir,
                    obfuscated,
                    PACKETS_PER_CHUNK,
                    chunk_pcap_dir,
                )
                chunk_index += 1
                records = []

        if records:
            _write_chunk(
                global_header,
                records,
                base_name,
                chunk_index,
                output_dir,
                obfuscated,
                PACKETS_PER_CHUNK,
                chunk_pcap_dir,
            )


def convert_directory(
    pcap_dir: Path,
    output_root: Optional[Path] = None,
    *,
    obfuscated: bool = False,
) -> None:
    pcap_dir = pcap_dir.resolve()
    output_root = output_root or (pcap_dir / NPRINT_OUTPUT_SUBDIR)
    output_root.mkdir(parents=True, exist_ok=True)

    for path in sorted(pcap_dir.glob("*.pcap")):
        split_and_convert_pcap_to_nprint(path, output_root, obfuscated=obfuscated)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Split PCAPs into 1024-packet blocks and encode them with nPrint.")
    parser.add_argument("--pcap", type=Path, help="A single PCAP file.")
    parser.add_argument("--pcap-dir", type=Path, help="A directory of PCAP files.")
    parser.add_argument("--output-dir", "-o", required=True, type=Path, help="Directory for the .nprint chunks.")
    parser.add_argument("--chunk-pcap-dir", type=Path,
                        help="Also keep each 1024-packet block as a PCAP (needed to reattach payloads).")
    parser.add_argument("--obfuscated", action="store_true", help="Append _obfuscated to the chunk names.")
    args = parser.parse_args()
    if bool(args.pcap) == bool(args.pcap_dir):
        parser.error("pass exactly one of --pcap or --pcap-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pcaps = [args.pcap] if args.pcap else sorted(args.pcap_dir.glob("*.pcap"))
    for pcap in pcaps:
        split_and_convert_pcap_to_nprint(pcap, args.output_dir, args.obfuscated, args.chunk_pcap_dir)


__all__ = ["split_and_convert_pcap_to_nprint", "convert_directory"]


if __name__ == "__main__":
    main()
