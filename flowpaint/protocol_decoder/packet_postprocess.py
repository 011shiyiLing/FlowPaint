#!/usr/bin/env python3
"""Intra-packet repair and inter-packet consistency restoration (paper Sec. 3.5).

Runs the NetDiffusion repair passes from ``reconstruction.py`` on a decoded nPrint
chunk, sized to the actual number of packets in the chunk, and writes the
repaired nPrint. ``reassembly.py`` then turns it into a replayable PCAP.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import math
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import pandas as pd
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parent
RECONSTRUCTION_PATH = HERE / "reconstruction.py"
DEFAULT_TEMPLATE = HERE / "correct_format.nprint"


def install_scapy_stub() -> None:
    """Register lightweight stand-ins so we can import without real scapy."""
    for name in [
        "scapy",
        "scapy.all",
        "scapy.utils",
        "scapy.layers",
        "scapy.layers.inet",
    ]:
        sys.modules.pop(name, None)

    def _unavailable(*_: object, **__: object) -> None:
        raise RuntimeError("scapy is unavailable in this environment")

    scapy_pkg = types.ModuleType("scapy")
    sys.modules["scapy"] = scapy_pkg

    scapy_all = types.ModuleType("scapy.all")
    scapy_all.__all__ = []  # type: ignore[attr-defined]
    scapy_all.rdpcap = _unavailable  # type: ignore[attr-defined]
    scapy_all.wrpcap = _unavailable  # type: ignore[attr-defined]
    sys.modules["scapy.all"] = scapy_all

    utils_mod = types.ModuleType("scapy.utils")

    class _DummyWriter:
        def __init__(self, *_: object, **__: object) -> None:  # type: ignore[no-untyped-def]
            _unavailable()

    utils_mod.PcapWriter = _DummyWriter  # type: ignore[attr-defined]
    sys.modules["scapy.utils"] = utils_mod

    layers_pkg = types.ModuleType("scapy.layers")
    sys.modules["scapy.layers"] = layers_pkg

    layers_inet = types.ModuleType("scapy.layers.inet")

    class _DummyLayer:
        pass

    for layer_name in ("IP", "TCP", "UDP", "ICMP"):
        setattr(layers_inet, layer_name, _DummyLayer)
    sys.modules["scapy.layers.inet"] = layers_inet


def _import_reconstruction(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, RECONSTRUCTION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for {RECONSTRUCTION_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_reconstruction_module():
    """Import the historical reconstruction script lazily."""
    module_name = "netdiffusion_reconstruction"
    install_scapy_stub()
    sys.modules.pop(module_name, None)
    return _import_reconstruction(module_name)


def extend_sequence(sequence: Sequence[int], target_len: int) -> List[int]:
    """Resize the Markov chain output so it matches the chunk length."""
    if target_len <= 0:
        return []
    seq_list = list(sequence)
    if not seq_list:
        return [0] * target_len
    if len(seq_list) >= target_len:
        return seq_list[:target_len]
    repeats = math.ceil(target_len / len(seq_list))
    extended = seq_list * repeats
    return extended[:target_len]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True, help="nPrint file emitted by diffusion.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the post-processed nPrint.")
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Reference nPrint used for column order + transition statistics.",
    )
    parser.add_argument(
        "--transition-reference",
        type=Path,
        help="Optional path providing src/dst transition statistics (defaults to --template).",
    )
    parser.add_argument("--src-ip", default="0.0.0.0", help="Optional fixed source IP override.")
    parser.add_argument("--dst-ip", default="0.0.0.0", help="Optional fixed destination IP override.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for all stochastic repairs.")
    return parser.parse_args()


def ensure_template(df: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns so downstream formatting matches expectations."""
    df = df.reindex(columns=template.columns, fill_value=0)
    # Address columns hold dotted-quad strings once ip_address_formatting runs.
    for column in ("src_ip", "dst_ip"):
        if column in df.columns:
            df[column] = df[column].astype(object)
    return df


def zero_ipv6(df: pd.DataFrame) -> None:
    cols = [column for column in df.columns if "ipv6" in column]
    if cols:
        df.loc[:, cols] = -1


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    recon = load_reconstruction_module()
    recon.args = SimpleNamespace(src_ip=args.src_ip, dst_ip=args.dst_ip)

    generated = recon.read_syn_nprint(str(args.generated))
    template = pd.read_csv(args.template)
    generated = ensure_template(generated, template)

    seq_source = args.transition_reference or args.template
    synthetic_sequence = recon.direction_sampleing(str(seq_source))
    synthetic_sequence = extend_sequence(synthetic_sequence, len(generated))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        generated = recon.ip_address_formatting(generated, synthetic_sequence)
        generated = recon.ipv4_ver_formatting(generated, template)
        generated = recon.ipv4_header_negative_removal(generated, template)
        generated = recon.ipv4_pro_formatting(generated, template)
        generated = recon.ipv4_option_removal(generated, template)
        generated = recon.ipv4_ttl_ensure(generated, template)
        generated = recon.ipv4_hl_formatting(generated, template)

        dominating_protocol = recon.protocol_determination(generated)
        if dominating_protocol == "tcp":
            generated = recon.tcp_header_negative_removal(generated, template)
            generated = recon.tcp_opt_formatting(generated, template)
            generated = recon.tcp_data_offset_calculation(generated, template)
            generated = recon.ipv4_tl_formatting_tcp(generated, template)
        elif dominating_protocol == "udp":
            generated = recon.udp_header_negative_removal(generated, template)
            generated = recon.udp_len_calculation(generated, template)
            generated = recon.ipv4_tl_formatting_udp(generated, template)

        generated = recon.id_num_initialization_src_dst(generated)
        generated = recon.id_num_initialization_dst_src(generated)
        generated = recon.ip_fragementation_bits(generated)

        dominating_protocol = recon.protocol_determination(generated)
        if dominating_protocol == "tcp":
            generated = recon.port_initialization(generated)
            generated = recon.seq_initialization_src_dst(generated)
            generated = recon.seq_initialization_dst_src(generated)
            generated = recon.three_way_handshake(generated)
            generated = recon.ackn_initialization_src_dst(generated)
        elif dominating_protocol == "udp":
            generated = recon.port_initialization(generated)

    zero_ipv6(generated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
