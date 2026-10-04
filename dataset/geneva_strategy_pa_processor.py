"""Synthesize Geneva evasive targets (paper Sec. 4).

Applies the GFW strategy used in the paper,
``[TCP:flags:PA]-duplicate(tamper{TCP:dataofs:replace:10}(tamper{TCP:chksum:corrupt},),)-|``,
to every PCAP in a folder and writes ``<name>_obfuscated.pcap`` targets.

Example:

    python dataset/geneva_strategy_pa_processor.py --input <benign_pcaps> --output <obfuscated_pcaps>
"""

import argparse
import os
from scapy.all import rdpcap, wrpcap
from scapy.layers.inet import TCP

# Your Geneva strategy function
def apply_geneva_strategy_1(packet):
    """
    Applies a given Geneva strategy to a single packet.
    Strategy: [TCP:flags:PA]-duplicate(tamper{TCP:dataofs:replace:10}(tamper{TCP:chksum:corrupt},),)-|
    """
    # Check if the packet is a TCP packet with PSH and ACK flags set
    # 0x18 is the bitmask for PSH (0x08) and ACK (0x10) flags
    if packet.haslayer(TCP) and packet[TCP].flags & 0x18 == 0x18:
        # Create a copy of the original packet to be tampered with
        tampered_packet = packet.copy()
        
        # Create a second copy to be sent as the duplicate
        duplicate_packet = packet.copy()
        
        # ----- Tamper the original packet -----
        if tampered_packet.haslayer(TCP):
            # Tamper the TCP dataofs field to a value of 10
            tampered_packet[TCP].dataofs = 10

            # Force an incorrect checksum and keep it; Scapy only recalculates when the
            # field is deleted/None, so assign a value that differs from the original.
            original_checksum = int(tampered_packet[TCP].chksum or 0)
            forged_checksum = (original_checksum + 1) & 0xFFFF
            if forged_checksum == original_checksum:
                forged_checksum ^= 0xFFFF
            tampered_packet[TCP].chksum = forged_checksum

        # The strategy sends the tampered packet first, then the original duplicate
        return [tampered_packet, duplicate_packet]
    
    # If the packet does not match the trigger, return it without any modification
    return [packet]

def process_folder(input_folder, output_obfuscated_folder, output_original_folder):
    """
    Iterates through all pcap files in a specified folder and applies the Geneva strategy.
    """
    # Ensure the output directories exist, create them if they don't
    if not os.path.exists(output_obfuscated_folder):
        os.makedirs(output_obfuscated_folder)
    if output_original_folder and not os.path.exists(output_original_folder):
        os.makedirs(output_original_folder)
    
    # Get all .pcap files from the input folder
    pcap_files = [f for f in os.listdir(input_folder) if f.endswith('.pcap')]
    
    if not pcap_files:
        print(f"No .pcap files found in folder {input_folder}.")
        return

    print(f"Found {len(pcap_files)} .pcap files in folder {input_folder}.")

    for file_name in pcap_files:
        original_pcap_path = os.path.join(input_folder, file_name)
        
        # Create the new filename with '_obfuscated' suffix
        name, ext = os.path.splitext(file_name)
        obfuscated_file_name = f"{name}_obfuscated{ext}"
        obfuscated_pcap_path = os.path.join(output_obfuscated_folder, obfuscated_file_name)
        

        print(f"\n--- Processing file: {file_name} ---")

        # Load the original pcap file
        try:
            packets = rdpcap(original_pcap_path)
            print(f"Successfully loaded {len(packets)} packets.")
        except Exception as e:
            print(f"Error: Unable to load file {file_name}. Skipping.")
            continue
        
        # A list to store the obfuscated packets
        obfuscated_packets = []

        # Optionally back up the original packets
        if output_original_folder:
            original_output_path = os.path.join(output_original_folder, file_name)
            wrpcap(original_output_path, packets)
            print(f"Original packets saved to {original_output_path}.")
        
        # Apply the Geneva strategy to each packet in the file
        for packet in packets:
            processed_packets = apply_geneva_strategy_1(packet)
            obfuscated_packets.extend(processed_packets)

        print(f"Generated {len(obfuscated_packets)} obfuscated packets.")

        # Save the obfuscated packets to the new pcap file
        wrpcap(obfuscated_pcap_path, obfuscated_packets)
        print(f"Obfuscated packets saved to {obfuscated_pcap_path}.")
        
    print("\n--- All files processed! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder of benign PCAPs.")
    parser.add_argument("--output", required=True, help="Folder for the obfuscated PCAPs.")
    parser.add_argument("--backup", help="Optional folder to copy the original PCAPs into.")
    args = parser.parse_args()
    process_folder(args.input, args.output, args.backup)
