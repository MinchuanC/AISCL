"""
Benchmark SC List (SCL) decoder.

Reports per-SNR: BER, FER, average runtime per frame (ms).

Usage: run from project root with virtualenv activated.
"""
import time
import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)


def run_compare(N=128, K=64, messages=1000, L=4, snr_range=None):
    """Benchmark SCL decoder performance.
    
    Parameters
    ----------
    N : int
        Code length
    K : int
        Information bits
    messages : int
        Number of test frames per SNR
    L : int
        List size
    snr_range : list
        SNR values in dB
    """
    if snr_range is None:
        snr_range = [i / 2 for i in range(11)]

    # Create codec
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print('SCL Decoder Benchmark')
    print(f'Parameters: N={N} K={K} L={L} messages={messages}')
    print('\nSNR(dB) | BER | FER | time_per_frame(ms)')

    # Prepare CSV
    csv_path = 'compare_results_scl_benchmark.csv'
    with open(csv_path, 'w') as fh:
        fh.write('snr,ber,fer,time_ms\n')

    for snr in snr_range:
        ber = 0
        fer = 0
        times = []

        for i in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr)

            # SCL decode
            t0 = time.perf_counter()
            decoded = scl_codec.decode(received)
            t1 = time.perf_counter()

            # Measure BER and FER
            bit_errors, frame_error = compute_fails(msg, decoded)

            ber += bit_errors
            fer += frame_error
            times.append((t1 - t0) * 1000.0)
            
            if (i + 1) % 100 == 0:
                print(f"  SNR {snr}: {i+1}/{messages} processed...", flush=True)

        # Compute averages
        total_bits = messages * K
        avg_time = np.mean(times)
        ber_val = ber / total_bits
        fer_val = fer / messages
        
        print(f"{snr:5.1f}  | {ber_val:.6e} | {fer_val:.6e} | {avg_time:8.3f}")
        
        # Write to CSV
        with open(csv_path, 'a') as fh:
            fh.write(f"{snr},{ber_val:.6e},{fer_val:.6e},{avg_time:.6f}\n")

    print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    run_compare(N=128, K=64, messages=1000, L=4)
