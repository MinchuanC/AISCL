"""
Compare SC List (SCL) and AI-augmented SCL (AISCL) decoding.

Reports per-SNR: BER, FER, average runtime per frame (ms), and simple operation counts.

Usage: run from project root with virtualenv activated.
"""
import time
import numpy as np
import torch

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)


def summarize_ops_from_decoder(decoder):
    """Read operation counters from decoder instance and return summary."""
    # instrumentation removed; return empty summary
    return {}


def run_compare(N=128, K=64, messages=1000, L=4, snr_range=None):
    if snr_range is None:
        snr_range = [i / 2 for i in range(11)]

    # Codecs
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    # Load trained AI model if available, otherwise use untrained instance
    ai_model = PathPruningNet(N)
    try:
        ai_model.load_state_dict(torch.load(f'trained_model_N{N}_K{K}.pt'))
        ai_model.eval()
    except Exception:
        # proceed with untrained model
        pass

    # Ensure model is in eval mode to avoid BatchNorm requiring batch>1
    try:
        ai_model.eval()
    except Exception:
        pass

    aiscl_codec = AISCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print('Comparing SCL vs AISCL')
    print(f'Parameters: N={N} K={K} L={L} messages={messages}')
    print('\nSNR(dB) | BER_scl | BER_aiscl | time_scl(ms) | time_aiscl(ms)')

    # prepare CSV
    csv_path = 'compare_results.csv'
    with open(csv_path, 'w') as fh:
        fh.write('snr,ber_scl,ber_aiscl,time_scl_ms,time_aiscl_ms\n')

    for snr in snr_range:
        ber_scl = 0
        fer_scl = 0
        ber_aiscl = 0
        fer_aiscl = 0

        times_scl = []
        times_aiscl = []

        for _ in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr)

            # Reset decoder internal counters
            dec1 = scl_codec.decoder
            dec2 = aiscl_codec.decoder
            if hasattr(dec1, '_reset_counters'):
                dec1._reset_counters()
            if hasattr(dec2, '_reset_counters'):
                dec2._reset_counters()

            # SCL decode
            t0 = time.perf_counter()
            decoded1 = scl_codec.decode(received)
            t1 = time.perf_counter()

            # AISCL decode
            t2 = time.perf_counter()
            decoded2 = aiscl_codec.decode(received)
            t3 = time.perf_counter()

            # Measure
            bit_errors1, frame_error1 = compute_fails(msg, decoded1)
            bit_errors2, frame_error2 = compute_fails(msg, decoded2)

            ber_scl += bit_errors1
            fer_scl += frame_error1
            ber_aiscl += bit_errors2
            fer_aiscl += frame_error2

            times_scl.append((t1 - t0) * 1000.0)
            times_aiscl.append((t3 - t2) * 1000.0)

            # instrumentation removed

        # Averages
        total_bits = messages * K
        avg_time_scl = np.mean(times_scl)
        avg_time_aiscl = np.mean(times_aiscl)

        ber_scl_val = ber_scl / total_bits
        ber_aiscl_val = ber_aiscl / total_bits
        print(f"{snr:5.1f}  | {ber_scl_val:.6e} | {ber_aiscl_val:.6e} | {avg_time_scl:8.3f} | {avg_time_aiscl:8.3f}")
        # write into the csv file
        with open(csv_path, 'a') as fh:
            fh.write(f"{snr},{ber_scl_val:.4f},{ber_aiscl_val:.4f},{avg_time_scl:.6f},{avg_time_aiscl:.6f}\n")


if __name__ == '__main__':
    # Increase messages for more accurate BER statistics (reduce variance)
    run_compare(N=128, K=64, messages=1000, L=4)
