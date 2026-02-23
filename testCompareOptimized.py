"""
Compare SCL vs AISCL (original) vs AISCL (optimized).

Tests whether optimized implementation beats standard SCL.
"""
import time
import numpy as np
import torch

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.optimized_codec import OptimizedAISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)


def run_compare(N=128, K=64, messages=100, L=4, snr_range=None):
    """Run comparison: SCL vs AISCL vs OptimizedAISCL."""
    if snr_range is None:
        snr_range = [i / 2 for i in range(11)]

    # Codecs
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    # Load trained AI model
    ai_model = PathPruningNet(N)
    try:
        ai_model.load_state_dict(torch.load(f'trained_model_N{N}_K{K}.pt'))
        ai_model.eval()
    except Exception:
        print("[Warning] AI model not found, using untrained model")
        ai_model.eval()
    
    aiscl_codec = AISCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)
    opt_aiscl_codec = OptimizedAISCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print('Comparing SCL vs AISCL vs OptimizedAISCL')
    print(f'Parameters: N={N} K={K} L={L} messages={messages}')
    print('\nSNR(dB) | BER_scl | BER_aiscl | BER_optaiscl | time_scl(ms) | time_aiscl(ms) | time_optaiscl(ms)')

    csv_path = 'compare_optimized_results.csv'
    with open(csv_path, 'w') as fh:
        fh.write('snr,ber_scl,ber_aiscl,ber_optaiscl,time_scl_ms,time_aiscl_ms,time_optaiscl_ms\n')

    for snr in snr_range:
        ber_scl = ber_aiscl = ber_optaiscl = 0
        times_scl = times_aiscl = times_optaiscl = []
        times_scl = []
        times_aiscl = []
        times_optaiscl = []

        for _ in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr)

            # SCL decode
            t0 = time.perf_counter()
            decoded_scl = scl_codec.decode(received)
            t1 = time.perf_counter()
            times_scl.append((t1 - t0) * 1000.0)
            ber_scl += compute_fails(msg, decoded_scl)[0]

            # AISCL decode
            t2 = time.perf_counter()
            decoded_aiscl = aiscl_codec.decode(received)
            t3 = time.perf_counter()
            times_aiscl.append((t3 - t2) * 1000.0)
            ber_aiscl += compute_fails(msg, decoded_aiscl)[0]

            # Optimized AISCL decode
            t4 = time.perf_counter()
            try:
                decoded_optaiscl = opt_aiscl_codec.decode(received)
                t5 = time.perf_counter()
                times_optaiscl.append((t5 - t4) * 1000.0)
                ber_optaiscl += compute_fails(msg, decoded_optaiscl)[0]
            except Exception as e:
                print(f"[Error in OptimizedAISCL at SNR={snr}]: {e}")
                times_optaiscl.append(np.nan)
                ber_optaiscl += K  # Penalize with max errors

        # Averages
        total_bits = messages * K
        avg_time_scl = np.nanmean(times_scl)
        avg_time_aiscl = np.nanmean(times_aiscl)
        avg_time_optaiscl = np.nanmean(times_optaiscl)

        ber_scl_val = ber_scl / total_bits
        ber_aiscl_val = ber_aiscl / total_bits
        ber_optaiscl_val = ber_optaiscl / total_bits

        ratio_opt = avg_time_optaiscl / avg_time_scl if avg_time_scl > 0 else np.inf
        ratio_aiscl = avg_time_aiscl / avg_time_scl if avg_time_scl > 0 else np.inf
        
        print(f"{snr:5.1f}  | {ber_scl_val:.6e} | {ber_aiscl_val:.6e} | {ber_optaiscl_val:.6e} | {avg_time_scl:8.3f} | {avg_time_aiscl:8.3f} | {avg_time_optaiscl:8.3f}")
        
        with open(csv_path, 'a') as fh:
            fh.write(f"{snr},{ber_scl_val:.6e},{ber_aiscl_val:.6e},{ber_optaiscl_val:.6e},{avg_time_scl:.6f},{avg_time_aiscl:.6f},{avg_time_optaiscl:.6f}\n")

    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    run_compare(N=128, K=64, messages=100, L=4)
