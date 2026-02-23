"""
Quick test: Compare SC List (SCL) and AI-Fast-SCL decoding with fewer SNR points.
"""
import time
import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)


def run_compare_quick(N=128, K=64, messages=50, L=4):
    """Run quick comparison test with fewer messages and SNR points."""
    
    # Test only 3 SNR points
    snr_range = [0.0, 2.0, 4.0]

    # Create codecs
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    # Load or create AI model for AIFastSCL
    ai_model = PathPruningNN(input_dim=7, hidden_dim=32)
    try:
        ai_model.load_weights(f'trained_model_N{N}_K{K}.pt')
        ai_model.eval()
        print(f"[Info] Loaded trained model: trained_model_N{N}_K{K}.pt")
    except Exception as e:
        print(f"[Info] Trained model not found, using untrained PathPruningNN: {e}")
        ai_model.eval()

    # Create AIFastSCL with trained model
    ai_fastsscl_codec = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L,
        ai_model=ai_model,
        ai_threshold=0.05,
        enable_ai_pruning=True
    )
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print(f'\nComparing SCL vs AI-Fast-SCL (Quick Test)')
    print(f'Parameters: N={N} K={K} L={L} messages={messages}')
    print('\nSNR(dB) | BER_scl | BER_aifscl | time_scl(ms) | time_aifscl(ms) | Speedup')

    for snr in snr_range:
        ber_scl = 0
        ber_aifastscl = 0

        times_scl = []
        times_aifastscl = []

        for i in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr)

            # SCL decode
            t0 = time.perf_counter()
            decoded_scl = scl_codec.decode(received)
            t1 = time.perf_counter()

            # AIFastSCL decode (with AI enabled)
            t2 = time.perf_counter()
            decoded_aifastscl = ai_fastsscl_codec.decode(received)
            t3 = time.perf_counter()

            # Measure BER
            bit_errors_scl, _ = compute_fails(msg, decoded_scl)
            bit_errors_aifastscl, _ = compute_fails(msg, decoded_aifastscl)

            ber_scl += bit_errors_scl
            ber_aifastscl += bit_errors_aifastscl

            times_scl.append((t1 - t0) * 1000.0)
            times_aifastscl.append((t3 - t2) * 1000.0)

            if (i + 1) % 10 == 0:
                print(f"  SNR {snr}: {i+1}/{messages} processed...", flush=True)

        # Compute averages
        total_bits = messages * K
        avg_time_scl = np.mean(times_scl)
        avg_time_aifastscl = np.mean(times_aifastscl)

        ber_scl_val = ber_scl / total_bits
        ber_aifastscl_val = ber_aifastscl / total_bits
        
        # Compute speedup
        speedup = avg_time_scl / avg_time_aifastscl if avg_time_aifastscl > 0 else 1.0

        print(f"{snr:5.1f}  | {ber_scl_val:.6e} | {ber_aifastscl_val:.6e} | {avg_time_scl:8.3f} | {avg_time_aifastscl:8.3f} | {speedup:6.2f}x")

    print("\nDone!")


if __name__ == '__main__':
    run_compare_quick(N=128, K=64, messages=50, L=4)
