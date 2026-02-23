"""
Compare SC List (SCL) and AI-Fast-SCL with the REAL trained model.
"""
import time
import os
import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)


def run_compare(N=128, K=64, messages=500, L=4):
    """Compare SCL vs AI-Fast-SCL with trained model."""
    
    snr_range = [0.0, 1.0, 2.0, 3.0, 4.0]

    # Create codecs
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    # Load trained model for AI-Fast-SCL
    ai_model = PathPruningNN(input_dim=7, hidden_dim=32)
    try:
        ai_model.load_weights('trained_model_real_N128_K64.pt')
        ai_model.eval()
        print(f"[✓] Loaded trained model")
    except Exception as e:
        print(f"[✗] Failed to load trained model: {e}")
        return
    # If a TorchScript traced network exists, load it and attach to the ai_model
    import torch as _torch
    ts_path = 'trained_model_real_N128_K64_ts.pt'
    if _torch and _torch.jit and _torch.jit.load and _torch.jit.is_scripting:
        pass
    try:
        if os.path.exists(ts_path):
            scripted_net = _torch.jit.load(ts_path, map_location='cpu')
            # Attach the scripted network to the existing model instance so
            # that ai_model.forward will use the optimized network.
            ai_model.network = scripted_net
            print(f"[✓] Attached TorchScript network from {ts_path}")
    except Exception:
        # Ignore TorchScript attach failures and continue with the original model
        pass
    
    # Create AI-Fast-SCL codec
    ai_fastsscl_codec = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L,
        ai_model=ai_model,
        ai_threshold=0.5,  # Use aggressive threshold
        enable_ai_pruning=True
    )
    # Apply tuned quick-pruning-only settings to maximize speed
    ai_fastsscl_codec.quick_percentile = 40.0
    ai_fastsscl_codec.topk_multiplier = 1.5
    ai_fastsscl_codec.force_quick_pruning = True
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

    print('\nComparing SCL vs AI-Fast-SCL (REAL Trained Model)')
    print(f'Parameters: N={N} K={K} L={L} messages={messages} per SNR')
    print('\nSNR(dB) | BER_SCL | BER_AIFSCL | time_SCL(ms) | time_AIFSCL(ms) | Speedup')
    print('-' * 75)

    csv_path = 'compare_results_scl_vs_aifastscl_real.csv'
    with open(csv_path, 'w') as fh:
        fh.write('snr,ber_scl,ber_aifastscl,time_scl_ms,time_aifastscl_ms,speedup\n')

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

            # AI-Fast-SCL decode
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
            
            if (i + 1) % 100 == 0:
                print(f"  SNR {snr}: {i+1}/{messages} messages...", flush=True)

        # Compute averages
        total_bits = messages * K
        avg_time_scl = np.mean(times_scl)
        avg_time_aifastscl = np.mean(times_aifastscl)

        ber_scl_val = ber_scl / total_bits
        ber_aifastscl_val = ber_aifastscl / total_bits
        
        speedup = avg_time_scl / avg_time_aifastscl if avg_time_aifastscl > 0 else 1.0

        status = "★ FASTER" if speedup > 1.0 else "✗ slower"
        print(f"{snr:5.1f}  | {ber_scl_val:.3e} | {ber_aifastscl_val:.3e} | {avg_time_scl:11.3f} | {avg_time_aifastscl:11.3f} | {speedup:5.2f}x {status}")
        
        with open(csv_path, 'a') as fh:
            fh.write(f"{snr},{ber_scl_val:.6e},{ber_aifastscl_val:.6e},{avg_time_scl:.6f},{avg_time_aifastscl:.6f},{speedup:.3f}\n")

    print('-' * 75)
    print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    run_compare(N=128, K=64, messages=500, L=4)
