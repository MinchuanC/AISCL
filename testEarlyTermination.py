"""
Test early termination AISCL vs SCL and standard AISCL.
Early termination prunes paths aggressively when AI detects a dominant path.
"""
import time
import numpy as np
import torch

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.early_termination_codec import EarlyTerminationAISCLCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import compute_fails, generate_binary_message


N, K, L, messages = 128, 64, 4, 100
snr_range = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

print("Testing Early Termination AISCL")
print("=" * 100)

# Create codecs
scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)

# Load AI model
ai_model = PathPruningNet(N)
try:
    ai_model.load_state_dict(torch.load(f'trained_model_N{N}_K{K}.pt'))
    ai_model.eval()
except:
    print("[Info] AI model not found, using untrained model")
    ai_model.eval()

aiscl_codec = AISCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)

# Test different dominance thresholds
thresholds = [1.5, 2.0, 2.5]

for threshold in thresholds:
    et_codec = EarlyTerminationAISCLCodec(N=N, K=K, design_snr=0.0, L=L, 
                                         ai_model=ai_model, dominance_threshold=threshold)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

    print(f"\nDominance Threshold: {threshold}")
    print("SNR(dB) | BER_SCL | BER_AISCL | BER_ET | TIME_SCL(ms) | TIME_AISCL(ms) | TIME_ET(ms) | ET_Speedup")

    for snr in snr_range:
        ber_scl = ber_aiscl = ber_et = 0
        times_scl = times_aiscl = times_et = []
        times_scl = []
        times_aiscl = []
        times_et = []
        
        for _ in range(messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            tx = bpsk.transmit(message=encoded, snr_db=snr)
            
            # SCL
            t0 = time.perf_counter()
            dec_scl = scl_codec.decode(tx)
            t1 = time.perf_counter()
            times_scl.append((t1-t0)*1000)
            ber_scl += compute_fails(msg, dec_scl)[0]
            
            # AISCL
            t2 = time.perf_counter()
            dec_aiscl = aiscl_codec.decode(tx)
            t3 = time.perf_counter()
            times_aiscl.append((t3-t2)*1000)
            ber_aiscl += compute_fails(msg, dec_aiscl)[0]
            
            # Early Termination AISCL
            t4 = time.perf_counter()
            dec_et = et_codec.decode(tx)
            t5 = time.perf_counter()
            times_et.append((t5-t4)*1000)
            ber_et += compute_fails(msg, dec_et)[0]
        
        ber_scl_val = ber_scl / (messages * K)
        ber_aiscl_val = ber_aiscl / (messages * K)
        ber_et_val = ber_et / (messages * K)
        avg_time_scl = np.mean(times_scl)
        avg_time_aiscl = np.mean(times_aiscl)
        avg_time_et = np.mean(times_et)
        
        et_faster_than_scl = avg_time_et < avg_time_scl
        speedup = avg_time_aiscl / avg_time_et if avg_time_et > 0 else 0
        marker = "✓ FASTER" if et_faster_than_scl else "slower"
        
        print(f"{snr:5.1f}  | {ber_scl_val:.3e} | {ber_aiscl_val:.3e} | {ber_et_val:.3e} | {avg_time_scl:8.3f} | {avg_time_aiscl:8.3f} | {avg_time_et:8.3f} | {speedup:.2f}x vs AISCL ({marker})")
