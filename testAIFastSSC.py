"""
Test AI-guided FastSSC vs regular SCL and AISCL.
AI-guided FastSSC uses fast SSC as baseline and only branches to multi-path when needed.
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
import torch

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.ai_fast_ssc_codec import AIGuidedFastSSCCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import compute_fails, generate_binary_message


N, K, L, messages = 128, 64, 4, 100
snr_range = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

print("Testing AI-Guided FastSSC vs SCL and AISCL")
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
ai_fastssc_codec = AIGuidedFastSSCCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)

bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

print("\nSNR(dB) | BER_SCL | BER_AISCL | BER_AIFastSSC | TIME_SCL(ms) | TIME_AISCL(ms) | TIME_AIFastSSC(ms) | FastSSC_vs_SCL")

for snr in snr_range:
    ber_scl = ber_aiscl = ber_fastssc = 0
    times_scl = times_aiscl = times_fastssc = []
    times_scl = []
    times_aiscl = []
    times_fastssc = []
    
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
        
        # AI-Guided FastSSC
        t4 = time.perf_counter()
        try:
            dec_fastssc = ai_fastssc_codec.decode(tx)
            t5 = time.perf_counter()
            times_fastssc.append((t5-t4)*1000)
            ber_fastssc += compute_fails(msg, dec_fastssc)[0]
        except Exception as e:
            print(f"[Error at SNR={snr}]: {e}")
            times_fastssc.append(np.nan)
            ber_fastssc += K
    
    ber_scl_val = ber_scl / (messages * K)
    ber_aiscl_val = ber_aiscl / (messages * K)
    ber_fastssc_val = ber_fastssc / (messages * K)
    avg_time_scl = np.nanmean(times_scl)
    avg_time_aiscl = np.nanmean(times_aiscl)
    avg_time_fastssc = np.nanmean(times_fastssc)
    
    ratio_fs = avg_time_fastssc / avg_time_scl if avg_time_scl > 0 else np.inf
    marker = "✓ FASTER" if avg_time_fastssc < avg_time_scl else "slower"
    
    print(f"{snr:5.1f}  | {ber_scl_val:.3e} | {ber_aiscl_val:.3e} | {ber_fastssc_val:.3e} | {avg_time_scl:8.3f} | {avg_time_aiscl:8.3f} | {avg_time_fastssc:8.3f} | {ratio_fs:.2f}x ({marker})")
