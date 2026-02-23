"""
Test AI-guided FastSSC v2 with intelligent SCL fallback.
Compares: SCL vs AISCL vs FastSSC vs AIGuidedFastSSC
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
from python_polar_coding.polar_codes.ai_scl.ai_fast_ssc_codec_v2 import AIGuidedFastSSCCodecV2
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import compute_fails, generate_binary_message


N, K, L, messages = 128, 64, 4, 100
snr_range = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

print("Testing AI-Guided FastSSC v2 with SCL Fallback")
print("=" * 140)

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
fastssc_codec = AIGuidedFastSSCCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)
aiguidedv2_codec = AIGuidedFastSSCCodecV2(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model, scl_threshold=0.5)

bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

print("\nSNR | BER_SCL | BER_AIS | BER_FS | BER_AIGv2 | TIME_SCL | TIME_AIS | TIME_FS | TIME_AIGv2 | FS_vs_SCL | AIGv2_RATIO | AIGv2_SCLFallback")
print("-" * 140)

for snr in snr_range:
    ber_scl = ber_aiscl = ber_fastssc = ber_aiguidedv2 = 0
    times_scl = times_aiscl = times_fastssc = times_aiguidedv2 = [], [], [], []
    
    for msg_idx in range(messages):
        msg = generate_binary_message(size=K)
        encoded = scl_codec.encode(msg)
        tx = bpsk.transmit(message=encoded, snr_db=snr)
        
        # SCL
        t0 = time.perf_counter()
        dec_scl = scl_codec.decode(tx)
        times_scl.append((time.perf_counter()-t0)*1000)
        ber_scl += compute_fails(msg, dec_scl)[0]
        
        # AISCL
        t0 = time.perf_counter()
        dec_aiscl = aiscl_codec.decode(tx)
        times_aiscl.append((time.perf_counter()-t0)*1000)
        ber_aiscl += compute_fails(msg, dec_aiscl)[0]
        
        # FastSSC baseline
        t0 = time.perf_counter()
        dec_fs = fastssc_codec.decode(tx)
        times_fastssc.append((time.perf_counter()-t0)*1000)
        ber_fastssc += compute_fails(msg, dec_fs)[0]
        
        # AI-guided v2
        t0 = time.perf_counter()
        dec_aigv2 = aiguidedv2_codec.decode(tx)
        times_aiguidedv2.append((time.perf_counter()-t0)*1000)
        ber_aiguidedv2 += compute_fails(msg, dec_aigv2)[0]
    
    ber_scl_val = ber_scl / (messages * K)
    ber_aiscl_val = ber_aiscl / (messages * K)
    ber_fastssc_val = ber_fastssc / (messages * K)
    ber_aiguidedv2_val = ber_aiguidedv2 / (messages * K)
    
    t_scl = np.mean(times_scl)
    t_aiscl = np.mean(times_aiscl)
    t_fs = np.mean(times_fastssc)
    t_aigv2 = np.mean(times_aiguidedv2)
    
    ratio_fs = t_fs / t_scl if t_scl > 0 else np.inf
    ratio_aigv2 = t_aigv2 / t_scl if t_scl > 0 else np.inf
    scl_fallback_pct = (aiguidedv2_codec.decoder.use_scl_count / messages * 100) if hasattr(aiguidedv2_codec.decoder, 'use_scl_count') else 0
    
    print(f"{snr:3.1f} | {ber_scl_val:.2e} | {ber_aiscl_val:.2e} | {ber_fastssc_val:.2e} | {ber_aiguidedv2_val:.2e} | {t_scl:7.2f} | {t_aiscl:7.2f} | {t_fs:6.2f} | {t_aigv2:7.2f} | {ratio_fs:7.2f}x | {ratio_aigv2:7.2f}x | {scl_fallback_pct:5.1f}%")

print("\nSummary: AIGuidedFastSSC v2 = FastSSC + intelligent SCL fallback")
print("Goal: Achieve FastSSC speed at high SNR, SCL performance at low SNR")
