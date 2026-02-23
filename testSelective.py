"""
Test selective AISCL vs SCL and standard AISCL.
Selective AISCL only uses AI when path metrics are similar (uncertainty).
Skips AI when metrics clearly differ, saving inference calls.
"""
import time
import numpy as np
import torch

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.selective_codec import SelectiveAISCLCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import compute_fails, generate_binary_message


N, K, L, messages = 128, 64, 4, 100
snr_range = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

print("Testing Selective AISCL (only AI when uncertain)")
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

# Test different uncertainty thresholds
thresholds = [0.05, 0.10, 0.15, 0.20]

for threshold in thresholds:
    sel_codec = SelectiveAISCLCodec(N=N, K=K, design_snr=0.0, L=L, 
                                    ai_model=ai_model, uncertainty_threshold=threshold)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

    print(f"\nUncertainty Threshold: {threshold:.2f} (use AI if std/mean < {threshold})")
    print("SNR(dB) | BER_SCL | BER_AISCL | BER_SEL | TIME_SCL(ms) | TIME_AISCL(ms) | TIME_SEL(ms) | SEL_vs_SCL")

    for snr in snr_range:
        ber_scl = ber_aiscl = ber_sel = 0
        times_scl = times_aiscl = times_sel = []
        times_scl = []
        times_aiscl = []
        times_sel = []
        
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
            
            # Selective AISCL
            t4 = time.perf_counter()
            dec_sel = sel_codec.decode(tx)
            t5 = time.perf_counter()
            times_sel.append((t5-t4)*1000)
            ber_sel += compute_fails(msg, dec_sel)[0]
        
        ber_scl_val = ber_scl / (messages * K)
        ber_aiscl_val = ber_aiscl / (messages * K)
        ber_sel_val = ber_sel / (messages * K)
        avg_time_scl = np.mean(times_scl)
        avg_time_aiscl = np.mean(times_aiscl)
        avg_time_sel = np.mean(times_sel)
        
        ratio_sel_scl = avg_time_sel / avg_time_scl if avg_time_scl > 0 else np.inf
        marker = "✓ FASTER" if avg_time_sel < avg_time_scl else "slower"
        
        print(f"{snr:5.1f}  | {ber_scl_val:.3e} | {ber_aiscl_val:.3e} | {ber_sel_val:.3e} | {avg_time_scl:8.3f} | {avg_time_aiscl:8.3f} | {avg_time_sel:8.3f} | {ratio_sel_scl:.2f}x ({marker})")
