"""
Test VectorizedAISCL vs SCL to see if array-based path representation is faster.
"""
import time
import numpy as np
import torch

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.vectorized_codec import VectorizedAISCLCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import compute_fails, generate_binary_message


N, K, L, messages = 128, 64, 4, 50
snr_range = [0.0, 1.0, 2.0, 3.0, 4.0]

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

vec_aiscl_codec = VectorizedAISCLCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)
bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

print("Testing VectorizedAISCL vs SCL")
print("SNR(dB) | BER_SCL | BER_VecAISCL | TIME_SCL(ms) | TIME_VecAISCL(ms)")

for snr in snr_range:
    ber_scl = ber_vec = 0
    times_scl = times_vec = []
    times_scl = []
    times_vec = []
    
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
        
        # Vectorized AISCL
        t2 = time.perf_counter()
        try:
            dec_vec = vec_aiscl_codec.decode(tx)
            t3 = time.perf_counter()
            times_vec.append((t3-t2)*1000)
            ber_vec += compute_fails(msg, dec_vec)[0]
        except Exception as e:
            print(f"[Error at SNR={snr}]: {e}")
            times_vec.append(np.nan)
            ber_vec += K
    
    ber_scl_val = ber_scl / (messages * K)
    ber_vec_val = ber_vec / (messages * K)
    avg_time_scl = np.nanmean(times_scl)
    avg_time_vec = np.nanmean(times_vec)
    
    ratio = avg_time_vec / avg_time_scl if avg_time_scl > 0 else np.inf
    
    print(f"{snr:5.1f}  | {ber_scl_val:.6e} | {ber_vec_val:.6e} | {avg_time_scl:8.3f} | {avg_time_vec:8.3f} (ratio: {ratio:.2f}x)")
