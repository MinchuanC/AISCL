"""Quick comparison: run SCL vs AISCL on same 50 messages per SNR and check BER/FLOPs match test.py"""
import numpy as np
import torch
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
from python_polar_coding.simulation.functions import compute_fails, generate_binary_message

N, K, L, messages = 128, 64, 4, 50
snr_range = [0.0, 1.0, 2.0, 3.0, 4.0]

# Create codecs
scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
ai_model = PathPruningNet(N)
try:
    ai_model.load_state_dict(torch.load(f'trained_model_N{N}_K{K}.pt'))
    ai_model.eval()
except:
    pass
aiscl_codec = AISCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model)
bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

print("SNR | BER_SCL | BER_AISCL | TIME_SCL(ms) | TIME_AISCL(ms)")
import time
for snr in snr_range:
    ber_scl, ber_aiscl = 0, 0
    times_scl, times_aiscl = [], []
    
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
    
    ber_scl_val = ber_scl / (messages * K)
    ber_aiscl_val = ber_aiscl / (messages * K)
    avg_time_scl = np.mean(times_scl)
    avg_time_aiscl = np.mean(times_aiscl)
    
    print(f"{snr:4.1f} | {ber_scl_val:.6e} | {ber_aiscl_val:.6e} | {avg_time_scl:12.3f} | {avg_time_aiscl:12.3f}")
