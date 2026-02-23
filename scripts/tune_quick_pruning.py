import os
import sys
import time
import numpy as np
# Ensure package root is importable when running from scripts/
sys.path.insert(0, os.getcwd())
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message, compute_fails

# Grid to search
PERCENTILES = [40.0, 50.0, 60.0]
TOPK_MULTIPLIERS = [1.5, 2.0, 3.0]

# Benchmark settings
N = 128
K = 64
L = 4
MESSAGES = 200
SNR_RANGE = [0.0, 1.0, 2.0]

# Load trained model
ai_model = PathPruningNN(input_dim=7, hidden_dim=32)
ai_model.load_weights('trained_model_real_N128_K64.pt')
ai_model.eval()
# Attach TorchScript network if available
try:
    import torch
    ts_path = 'trained_model_real_N128_K64_ts.pt'
    if os.path.exists(ts_path):
        scripted_net = torch.jit.load(ts_path, map_location='cpu')
        ai_model.network = scripted_net
        print(f"Attached TorchScript network from {ts_path}")
except Exception:
    pass

bpsk = SimpleBPSKModulationAWGN(fec_rate=K / N)

results = []

for pct in PERCENTILES:
    for mul in TOPK_MULTIPLIERS:
        print(f"Testing quick_percentile={pct}, topk_multiplier={mul}")
        # Create codecs
        scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
        ai_codec = AIFastSCLPolarCodec(N=N, K=K, design_snr=0.0, L=L, ai_model=ai_model, ai_threshold=0.5, enable_ai_pruning=True)
        # set quick params
        ai_codec.quick_percentile = pct
        ai_codec.topk_multiplier = mul

        # init decoders
        scl_decoder = scl_codec.init_decoder()
        ai_decoder = ai_codec.init_decoder()

        times_scl = []
        times_ai = []
        ber_scl = 0
        ber_ai = 0

        for snr in SNR_RANGE:
            for i in range(MESSAGES):
                msg = generate_binary_message(size=K)
                encoded = scl_codec.encode(msg)
                received = bpsk.transmit(message=encoded, snr_db=snr)

                t0 = time.perf_counter()
                dec_scl = scl_codec.decode(received)
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                dec_ai = ai_codec.decode(received)
                t3 = time.perf_counter()

                times_scl.append((t1 - t0) * 1000.0)
                times_ai.append((t3 - t2) * 1000.0)

                be_scl, _ = compute_fails(msg, dec_scl)
                be_ai, _ = compute_fails(msg, dec_ai)
                ber_scl += be_scl
                ber_ai += be_ai

        avg_scl = np.mean(times_scl)
        avg_ai = np.mean(times_ai)
        speedup = avg_scl / avg_ai if avg_ai > 0 else 1.0
        ber_scl_val = ber_scl / (MESSAGES * K * len(SNR_RANGE))
        ber_ai_val = ber_ai / (MESSAGES * K * len(SNR_RANGE))

        results.append((pct, mul, speedup, avg_scl, avg_ai, ber_scl_val, ber_ai_val))
        print(f"Result pct={pct} mul={mul} -> speedup={speedup:.3f}, BER_SCL={ber_scl_val:.3e}, BER_AI={ber_ai_val:.3e}")

# Pick best speedup with no BER regression
best = None
for r in results:
    pct, mul, speedup, *_ = r
    if best is None or speedup > best[2]:
        best = r

print('\nGrid results:')
for r in results:
    print(r)

print('\nBest setting:')
print(best)
