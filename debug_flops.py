#!/usr/bin/env python3
"""Debug FLOP counting."""
from python_polar_coding.polar_codes.ai_scl.codec import AISCLPolarCodec
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.channels import SimpleBPSKModulationAWGN
import torch
import numpy as np

# Load model
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet
model_state = torch.load('trained_model_N128_K64.pt', map_location='cpu')
model = PathPruningNet(128)
model.load_state_dict(model_state)
model.eval()

# Test on N=128
codec_scl = SCListPolarCodec(N=128, K=64, design_snr=1.0, L=4)
codec_ai = AISCLPolarCodec(N=128, K=64, design_snr=1.0, L=4, ai_model=model)
channel = SimpleBPSKModulationAWGN(fec_rate=0.5)

msg = np.random.randint(0, 2, 64)
encoded = codec_scl.encode(msg)
received_llr = channel.transmit(encoded, snr_db=2.0)

# SCL decode
result_scl = codec_scl.decoder.decode_internal(received_llr)
print(f"SCL FLOPs: {codec_scl.decoder.op_counts['flops']}")
print(f"  Alpha per pos: {codec_scl.decoder.op_counts['flops'] * 0.5 / 128:.1f}")  # rough estimate
print(f"  Populate calls: {codec_scl.decoder.op_counts['populate_calls']}")
print(f"  Select calls: {codec_scl.decoder.op_counts['select_calls']}")
print()

# AISCL decode
result_ai = codec_ai.decoder.decode_internal(received_llr)
print(f"AISCL FLOPs: {codec_ai.decoder.op_counts['flops']}")
print(f"  Populate calls: {codec_ai.decoder.op_counts['populate_calls']}")
print(f"  Select calls: {codec_ai.decoder.op_counts['select_calls']}")
print()

print(f"FLOP ratio (AISCL/SCL): {codec_ai.decoder.op_counts['flops'] / codec_scl.decoder.op_counts['flops']:.2f}x")
