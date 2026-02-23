import torch
import os
import sys

# Ensure project package can be imported when running from scripts/
sys.path.insert(0, os.getcwd())
from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN

WEIGHT_FILE = 'trained_model_real_N128_K64.pt'
TS_FILE = 'trained_model_real_N128_K64_ts.pt'
ONNX_FILE = 'trained_model_real_N128_K64.onnx'

if not os.path.exists(WEIGHT_FILE):
    print(f'Weight file {WEIGHT_FILE} not found in cwd: {os.getcwd()}')
    raise SystemExit(1)

model = PathPruningNN(input_dim=7, hidden_dim=32)
model.load_weights(WEIGHT_FILE)
model.eval()

# Create dummy input for tracing
dummy = torch.randn(8, 7)

# Trace and save TorchScript
try:
    scripted = torch.jit.trace(model.network, dummy)
    torch.jit.save(scripted, TS_FILE)
    print(f'Saved TorchScript model to {TS_FILE}')
except Exception as e:
    print('TorchScript tracing failed:', e)

# Export ONNX using the full model wrapper
try:
    torch.onnx.export(model, dummy, ONNX_FILE, opset_version=11,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch'}})
    print(f'Saved ONNX model to {ONNX_FILE}')
except Exception as e:
    print('ONNX export failed:', e)
