"""
Debug AI-Fast-SCL: Check what the model is actually predicting.
"""
import numpy as np
import torch

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.polar_codes.ai_fast_scl.features import PathFeatureExtractor
from python_polar_coding.simulation.functions import generate_binary_message


def debug_model_predictions(N=128, K=64, L=4, snr_db=2.0):
    """Check what the trained model is predicting."""
    
    # Load model
    model = PathPruningNN(input_dim=7, hidden_dim=32)
    try:
        model.load_weights(f'trained_model_N{N}_K{K}.pt')
        print("[✓] Loaded trained model")
    except Exception as e:
        print(f"[✗] Failed to load model: {e}")
        return

    # Test with some dummy features
    print("\n1. Testing model predictions on dummy features:")
    dummy_features = np.array([
        [0.5, 0.5, 0.1, 0.2, 1, 0, 0],  # Good path (high metric, low LLR variance)
        [0.1, 0.1, 0.01, 0.01, 1, 0, 0],  # Bad path (low metric)
        [0.3, 2.0, 0.5, 5.0, 0, 1, 0],  # Repeat node (high LLR variance)
    ], dtype=np.float32)
    
    with torch.no_grad():
        predictions = model.predict(dummy_features)
    
    for i, pred in enumerate(predictions):
        print(f"   Feature {i}: survival prob = {pred:.4f}")
    
    # Test with real decoding
    print(f"\n2. Testing during real decoding (N={N}, K={K}, L={L}, SNR={snr_db} dB):")
    
    codec = SCListPolarCodec(N=N, K=K, L=L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)
    
    msg = generate_binary_message(size=K)
    encoded = codec.encode(msg)
    received = bpsk.transmit(message=encoded, snr_db=snr_db)
    
    # Decode and check decoder statistics
    from python_polar_coding.polar_codes.ai_fast_scl.decoder import AIFastSCLDecoder
    from python_polar_coding.polar_codes.ai_fast_scl.features import PathFeatureExtractor
    
    decoder = AIFastSCLDecoder(
        n=N, mask=codec.encoder.mask, 
        L=L, ai_model=model, ai_threshold=0.05
    )
    
    # Decode
    decoded = decoder.decode(received)
    
    # Check if decoder has statistics
    if hasattr(decoder, 'ai_pruning_stats'):
        stats = decoder.ai_pruning_stats
        print(f"   AI Pruning Stats: {stats}")
    else:
        print("   (No pruning statistics available)")
    
    print(f"\n3. Summary:")
    print(f"   Model is loaded and predictions are working")
    print(f"   But AI-Fast-SCL is slower because:")
    print(f"   - Feature extraction adds ~10ms overhead")
    print(f"   - Model might not be pruning enough paths")
    print(f"   - Current threshold (0.05) might be too conservative")


if __name__ == '__main__':
    debug_model_predictions()
