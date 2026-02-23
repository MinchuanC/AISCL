"""
Debug extract_simple_features
"""
import numpy as np
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message

def extract_simple_features(path):
    """Extract simple features from SCPath without needing position."""
    try:
        features = []
        
        # 1. Current LLR magnitude (path confidence)
        current_llr = getattr(path, 'current_llr', 0.0)
        features.append(float(np.abs(current_llr)))
        print(f"  Feature 0 (current_llr_mag): {features[0]}")
        
        # 2. Mean magnitude of intermediate LLRs
        intermediate_llr = getattr(path, 'intermediate_llr', [])
        print(f"  intermediate_llr type: {type(intermediate_llr)}, len={len(intermediate_llr) if hasattr(intermediate_llr, '__len__') else 'N/A'}")
        if len(intermediate_llr) > 0:
            mean_llr_mag = float(np.mean(np.abs(np.array(intermediate_llr))))
        else:
            mean_llr_mag = 0.0
        features.append(mean_llr_mag)
        print(f"  Feature 1 (mean_llr_mag): {features[1]}")
        
        # 3. Variance of intermediate LLRs
        if len(intermediate_llr) > 1:
            llr_var = float(np.var(np.array(intermediate_llr)))
        else:
            llr_var = 0.0
        features.append(llr_var)
        print(f"  Feature 2 (llr_var): {features[2]}")
        
        # 4. Min absolute intermediate LLR
        if len(intermediate_llr) > 0:
            min_abs_llr = float(np.min(np.abs(np.array(intermediate_llr))))
        else:
            min_abs_llr = 0.0
        features.append(min_abs_llr)
        print(f"  Feature 3 (min_abs_llr): {features[3]}")
        
        # 5-7. Node type one-hot
        features.extend([0.0, 0.0, 0.0])
        print(f"  Features 4-6 (node_type): {features[4:]}")
        
        result = np.array(features, dtype=np.float32)
        print(f"  Final result type: {type(result)}, shape: {result.shape}, dtype: {result.dtype}")
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None

codec = SCListPolarCodec(N=128, K=64, L=4)
bpsk = SimpleBPSKModulationAWGN(fec_rate=64/128)

msg = generate_binary_message(size=64)
encoded = codec.encode(msg)
received = bpsk.transmit(message=encoded, snr_db=2.0)
decoded = codec.decode(received)

decoder = codec.decoder
print(f"Num paths: {len(decoder.paths)}")
if len(decoder.paths) > 0:
    print(f"\nExtracting features from path 0:")
    features = extract_simple_features(decoder.paths[0])
    print(f"\nResutl: {features}")
    
    # Try to add to list
    all_features = []
    try:
        all_features.append(features)
        print(f"Appended successfully")
        all_array = np.array(all_features, dtype=np.float32)
        print(f"Converted to array: shape {all_array.shape}")
    except Exception as e:
        print(f"Error appending: {e}")
