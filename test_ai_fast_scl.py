"""
Example and test script for AI-guided Fast SCL decoder.

Demonstrates:
1. Creating dummy NN model
2. Basic decoding with AI pruning
3. Comparing with standard SCL baseline
4. Performance metrics
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import time
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
from python_polar_coding.polar_codes.ai_fast_scl.utils import MockPathPruningNN, create_default_model
from python_polar_coding.simulation.functions import compute_fails, generate_binary_message


def test_ai_fast_scl():
    """Test AI-Fast-SCL decoder."""
    
    # Parameters
    N, K, L = 128, 64, 4
    num_messages = 50
    snr_range = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    print("=" * 100)
    print("AI-Fast-SCL Decoder Test")
    print("=" * 100)
    print(f"\nCode: ({N}, {K}) Polar code, List size L={L}, Messages={num_messages}")
    print("\nNote: Using mock NN (returns uniform high probabilities) for demonstration.")
    print("      Replace with trained weights for actual AI-guided pruning.\n")
    
    # Create codecs
    scl_codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=L)
    
    # Create mock NN model for testing
    # In production, load trained weights: PathPruningNN().load_weights('weights.pt')
    mock_nn = MockPathPruningNN()
    
    ai_scl_codec = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L,
        ai_model=mock_nn,
        ai_threshold=0.05,
        enable_ai_pruning=True
    )
    
    # AI-disabled baseline (same architecture, no pruning)
    ai_disabled_codec = AIFastSCLPolarCodec(
        N=N, K=K, design_snr=0.0, L=L,
        ai_model=mock_nn,
        ai_threshold=0.05,
        enable_ai_pruning=False
    )
    
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)
    
    print(f"{'SNR(dB)':<8} {'BER_SCL':<12} {'BER_AISCLv1':<12} {'BER_AIDisabled':<12} "
          f"{'Time_SCL(ms)':<12} {'Time_AISCL(ms)':<12} {'AISCL/SCL':<10}")
    print("-" * 100)
    
    for snr in snr_range:
        ber_scl = ber_aiscl = ber_aidisabled = 0
        times_scl = times_aiscl = times_aidisabled = []
        
        for _ in range(num_messages):
            msg = generate_binary_message(size=K)
            encoded = scl_codec.encode(msg)
            tx = bpsk.transmit(message=encoded, snr_db=snr)
            
            # Standard SCL
            t0 = time.perf_counter()
            dec_scl = scl_codec.decode(tx)
            times_scl.append((time.perf_counter() - t0) * 1000)
            ber_scl += compute_fails(msg, dec_scl)[0]
            
            # AI-Fast-SCL
            t0 = time.perf_counter()
            dec_aiscl = ai_scl_codec.decode(tx)
            times_aiscl.append((time.perf_counter() - t0) * 1000)
            ber_aiscl += compute_fails(msg, dec_aiscl)[0]
            
            # AI-disabled (control)
            t0 = time.perf_counter()
            dec_aidisabled = ai_disabled_codec.decode(tx)
            times_aidisabled.append((time.perf_counter() - t0) * 1000)
            ber_aidisabled += compute_fails(msg, dec_aidisabled)[0]
        
        ber_scl_val = ber_scl / (num_messages * K)
        ber_aiscl_val = ber_aiscl / (num_messages * K)
        ber_aidisabled_val = ber_aidisabled / (num_messages * K)
        
        t_scl_avg = np.mean(times_scl)
        t_aiscl_avg = np.mean(times_aiscl)
        t_aidisabled_avg = np.mean(times_aidisabled)
        
        ratio = t_aiscl_avg / t_scl_avg if t_scl_avg > 0 else 1.0
        
        print(f"{snr:<8.1f} {ber_scl_val:<12.3e} {ber_aiscl_val:<12.3e} {ber_aidisabled_val:<12.3e} "
              f"{t_scl_avg:<12.3f} {t_aiscl_avg:<12.3f} {ratio:<10.3f}x")
        
        # Print decoder statistics
        if hasattr(ai_scl_codec.decoder, 'get_statistics'):
            stats = ai_scl_codec.decoder.get_statistics()
            print(f"       AI Stats: calls={stats['ai_calls']}, "
                  f"pruned={stats['ai_pruned_count']}, "
                  f"avg_per_call={stats['avg_pruned_per_call']:.2f}")
    
    print("\n" + "=" * 100)
    print("Test complete. Expected results:")
    print("- BER should match between SCL and AI-SCL (same algorithm, just with pruning)")
    print("- With trained NN weights, AI-SCL can be faster due to path reduction")
    print("- With mock NN, AI-SCL timing may be slightly slower due to NN inference overhead")
    print("=" * 100)


def test_model_creation():
    """Test NN model creation and basic operations."""
    print("\n" + "=" * 100)
    print("Neural Network Model Test")
    print("=" * 100)
    
    # Create model
    model = create_default_model(input_dim=7, hidden_dim=32)
    print(f"\nCreated PathPruningNN with:")
    print(f"  - Input features: 7 (path_metric, mean_llr, min_llr, llr_var, 3x node_type)")
    print(f"  - Hidden layers: 2 x 32 neurons")
    print(f"  - Output: sigmoid -> survival probability")
    
    # Test inference
    test_features = np.random.randn(8, 7).astype(np.float32)  # 8 paths, 7 features
    predictions = model.predict(test_features)
    print(f"\nTest inference:")
    print(f"  - Input shape: {test_features.shape}")
    print(f"  - Output shape: {predictions.shape}")
    print(f"  - Output range: [{predictions.min():.4f}, {predictions.max():.4f}] (should be [0, 1])")
    
    # Model info
    from python_polar_coding.polar_codes.ai_fast_scl.utils import get_model_info
    info = get_model_info(model)
    print(f"\nModel statistics:")
    for key, val in info.items():
        print(f"  - {key}: {val}")
    
    print("=" * 100)


if __name__ == '__main__':
    try:
        test_model_creation()
        test_ai_fast_scl()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
