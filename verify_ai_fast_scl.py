"""
Verification script: Check all AI-Fast-SCL components are working.
"""

import sys
sys.path.insert(0, '.')

import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("\n1. Testing imports...")
    try:
        from python_polar_coding.polar_codes.ai_fast_scl import (
            AIFastSCLPolarCodec, AIFastSCLDecoder, PathPruningNN
        )
        from python_polar_coding.polar_codes.ai_fast_scl.features import PathFeatureExtractor
        from python_polar_coding.polar_codes.ai_fast_scl.utils import (
            create_default_model, load_model_from_file, MockPathPruningNN
        )
        print("   ✓ All imports successful")
        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test NN model creation."""
    print("\n2. Testing neural network model...")
    try:
        from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
        import numpy as np
        import torch
        
        model = PathPruningNN(input_dim=7, hidden_dim=32)
        
        # Test inference
        test_input = np.random.randn(5, 7).astype(np.float32)
        output = model.predict(test_input)
        
        assert output.shape == (5,), f"Output shape mismatch: {output.shape}"
        assert np.all((output >= 0) & (output <= 1)), "Output out of [0, 1] range"
        
        print(f"   ✓ Model created, inference works (5 paths -> {output.shape[0]} predictions)")
        return True
    except Exception as e:
        print(f"   ✗ Model test failed: {e}")
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\n3. Testing feature extraction...")
    try:
        from python_polar_coding.polar_codes.ai_fast_scl.features import PathFeatureExtractor
        import numpy as np
        
        N = 128
        mask = np.ones(N, dtype=int)
        extractor = PathFeatureExtractor(N=N, mask=mask)
        
        # Create mock path object
        class MockPath:
            def __init__(self):
                self._path_metric = -5.0
                self.intermediate_llr = [np.random.randn(10) for _ in range(10)]
        
        path = MockPath()
        features = extractor.extract_features(path, position=64)
        
        assert features.shape == (7,), f"Feature shape mismatch: {features.shape}"
        assert np.all(~np.isnan(features)), "Features contain NaN"
        assert np.all(~np.isinf(features)), "Features contain inf"
        
        print(f"   ✓ Feature extraction works (7-dim vector for 1 path)")
        
        # Test batch extraction
        paths = [MockPath() for _ in range(3)]
        batch_features = extractor.extract_batch_features(paths, position=64)
        
        assert batch_features.shape == (3, 7), f"Batch feature shape mismatch: {batch_features.shape}"
        print(f"   ✓ Batch extraction works ({batch_features.shape[0]} paths -> {batch_features.shape})")
        
        return True
    except Exception as e:
        print(f"   ✗ Feature extraction test failed: {e}")
        traceback.print_exc()
        return False


def test_decoder_creation():
    """Test AI-guided SCL decoder creation."""
    print("\n4. Testing decoder creation...")
    try:
        from python_polar_coding.polar_codes.ai_fast_scl.decoder import AIFastSCLDecoder
        from python_polar_coding.polar_codes.ai_fast_scl.utils import MockPathPruningNN
        import numpy as np
        
        N = 128
        mask = np.ones(N, dtype=int)
        model = MockPathPruningNN()
        
        decoder = AIFastSCLDecoder(
            n=int(np.log2(N)),
            mask=mask,
            is_systematic=True,
            L=4,
            ai_model=model,
            ai_threshold=0.05,
            enable_ai_pruning=True
        )
        
        assert hasattr(decoder, '_ai_prune_paths'), "Missing _ai_prune_paths method"
        assert hasattr(decoder, 'get_statistics'), "Missing statistics method"
        
        print(f"   ✓ Decoder created successfully")
        return True
    except Exception as e:
        print(f"   ✗ Decoder creation failed: {e}")
        traceback.print_exc()
        return False


def test_codec_creation():
    """Test codec creation."""
    print("\n5. Testing codec creation...")
    try:
        from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
        from python_polar_coding.polar_codes.ai_fast_scl.utils import MockPathPruningNN
        
        model = MockPathPruningNN()
        
        codec = AIFastSCLPolarCodec(
            N=128, K=64, L=4,
            ai_model=model,
            ai_threshold=0.05,
            enable_ai_pruning=True
        )
        
        assert hasattr(codec, 'encode'), "Missing encode method"
        assert hasattr(codec, 'decode'), "Missing decode method"
        assert codec.decoder is not None, "Decoder not initialized"
        
        print(f"   ✓ Codec created: ({codec.N}, {codec.K}), List size {codec.L}")
        return True
    except Exception as e:
        print(f"   ✗ Codec creation failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test end-to-end encoding and decoding."""
    print("\n6. Testing end-to-end encode/decode...")
    try:
        from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
        from python_polar_coding.polar_codes.ai_fast_scl.utils import MockPathPruningNN
        from python_polar_coding.channels import SimpleBPSKModulationAWGN
        from python_polar_coding.simulation.functions import generate_binary_message, compute_fails
        import numpy as np
        
        N, K = 128, 64
        
        # Create codec
        model = MockPathPruningNN()
        codec = AIFastSCLPolarCodec(N=N, K=K, L=4, ai_model=model)
        
        # Generate message
        msg = generate_binary_message(size=K)
        
        # Encode
        encoded = codec.encode(msg)
        assert encoded.shape == (N,), f"Encoded shape mismatch: {encoded.shape}"
        
        # Channel
        bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)
        received = bpsk.transmit(encoded, snr_db=3.0)
        
        # Decode
        decoded = codec.decode(received)
        assert decoded.shape == (K,), f"Decoded shape mismatch: {decoded.shape}"
        
        # Check BER
        ber_count = compute_fails(msg, decoded)[0]
        
        print(f"   ✓ End-to-end test passed")
        print(f"     Original message: {msg[:10]}... (length {len(msg)})")
        print(f"     Decoded message:  {decoded[:10]}... (length {len(decoded)})")
        print(f"     Bit errors: {ber_count}/{K} ({100*ber_count/K:.2f}%)")
        
        return True
    except Exception as e:
        print(f"   ✗ Integration test failed: {e}")
        traceback.print_exc()
        return False


def test_statistics():
    """Test statistics tracking."""
    print("\n7. Testing statistics tracking...")
    try:
        from python_polar_coding.polar_codes.ai_fast_scl.codec import AIFastSCLPolarCodec
        from python_polar_coding.polar_codes.ai_fast_scl.utils import MockPathPruningNN
        from python_polar_coding.channels import SimpleBPSKModulationAWGN
        from python_polar_coding.simulation.functions import generate_binary_message
        
        codec = AIFastSCLPolarCodec(N=128, K=64, L=4, 
                                   ai_model=MockPathPruningNN())
        bpsk = SimpleBPSKModulationAWGN(fec_rate=64/128)
        
        # Decode 5 messages
        for _ in range(5):
            msg = generate_binary_message(size=64)
            encoded = codec.encode(msg)
            received = bpsk.transmit(encoded, snr_db=2.0)
            codec.decode(received)
        
        # Get statistics
        stats = codec.decoder.get_statistics()
        
        assert 'ai_calls' in stats, "Missing ai_calls in statistics"
        assert 'ai_pruned_count' in stats, "Missing ai_pruned_count in statistics"
        assert stats['ai_calls'] > 0, "No AI calls recorded"
        
        print(f"   ✓ Statistics tracking works")
        print(f"     AI calls: {stats['ai_calls']}")
        print(f"     Paths pruned: {stats['ai_pruned_count']}")
        print(f"     Avg pruned per call: {stats['avg_pruned_per_call']:.2f}")
        
        return True
    except Exception as e:
        print(f"   ✗ Statistics test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("AI-FAST-SCL VERIFICATION SUITE")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Feature Extraction", test_feature_extraction),
        ("Decoder Creation", test_decoder_creation),
        ("Codec Creation", test_codec_creation),
        ("Integration", test_integration),
        ("Statistics", test_statistics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name}: Unexpected error")
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<50} {status}")
    
    print("=" * 80)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - AI-FAST-SCL is ready for use")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed - please review errors above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
