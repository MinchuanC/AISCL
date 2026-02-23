"""
Generate real training data from instrumented SCL decoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

import sys
sys.path.insert(0, '.')

from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message


def extract_simple_features(path):
    """Extract simple features from SCPath - very basic version."""
    try:
        features = []
        
        # 1. Current LLR magnitude
        current_llr = getattr(path, 'current_llr', 0.0)
        features.append(float(current_llr) if isinstance(current_llr, (int, float)) else 0.0)
        
        # 2-7: Fill with zeros for now (simplified)
        for _ in range(6):
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    except Exception:
        return None


def generate_real_training_data(N=128, K=64, L=4, num_frames=500, snr_db=2.0, label_percentile=50):
    """Generate training data using LLR-based heuristics from SCL decoder."""
    
    print(f"Generating training data ({num_frames} frames at SNR={snr_db} dB)...")
    
    codec = SCListPolarCodec(N=N, K=K, L=L)
    bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)
    
    all_features = []
    all_labels = []
    samples_collected = 0
    
    for frame_idx in range(num_frames):
        try:
            msg = generate_binary_message(size=K)
            encoded = codec.encode(msg)
            received = bpsk.transmit(message=encoded, snr_db=snr_db)
            
            # Decode normally
            decoded = codec.decode(received)
            
            decoder = codec.decoder
            if hasattr(decoder, 'paths') and len(decoder.paths) > 1:
                # Compute path quality based on intermediate LLR magnitudes
                path_qualities = []
                for path in decoder.paths:
                    if hasattr(path, 'intermediate_llr') and len(path.intermediate_llr) > 0:
                        # Path quality = mean magnitude of intermediate LLRs
                        # intermediate_llr might contain arrays, so flatten carefully
                        llr_list = []
                        for item in path.intermediate_llr:
                            if isinstance(item, np.ndarray):
                                llr_list.extend(item.flatten().tolist())
                            elif isinstance(item, (list, tuple)):
                                llr_list.extend(item)
                            else:
                                try:
                                    llr_list.append(float(item))
                                except (TypeError, ValueError):
                                    pass  # Skip items that can't be converted
                        
                        if len(llr_list) > 0:
                            quality = np.mean(np.abs(np.array(llr_list, dtype=np.float32)))
                        else:
                            quality = 0.0
                        path_qualities.append(quality)
                    else:
                        path_qualities.append(0.0)
                
                if len(path_qualities) > 1:
                    # Collect per-frame feature-quality pairs, then label top tail
                    frame_items = []
                    for i, path in enumerate(decoder.paths):
                        features = extract_simple_features(path)
                        if features is not None:
                            frame_items.append((features, float(path_qualities[i])))

                    if not frame_items:
                        continue

                    # Determine per-frame cutoff so top (100 - label_percentile)% are positive
                    qualities = np.array([q for (_, q) in frame_items], dtype=np.float32)
                    try:
                        cutoff = float(np.percentile(qualities, label_percentile))
                    except Exception as med_err:
                        if frame_idx < 3:
                            print(f"  Percentile error: {str(med_err)[:40]}")
                        continue

                    # Assign labels per-frame
                    for features, q in frame_items:
                        label = 1.0 if q >= cutoff else 0.0
                        all_features.append(features)
                        all_labels.append(label)
                        samples_collected += 1
        except Exception as e:
            if frame_idx < 5:
                print(f"  Frame {frame_idx}: {str(e)[:40]}")
                import traceback
                traceback.print_exc()
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames ({samples_collected} samples)")
    
    if not all_features:
        print("ERROR: No training data collected!")
        return None, None
    
    try:
        features_all = np.array(all_features, dtype=np.float32)
        labels_all = np.array(all_labels, dtype=np.float32)
    except Exception as e:
        print(f"ERROR converting to arrays: {e}")
        print(f"  all_features types: {[type(f) for f in all_features[:3]]}")
        print(f"  all_features shapes: {[f.shape if hasattr(f, 'shape') else 'N/A' for f in all_features[:3]]}")
        return None, None
    
    print(f"\n✓ Collected {len(features_all)} training samples")
    print(f"  Positive (high LLR mag): {labels_all.sum():.0f} ({100*labels_all.mean():.1f}%)")
    
    return features_all, labels_all


def train_model(features, labels, model, epochs=50, batch_size=64, lr=1e-3):
    """Train the model on real data."""
    
    print(f"\nTraining PathPruningNN on {len(features)} real samples...")
    
    # Convert to PyTorch
    features_t = torch.from_numpy(features).float()
    labels_t = torch.from_numpy(labels).float().unsqueeze(1)
    
    # Create dataset
    dataset = TensorDataset(features_t, labels_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
    model.train()
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            predictions = model.forward(batch_features)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    print("Training complete.")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model.forward(features_t)
        predictions_binary = (predictions > 0.5).float()
        
        correct = (predictions_binary == labels_t).float().mean()
        
        # Count true positives and true negatives
        tp = ((predictions_binary == 1) & (labels_t == 1)).float().sum()
        tn = ((predictions_binary == 0) & (labels_t == 0)).float().sum()
        fp = ((predictions_binary == 1) & (labels_t == 0)).float().sum()
        fn = ((predictions_binary == 0) & (labels_t == 1)).float().sum()
        
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        
        print(f"\nModel Evaluation:")
        print(f"  Accuracy:    {correct:.6f}")
        print(f"  Sensitivity: {sensitivity:.6f} (catch paths that survive)")
        print(f"  Specificity: {specificity:.6f} (catch paths to prune)")
        
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--snr', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=str, default='trained_model_real_N128_K64.pt')
    args = parser.parse_args()
    
    print("=" * 80)
    print("PathPruningNN Training with REAL Data")
    print("=" * 80)
    
    # Generate real training data
    # Use 75th percentile cutoff (top 25% are labeled positive)
    features, labels = generate_real_training_data(
        N=args.N, K=args.K, L=args.L,
        num_frames=args.frames, snr_db=args.snr, label_percentile=75
    )
    
    if features is None:
        print("Failed to generate training data!")
        return
    
    # Create and train model
    model = PathPruningNN(input_dim=7, hidden_dim=32)
    model = train_model(features, labels, model, 
                       epochs=args.epochs, batch_size=args.batch_size)
    
    # Save model
    model.save_weights(args.output)
    print(f"\nModel saved to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
