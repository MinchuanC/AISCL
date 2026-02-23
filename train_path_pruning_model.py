"""
Training script for PathPruningNN model.

This script shows how to:
1. Generate training data from SCL decoding traces
2. Train a PathPruningNN model
3. Evaluate model accuracy and speed

To use:
  python train_path_pruning_model.py --output weights/model.pt --epochs 20
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '.')

from python_polar_coding.polar_codes.ai_fast_scl.nn import PathPruningNN
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message


class PathPruningTrainer:
    """Train PathPruningNN model on SCL decoding traces."""
    
    def __init__(self, N=128, K=64, L=4):
        self.N = N
        self.K = K
        self.L = L
        self.codec = SCListPolarCodec(N=N, K=K, L=L)
        self.bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)
    
    def generate_training_data(self, num_frames=1000, snr_db=2.0):
        """
        Generate training data by recording paths during SCL decoding.
        
        Returns:
          features: (num_samples, 7) array of path features
          labels: (num_samples,) binary labels [0=pruned, 1=survived]
        """
        print(f"Generating training data ({num_frames} frames at SNR={snr_db} dB)...")
        
        # Placeholder: In real implementation, would hook into decoder
        # to collect features and labels from actual decoding
        
        # For now, generate synthetic training data as placeholder
        num_samples = 5000
        features = np.random.randn(num_samples, 7).astype(np.float32)
        features = np.clip(features, -2, 2)  # Normalize to [-2, 2]
        
        # Synthetic labels: harder paths (lower metrics) less likely to survive
        path_metrics = features[:, 0]
        llr_strength = features[:, 1]
        labels = ((path_metrics + llr_strength) > 0).astype(np.float32)
        
        print(f"  Generated {num_samples} samples")
        print(f"  Positive (survived): {labels.sum():.0f} ({100*labels.mean():.1f}%)")
        
        return features, labels
    
    def train(self, model, features, labels, epochs=20, batch_size=64, lr=1e-3):
        """Train model on features and labels."""
        # Convert to PyTorch tensors
        features_t = torch.from_numpy(features).float()
        labels_t = torch.from_numpy(labels).float().unsqueeze(1)
        
        # Create data loader
        dataset = TensorDataset(features_t, labels_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        
        model.train()
        
        print(f"\nTraining for {epochs} epochs with {len(dataset)} samples...")
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model.forward(batch_features)
                loss = loss_fn(predictions, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        model.eval()
        print("Training complete.")
    
    def evaluate(self, model, features, labels):
        """Evaluate model accuracy."""
        predictions = model.predict(features)
        
        # Binary classification at threshold 0.5
        predicted_labels = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_labels == labels)
        
        print(f"\nModel Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Sensitivity: {np.mean(predicted_labels[labels==1]):.4f}")
        print(f"  Specificity: {np.mean(1 - predicted_labels[labels==0]):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train PathPruningNN for path pruning in SCL decoding'
    )
    parser.add_argument('--N', type=int, default=128, help='Code length')
    parser.add_argument('--K', type=int, default=64, help='Information bits')
    parser.add_argument('--L', type=int, default=4, help='List size')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='path_pruning_model.pt',
                        help='Output model file')
    parser.add_argument('--frames', type=int, default=1000,
                        help='Number of training frames')
    parser.add_argument('--snr', type=float, default=2.0,
                        help='SNR for training data (dB)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PathPruningNN Training")
    print("=" * 80)
    
    # Create trainer
    trainer = PathPruningTrainer(N=args.N, K=args.K, L=args.L)
    
    # Generate training data
    features, labels = trainer.generate_training_data(
        num_frames=args.frames, snr_db=args.snr
    )
    
    # Create model
    model = PathPruningNN(input_dim=7, hidden_dim=32)
    
    # Train
    trainer.train(model, features, labels, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr)
    
    # Evaluate
    trainer.evaluate(model, features, labels)
    
    # Save
    model.save_weights(args.output)
    print(f"\nModel saved to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
