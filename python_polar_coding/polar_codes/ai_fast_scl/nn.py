"""Neural network module for AI-based path pruning in SCL decoding."""

import numpy as np
import torch
import torch.nn as nn


class PathPruningNN(nn.Module):
    """
    Lightweight MLP for predicting path survival probability.
    
    Input features (7 total):
    - current_path_metric: normalized path metric
    - mean_llr_magnitude: mean |LLR| over current bits
    - min_abs_llr: minimum absolute LLR value
    - llr_variance: variance of LLR values
    - node_type_onehot: [is_rate1, is_rep, is_spc] (3 one-hot features)
    
    Output:
    - survival_probability: [0, 1] probability this path will survive pruning
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32):
        """
        Initialize path pruning neural network.
        
        Parameters
        ----------
        input_dim : int
            Number of input features (default 7)
        hidden_dim : int
            Number of neurons in hidden layers (default 32, ~128 total parameters)
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Batch of input features, shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Survival probabilities, shape (batch_size, 1)
        """
        return self.network(x)
    
    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict path survival probabilities (inference mode).
        
        Parameters
        ----------
        features : np.ndarray
            Array of shape (num_paths, num_features)
        
        Returns
        -------
        np.ndarray
            Survival probabilities, shape (num_paths,)
        """
        x = torch.from_numpy(features).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        probs = self.forward(x).squeeze(-1)
        return probs.cpu().numpy()
    
    def save_weights(self, filepath: str):
        """Save model weights to file."""
        torch.save(self.state_dict(), filepath)
    
    def load_weights(self, filepath: str):
        """Load model weights from file."""
        self.load_state_dict(torch.load(filepath, map_location='cpu'))
        self.eval()


def extract_node_type_features(position: int, mask: np.ndarray, N: int) -> np.ndarray:
    """
    Extract node type features (one-hot encoding).
    
    Returns [is_rate1, is_rep, is_spc] based on polar node structure.
    For simplicity, currently returns [0, 0, 0] (normal node).
    Can be extended with actual node type detection.
    
    Parameters
    ----------
    position : int
        Current bit position
    mask : np.ndarray
        Polar code mask
    N : int
        Code length
    
    Returns
    -------
    np.ndarray
        Shape (3,) one-hot node type features
    """
    # TODO: Implement actual node type detection from polar code structure
    # For now, all nodes are treated as normal (Rate-0/Rate-1 leaf)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)
