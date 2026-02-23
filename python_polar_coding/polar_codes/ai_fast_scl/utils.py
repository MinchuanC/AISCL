"""Utilities for AI-guided Fast SCL path pruning network."""

import os
import numpy as np
import torch

from .nn import PathPruningNN


def create_default_model(input_dim: int = 7, hidden_dim: int = 32) -> PathPruningNN:
    """
    Create a new PathPruningNN with default architecture.
    
    Parameters
    ----------
    input_dim : int
        Input dimension (default 7 features)
    hidden_dim : int
        Hidden layer dimension (default 32 neurons)
    
    Returns
    -------
    PathPruningNN
        Newly initialized model
    """
    return PathPruningNN(input_dim=input_dim, hidden_dim=hidden_dim)


def load_model_from_file(filepath: str, input_dim: int = 7, hidden_dim: int = 32) -> PathPruningNN:
    """
    Load a pre-trained PathPruningNN from file.
    
    Parameters
    ----------
    filepath : str
        Path to saved model weights
    input_dim : int
        Input dimension (must match training)
    hidden_dim : int
        Hidden layer dimension (must match training)
    
    Returns
    -------
    PathPruningNN
        Model with loaded weights
    """
    model = PathPruningNN(input_dim=input_dim, hidden_dim=hidden_dim)
    if os.path.exists(filepath):
        model.load_weights(filepath)
    return model


def save_model(model: PathPruningNN, filepath: str):
    """
    Save model weights to file.
    
    Parameters
    ----------
    model : PathPruningNN
        Model to save
    filepath : str
        Destination file path
    """
    model.save_weights(filepath)


class MockPathPruningNN(PathPruningNN):
    """
    Mock NN for testing without pre-trained weights.
    
    Returns uniform high probabilities (all paths survive NN filter).
    Useful for testing integration without AI weights.
    """
    
    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return uniform high probabilities."""
        if isinstance(features, np.ndarray):
            num_paths = features.shape[0]
        else:
            num_paths = 1
        return np.ones(num_paths, dtype=np.float32) * 0.9  # All paths survive


def get_model_info(model: PathPruningNN) -> dict:
    """
    Get information about model architecture and parameters.
    
    Parameters
    ----------
    model : PathPruningNN
        Model to inspect
    
    Returns
    -------
    dict
        Model information including parameter count
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': 'PathPruningNN',
        'architecture': '3-layer MLP with ReLU + Sigmoid',
    }
