"""Feature extraction utilities for AI-based path pruning."""

import numpy as np


class PathFeatureExtractor:
    """Extract features from SCL paths for neural network inference."""
    
    def __init__(self, N: int, mask: np.ndarray):
        """
        Initialize feature extractor.
        
        Parameters
        ----------
        N : int
            Code length
        mask : np.ndarray
            Polar code mask
        """
        self.N = N
        self.mask = mask
    
    def extract_features(self, path, position: int) -> np.ndarray:
        """
        Extract features from a single path for NN prediction.
        
        Features (7 total):
        [path_metric_norm, mean_llr_mag, min_abs_llr, llr_variance, 
         node_type_rate1, node_type_rep, node_type_spc]
        
        Parameters
        ----------
        path : SCPath
            Decoding path object
        position : int
            Current decoding position
        
        Returns
        -------
        np.ndarray
            Shape (7,) feature vector
        """
        features = []
        
        # 1. Normalized path metric (log-domain, inverted for probability)
        # Higher metric = better path, so we use exp(-metric) for probability
        path_metric = getattr(path, '_path_metric', 0.0)
        path_metric_norm = float(np.exp(min(path_metric, 100)) / (1 + np.exp(min(path_metric, 100))))
        features.append(path_metric_norm)
        
        # 2-4. LLR statistics from current intermediate alpha values
        # Get the LLR vector at root or current leaf
        llr_vec = self._get_path_llr_values(path, position)
        
        if len(llr_vec) > 0:
            mean_llr_mag = float(np.mean(np.abs(llr_vec)))
            min_abs_llr = float(np.min(np.abs(llr_vec)))
            llr_variance = float(np.var(llr_vec))
        else:
            mean_llr_mag = 0.0
            min_abs_llr = 0.0
            llr_variance = 0.0
        
        features.append(min(mean_llr_mag, 100.0))  # Clip for numerical stability
        features.append(min(min_abs_llr, 100.0))
        features.append(min(llr_variance, 100.0))
        
        # 5-7. Node type (one-hot: Rate-1, Rep, SPC) - currently all zeros (normal nodes)
        # Can be extended with actual node type detection
        features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_batch_features(self, paths: list, position: int) -> np.ndarray:
        """
        Extract features from multiple paths.
        
        Parameters
        ----------
        paths : list
            List of SCPath objects
        position : int
            Current decoding position
        
        Returns
        -------
        np.ndarray
            Shape (num_paths, 7) feature matrix
        """
        features_list = []
        for path in paths:
            features = self.extract_features(path, position)
            features_list.append(features)
        
        return np.vstack(features_list) if features_list else np.zeros((0, 7), dtype=np.float32)
    
    def _get_path_llr_values(self, path, position: int) -> np.ndarray:
        """
        Extract LLR values from a path's current state.
        
        Parameters
        ----------
        path : SCPath
            Decoding path
        position : int
            Current position
        
        Returns
        -------
        np.ndarray
            LLR values seen so far (up to position)
        """
        try:
            # Try to get intermediate alpha (LLR) from path
            # This depends on the path implementation
            if hasattr(path, 'intermediate_llr') and isinstance(path.intermediate_llr, (list, tuple)):
                llr_list = []
                for alpha in path.intermediate_llr[:position+1]:
                    if alpha is not None and hasattr(alpha, '__len__'):
                        llr_list.extend(np.abs(alpha).flatten())
                if llr_list:
                    return np.array(llr_list[:100])  # Limit to first 100 values
            
            if hasattr(path, 'current_llr'):
                return np.abs(path.current_llr).flatten()
        except Exception:
            pass
        
        # Fallback: return empty array
        return np.array([], dtype=np.float32)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to improve NN performance.
        
        Parameters
        ----------
        features : np.ndarray
            Shape (num_paths, 7) raw features
        
        Returns
        -------
        np.ndarray
            Shape (num_paths, 7) normalized features (clipped to [0, 1])
        """
        # Clip to reasonable ranges
        normalized = features.copy()
        
        # Path metric already normalized in extract_features
        # LLR magnitudes clipped in extract_features
        # Node types are already one-hot
        
        # Ensure all values are in [0, 1] by clipping
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
