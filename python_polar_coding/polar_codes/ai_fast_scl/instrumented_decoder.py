"""
Instrumented SCL Decoder for generating real training data.
Records which paths survive to produce the final decoded output.
"""
import numpy as np
from python_polar_coding.polar_codes.sc_list.decoder import SCListDecoder
from python_polar_coding.polar_codes.ai_fast_scl.features import PathFeatureExtractor


class InstrumentedSCLDecoder(SCListDecoder):
    """SCL decoder that records path survival data for training."""
    
    def __init__(self, n, mask, L, feature_dim=7):
        super().__init__(n=n, mask=mask, L=L)
        self.feature_dim = feature_dim
        self.feature_extractor = PathFeatureExtractor(N=n, mask=mask)
        
        # Storage for training data
        self.path_snapshots = []  # List of (position, path_id, features, metric)
        self.final_surviving_path_id = None
    
    def decode_internal(self, received_llr):
        """Decode with instrumentation."""
        
        # Run normal SCL decoding
        decoded = super().decode_internal(received_llr)
        
        # After decoding, identify which path won (has best metric)
        if len(self.paths) > 0:
            best_path = max(self.paths, key=lambda p: -p.metric)  # Note: metric is negative
            self.final_surviving_path_id = id(best_path)
        
        return decoded
    
    def _decode_position(self, position):
        """Override to record path snapshots at each decision point."""
        
        # Record current paths before decoding this position
        if len(self.paths) > 0:
            for path in self.paths:
                try:
                    features = self.feature_extractor.extract_features(path)
                    if features is not None:
                        self.path_snapshots.append({
                            'position': position,
                            'path_id': id(path),
                            'metric': float(path.metric),
                            'features': features.copy()
                        })
                except Exception:
                    pass  # Skip paths with extraction issues
        
        # Run normal decoding
        super()._decode_position(position)
    
    def get_training_labels(self):
        """
        Convert snapshots to training examples.
        Label = 1 if path survives to final decoder, 0 otherwise.
        """
        if self.final_surviving_path_id is None or not self.path_snapshots:
            return None, None
        
        features_list = []
        labels_list = []
        
        for snapshot in self.path_snapshots:
            path_id = snapshot['path_id']
            survived = (path_id == self.final_surviving_path_id)
            
            features_list.append(snapshot['features'])
            labels_list.append(1.0 if survived else 0.0)
        
        if not features_list:
            return None, None
        
        return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.float32)
