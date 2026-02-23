"""
Ultra-lightweight AI-augmented SCL decoder using numpy array-based path states.
Avoids per-path SCDecoder objects entirely - instead computes path metrics from LLR arrays.
"""
import numpy as np
import torch
from python_polar_coding.polar_codes.base.decoder import BaseDecoder
from python_polar_coding.polar_codes.sc.decoder import SCDecoder


class VectorizedAISCLDecoder(BaseDecoder):
    """
    Vectorized AISCL using single reference decoder and numpy path states.
    Each path is represented as (llr_snapshot, bits_snapshot, metric) tuple.
    Eliminates per-path object overhead.
    """
    
    def __init__(self, n: int, mask: np.array, is_systematic: bool = True,
                 L: int = 1, ai_model=None):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self.L = L
        self.ai_model = ai_model
        if self.ai_model is not None:
            self.ai_model.eval()
        
        # Reference decoder for LLR computation
        self.ref_decoder = SCDecoder(n=n, mask=mask, is_systematic=is_systematic)
        # Best path decoder for result extraction
        self.best_decoder = None
        
        # Path states: list of (llr_array, bits_array, metric)
        self.paths = []
    
    @property
    def result(self):
        """Return result from best path."""
        if self.best_decoder is not None:
            return self.best_decoder.result
        return None
    
    @property
    def best_result(self):
        return self.result
    
    def decode_internal(self, received_llr: np.array) -> np.array:
        """Vectorized decoding."""
        self.ref_decoder._set_initial_state(received_llr)
        
        # Initialize: single path with zeros
        self.paths = [(
            received_llr.copy(),  # llr snapshot
            np.zeros(1, dtype=np.int32),  # bits
            0.0  # metric
        )]
        
        for pos in range(self.N):
            self._decode_position(pos)
        
        # Reconstruct best path using reference decoder
        self._reconstruct_best_path()
        return self.best_result
    
    def _decode_position(self, pos):
        """Decode one position."""
        # Update reference decoder alpha
        self.ref_decoder._set_decoder_state(pos)
        self.ref_decoder._compute_intermediate_alpha(pos)
        
        # Branch on information bits
        if self.mask[pos] == 1:
            self._branch_paths(pos)
        else:
            # Frozen bit: no branching
            self._update_frozen_paths(pos)
        
        # Select best paths
        if len(self.paths) > self.L:
            self._select_best_paths(pos)
        
        # Update beta
        self.ref_decoder._compute_intermediate_beta(pos)
    
    def _branch_paths(self, pos):
        """Branch each path into two (bit=0 and bit=1)."""
        new_paths = []
        for llr_snap, bits_snap, metric in self.paths:
            # Compute metric delta for bit 0
            llr_val = self.ref_decoder.intermediate_llr[0][0] if self.ref_decoder.intermediate_llr[0] is not None else 0.0
            metric_delta_0 = max(0.0, llr_val) if llr_val >= 0 else -llr_val
            
            # Branch 0: bit=0
            bits_0 = np.append(bits_snap, 0)
            metric_0 = metric + metric_delta_0
            new_paths.append((llr_snap.copy(), bits_0, metric_0))
            
            # Branch 1: bit=1
            metric_delta_1 = max(0.0, -llr_val) if llr_val < 0 else llr_val
            bits_1 = np.append(bits_snap, 1)
            metric_1 = metric + metric_delta_1
            new_paths.append((llr_snap.copy(), bits_1, metric_1))
        
        self.paths = new_paths
    
    def _update_frozen_paths(self, pos):
        """Update metrics for frozen bit (always 0)."""
        updated = []
        for llr_snap, bits_snap, metric in self.paths:
            bits_new = np.append(bits_snap, 0)
            llr_val = self.ref_decoder.intermediate_llr[0][0] if self.ref_decoder.intermediate_llr[0] is not None else 0.0
            metric_delta = max(0.0, llr_val) if llr_val >= 0 else -llr_val
            metric_new = metric + metric_delta
            updated.append((llr_snap, bits_new, metric_new))
        self.paths = updated
    
    def _select_best_paths(self, pos):
        """Select L best paths using hybrid scoring."""
        if self.ai_model is None:
            # No AI: use path metric only
            self.paths.sort(key=lambda x: x[2], reverse=True)
            self.paths = self.paths[:self.L]
            return
        
        # Batch score all paths
        ai_scores = self._batch_score_paths()
        metrics = np.array([p[2] for p in self.paths])
        
        # Normalize
        ai_scores = np.asarray(ai_scores)
        ai_norm = (ai_scores - ai_scores.min()) / (ai_scores.max() - ai_scores.min() + 1e-8)
        metric_norm = (metrics - metrics.min()) / (metrics.max() - metrics.min() + 1e-8)
        
        # Hybrid: prefer strong AI predictions with good metrics
        hybrid = 0.6 * metric_norm + 0.4 * ai_norm
        idx = np.argsort(hybrid)[-self.L:][::-1]
        self.paths = [self.paths[i] for i in idx]
    
    def _batch_score_paths(self):
        """Batch score all paths using AI model."""
        features = []
        for llr_snap, bits_snap, _ in self.paths:
            N = len(llr_snap)
            bits_pad = np.pad(bits_snap, (0, N - len(bits_snap)), 'constant')
            X = np.concatenate([llr_snap, bits_pad])
            features.append(X)
        
        if features:
            X_batch = np.stack(features, axis=0)
            X_tensor = torch.from_numpy(X_batch.astype(np.float32))
            with torch.no_grad():
                scores = self.ai_model.score_batch(X_tensor)
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            elif hasattr(scores, 'numpy'):
                scores = scores.numpy()
            return np.asarray(scores, dtype=np.float32).tolist()
        
        return [0.0] * len(self.paths)
    
    def _reconstruct_best_path(self):
        """Reconstruct best path using stored bits."""
        if not self.paths:
            return
        
        # Find best path by metric
        best_idx = np.argmax([p[2] for p in self.paths])
        _, best_bits, _ = self.paths[best_idx]
        
        # Reconstruct by re-decoding with best bits
        self.best_decoder = SCDecoder(n=self.N, mask=self.mask, is_systematic=self.is_systematic)
        self.best_decoder._set_initial_state(self.ref_decoder.intermediate_llr[0])
        
        # Re-execute decoding with stored decisions
        for pos in range(self.N):
            self.best_decoder._set_decoder_state(pos)
            self.best_decoder._compute_intermediate_alpha(pos)
            if pos < len(best_bits):
                self.best_decoder._current_decision = best_bits[pos]
            self.best_decoder.update_path_metric()
            self.best_decoder._compute_intermediate_beta(pos)
            self.best_decoder._update_decoder_state()
