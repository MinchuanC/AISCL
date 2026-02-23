"""
Optimized AI-augmented SCL decoder using efficient path representation.
"""
import numpy as np
import torch
from python_polar_coding.polar_codes.base.decoder import BaseDecoder
from python_polar_coding.polar_codes.base.decoding_path import DecodingPathMixin
from python_polar_coding.polar_codes.sc.decoder import SCDecoder


class FastAIPath(DecodingPathMixin, SCDecoder):
    """Lightweight AI path with minimal overhead."""
    def __init__(self, ai_model=None, **kwargs):
        super().__init__(**kwargs)
        self.ai_model = ai_model
    
    def score_ai(self):
        """Fast hybrid score using AI model."""
        if self.ai_model is None:
            return self._path_metric
        try:
            llr_vec = self.intermediate_llr[0]
            bits_vec = self.intermediate_bits[-1]
            if llr_vec is not None and bits_vec is not None:
                return self.ai_model.score(llr_vec, bits_vec)
        except Exception:
            pass
        return self._path_metric


class OptimizedAISCLDecoder(BaseDecoder):
    """
    Optimized AI-augmented SCL using FastAIPath for minimal overhead.
    Uses pre-allocated paths and efficient batch scoring.
    """
    path_class = FastAIPath

    def __init__(self, n: int, mask: np.array, is_systematic: bool = True,
                 L: int = 1, ai_model=None):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self.L = L
        self.ai_model = ai_model
        if self.ai_model is not None:
            self.ai_model.eval()
        
        self.paths = [
            self.path_class(n=n, mask=mask, is_systematic=is_systematic, ai_model=ai_model),
        ]

    @property
    def result(self):
        return [path.result for path in self.paths]

    @property
    def best_result(self):
        return self.result[0]

    def decode_internal(self, received_llr: np.array) -> np.array:
        self._reset_counters()
        self._set_initial_state(received_llr)
        for pos in range(self.N):
            self._decode_position(pos)
        return self.best_result

    def _set_initial_state(self, received_llr):
        for path in self.paths:
            path._set_initial_state(received_llr)

    def _decode_position(self, position):
        self.set_decoder_state(position)
        self._compute_intermediate_alpha(position)
        if self.mask[position] == 1:
            self._populate_paths()
        if self.mask[position] == 0:
            self.set_frozen_value()
        self._update_paths_metrics()
        self._select_best_paths()
        self._compute_bits(position)

    def set_decoder_state(self, position):
        for path in self.paths:
            path._set_decoder_state(position)

    def _compute_intermediate_alpha(self, position):
        for path in self.paths:
            path._compute_intermediate_alpha(position)

    def set_frozen_value(self):
        for path in self.paths:
            path._current_decision = 0

    def _populate_paths(self):
        new_paths = list()
        for path in self.paths:
            split_result = path.split_path()
            new_paths += split_result
        self.paths = new_paths

    def _update_paths_metrics(self):
        for path in self.paths:
            path.update_path_metric()

    def _select_best_paths(self):
        if len(self.paths) <= self.L:
            self.paths = sorted(self.paths, reverse=True)
            return

        if self.ai_model is not None:
            # Optimized batch scoring: collect all features once
            batch_features = []
            batch_indices = []
            
            for i, path in enumerate(self.paths):
                try:
                    llr_vec = path.intermediate_llr[0]
                    bits_vec = path.intermediate_bits[-1]
                    if llr_vec is not None and bits_vec is not None:
                        llr_vec = np.asarray(llr_vec, dtype=np.float32)
                        bits_vec = np.asarray(bits_vec, dtype=np.float32)
                        N = len(llr_vec)
                        bits_pad = np.pad(bits_vec, (0, N - len(bits_vec)), 'constant')
                        X = np.concatenate([llr_vec, bits_pad])
                        batch_features.append(X)
                        batch_indices.append(i)
                except Exception:
                    pass
            
            # Single batch call to model
            ai_scores = np.full(len(self.paths), np.nan)
            if batch_features:
                X_batch = np.stack(batch_features, axis=0)
                X_tensor = torch.from_numpy(X_batch)
                with torch.no_grad():
                    scores = self.ai_model.score_batch(X_tensor)
                if hasattr(scores, 'cpu'):
                    scores = scores.cpu().numpy()
                elif hasattr(scores, 'numpy'):
                    scores = scores.numpy()
                scores = np.asarray(scores, dtype=np.float32)
                for j, idx in enumerate(batch_indices):
                    ai_scores[idx] = scores[j]
            
            # Fill missing scores with path metric
            for i in range(len(self.paths)):
                if np.isnan(ai_scores[i]):
                    ai_scores[i] = self.paths[i]._path_metric
            
            # Hybrid scoring
            ai_scores_norm = (ai_scores - ai_scores.min()) / (ai_scores.max() - ai_scores.min() + 1e-8)
            path_metrics = np.array([path._path_metric for path in self.paths])
            path_metrics_norm = (path_metrics - path_metrics.min()) / (path_metrics.max() - path_metrics.min() + 1e-8)
            hybrid_scores = 0.7 * path_metrics_norm + 0.3 * ai_scores_norm
            idx = np.argsort(hybrid_scores)[-self.L:][::-1]
            self.paths = [self.paths[j] for j in idx]
        else:
            self.paths = sorted(self.paths, reverse=True)[:self.L]

    def _reset_counters(self):
        pass

    def _finalize_counters(self):
        pass

    def _compute_bits(self, position):
        for path in self.paths:
            path._compute_intermediate_beta(position)
            path._update_decoder_state()


