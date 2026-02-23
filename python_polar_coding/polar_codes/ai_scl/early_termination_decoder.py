"""
Early termination AISCL decoder.
When AI detects a dominant path, stop branching and follow it.
Avoids exponential path explosion by pruning aggressively.
"""
import numpy as np
from python_polar_coding.polar_codes.base.decoder import BaseDecoder
from .decoding_path import AIPath


class EarlyTerminationAISCLDecoder(BaseDecoder):
    """AI-based SCL with early termination when one path dominates."""
    path_class = AIPath

    def __init__(self, n: int, mask: np.array, is_systematic: bool = True,
                 L: int = 1, ai_model=None, dominance_threshold: float = 2.0):
        """
        Parameters
        ----------
        dominance_threshold : float
            Number of standard deviations above mean for a path to be considered dominant.
            Lower = more aggressive pruning. Default 2.0 = 95% confidence dominant.
        """
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self.L = L
        self.ai_model = ai_model
        self.dominance_threshold = dominance_threshold
        if self.ai_model is not None:
            try:
                self.ai_model.eval()
            except Exception:
                pass
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
        """Select best paths, using AI for early termination."""
        if len(self.paths) <= self.L:
            self.paths = sorted(self.paths, reverse=True)
            return

        # Get AI scores for early termination check
        if self.ai_model is not None and len(self.paths) > self.L:
            ai_scores = np.array([path.score_ai() for path in self.paths])
            
            # Check if one path is extremely dominant
            ai_mean = ai_scores.mean()
            ai_std = ai_scores.std()
            
            if ai_std > 0:
                # Find the most dominant path
                best_idx = np.argmax(ai_scores)
                best_score = ai_scores[best_idx]
                
                # If dominant path is far above mean, keep only top L paths by AI
                if best_score > ai_mean + self.dominance_threshold * ai_std:
                    # Early termination: just keep top L by AI + metric hybrid
                    ai_scores_norm = (ai_scores - ai_scores.min()) / (ai_scores.max() - ai_scores.min() + 1e-8)
                    path_metrics = np.array([path._path_metric for path in self.paths])
                    path_metrics_norm = (path_metrics - path_metrics.min()) / (path_metrics.max() - path_metrics.min() + 1e-8)
                    
                    # Use higher AI weight since AI detected a clear winner
                    hybrid_scores = 0.5 * path_metrics_norm + 0.5 * ai_scores_norm
                    idx = np.argsort(hybrid_scores)[-self.L:][::-1]
                    self.paths = [self.paths[j] for j in idx]
                    return
            
            # No early termination: use standard hybrid scoring
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
