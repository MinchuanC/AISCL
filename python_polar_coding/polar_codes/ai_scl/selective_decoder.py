"""
Selective AI scoring AISCL decoder.
Only apply AI when path metrics are close (uncertainty).
Skip AI for clear winners/losers to save inference calls.
"""
import numpy as np
from python_polar_coding.polar_codes.base.decoder import BaseDecoder
from .decoding_path import AIPath


class SelectiveAISCLDecoder(BaseDecoder):
    """AI-based SCL with selective AI scoring - only AI when uncertain."""
    path_class = AIPath

    def __init__(self, n: int, mask: np.array, is_systematic: bool = True,
                 L: int = 1, ai_model=None, uncertainty_threshold: float = 0.1):
        """
        Parameters
        ----------
        uncertainty_threshold : float
            If std(metrics) / mean(metrics) < this, metrics are similar enough to use AI.
            Higher = use AI more often. Default 0.1 = only use AI when metrics very close.
        """
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self.L = L
        self.ai_model = ai_model
        self.uncertainty_threshold = uncertainty_threshold
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
        """Select best paths, using AI only when metrics are uncertain."""
        if len(self.paths) <= self.L:
            self.paths = sorted(self.paths, reverse=True)
            return

        # Check metric uncertainty first
        path_metrics = np.array([path._path_metric for path in self.paths])
        metric_mean = path_metrics.mean()
        metric_std = path_metrics.std()
        
        # If metrics are clear, just use them (skip expensive AI scoring)
        if metric_mean > 0 and metric_std / metric_mean < self.uncertainty_threshold:
            # Metrics are decisive: metrics differ by >10%, use only metric
            self.paths = sorted(self.paths, reverse=True)[:self.L]
            return
        
        # Metrics are uncertain: use AI to break ties
        if self.ai_model is not None:
            ai_scores = np.array([path.score_ai() for path in self.paths])
            ai_scores_norm = (ai_scores - ai_scores.min()) / (ai_scores.max() - ai_scores.min() + 1e-8)
            metric_norm = (path_metrics - path_metrics.min()) / (path_metrics.max() - path_metrics.min() + 1e-8)
            
            # Higher AI weight when metrics are uncertain
            hybrid_scores = 0.5 * metric_norm + 0.5 * ai_scores_norm
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
