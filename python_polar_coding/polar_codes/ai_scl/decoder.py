import numpy as np
from python_polar_coding.polar_codes.base.decoder import BaseDecoder
from .decoding_path import AIPath

class AISCLDecoder(BaseDecoder):
    """AI-based SCL List decoding."""
    path_class = AIPath

    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 L: int = 1,
                 ai_model=None):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self.L = L
        self.ai_model = ai_model
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
        # reset counters per frame
        self._reset_counters()
        self._set_initial_state(received_llr)
        for pos in range(self.N):
            self._decode_position(pos)
        return self.best_result

    def _set_initial_state(self, received_llr):
        for path in self.paths:
            path._set_initial_state(received_llr)

    def _decode_position(self, position):
        # track current position for adaptive AI scoring frequency
        self.current_pos = position
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
        # original implementation: no instrumentation
        pass

    def set_frozen_value(self):
        for path in self.paths:
            path._current_decision = 0

    def _populate_paths(self):
        new_paths = list()
        for path in self.paths:
            split_result = path.split_path()
            new_paths += split_result
        self.paths = new_paths
        # original implementation: no instrumentation
        pass

    def _update_paths_metrics(self):
        for path in self.paths:
            path.update_path_metric()
        # original implementation: no instrumentation
        pass

    def _select_best_paths(self):
        # original simple selection: hybrid score if ai_model present, otherwise metric only
        if len(self.paths) <= self.L:
            self.paths = sorted(self.paths, reverse=True)
            return

        if self.ai_model is not None:
            ai_scores = np.array([path.score_ai() for path in self.paths])
            ai_scores_norm = (ai_scores - ai_scores.min()) / (ai_scores.max() - ai_scores.min() + 1e-8)
            path_metrics = np.array([path._path_metric for path in self.paths])
            path_metrics_norm = (path_metrics - path_metrics.min()) / (path_metrics.max() - path_metrics.min() + 1e-8)
            hybrid_scores = 0.7 * path_metrics_norm + 0.3 * ai_scores_norm
            idx = np.argsort(hybrid_scores)[-self.L:][::-1]
            self.paths = [self.paths[j] for j in idx]
        else:
            self.paths = sorted(self.paths, reverse=True)[:self.L]

    def _reset_counters(self):
        # no-op: removed instrumentation
        pass

    def _compute_bits(self, position):
        for path in self.paths:
            path._compute_intermediate_beta(position)
            path._update_decoder_state()
        # original implementation: no instrumentation
        pass
