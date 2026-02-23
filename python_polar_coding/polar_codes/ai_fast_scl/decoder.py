"""AI-guided Fast SCL decoder with neural network-based path pruning."""

import numpy as np
import torch

from python_polar_coding.polar_codes.sc_list.decoder import SCListDecoder
from python_polar_coding.polar_codes.sc_list.decoding_path import SCPath

from .nn import PathPruningNN
from .features import PathFeatureExtractor


class AIFastSCLDecoder(SCListDecoder):
    """
    SCL decoder with AI-guided path pruning.
    
    Before metric-based selection, uses a neural network to predict path survival
    probability and discards low-probability paths. Falls back to metric-only pruning
    if too many paths are removed by the AI.
    """
    
    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 L: int = 4,
                 ai_model: PathPruningNN = None,
                 ai_threshold: float = 0.05,
                 enable_ai_pruning: bool = True,
                 force_quick_pruning: bool = True,
                 quick_percentile: float = 50.0,
                 topk_multiplier: float = 2.0):
        """
        Initialize AI-guided SCL decoder.
        
        Parameters
        ----------
        n : int
            log2(N)
        mask : np.ndarray
            Polar code mask
        is_systematic : bool
            Systematic encoding
        L : int
            List size (number of paths to maintain)
        ai_model : PathPruningNN or None
            Pre-trained neural network for path pruning.
            If None, falls back to standard SCL (no AI pruning).
        ai_threshold : float
            Probability threshold for path pruning. Default 0.05.
            Paths with predicted survival probability < threshold are discarded
            (unless fallback is triggered).
        enable_ai_pruning : bool
            Whether to enable AI pruning. Can be disabled to compare against baseline.
        """
        super().__init__(n=n, mask=mask, is_systematic=is_systematic, L=L)
        
        self.ai_model = ai_model
        self.ai_threshold = ai_threshold
        self.enable_ai_pruning = enable_ai_pruning and (ai_model is not None)
        # If True, apply the cheap quick-pruning heuristic and skip NN
        self.force_quick_pruning = force_quick_pruning
        
        # Feature extractor for NN input
        self.feature_extractor = PathFeatureExtractor(N=self.N, mask=self.mask)
        
        # Statistics (for monitoring)
        self.ai_pruned_count = 0  # Paths removed by AI
        self.ai_calls = 0  # Number of times AI pruning was invoked
        # Quick-pruning tuning parameters
        self.quick_percentile = float(quick_percentile)
        self.topk_multiplier = float(topk_multiplier)
    
    def _decode_position(self, position):
        """
        Single step of SCL-decoding with AI-guided pruning.
        
        Modified flow:
        1. Set decoder state
        2. Compute intermediate alpha values
        3. If information bit: populate paths (branch)
        4. Update path metrics
        5. **NEW: AI-guided path pruning**
        6. Select best L paths
        7. Compute bits
        """
        self.set_decoder_state(position)
        self._compute_intermediate_alpha(position)
        
        if self.mask[position] == 1:
            self._populate_paths()
        if self.mask[position] == 0:
            self.set_frozen_value()
        
        self._update_paths_metrics()
        
        # AI-guided path pruning (new step)
        if self.enable_ai_pruning and len(self.paths) > self.L:
            self._ai_prune_paths(position)
        
        self._select_best_paths()
        self._compute_bits(position)
    
    def _ai_prune_paths(self, position: int):
        """
        AI-guided path pruning before metric-based selection.
        
        Discards candidate paths with low predicted survival probability,
        with safety fallback to metric-only pruning if too many are removed.
        
        Parameters
        ----------
        position : int
            Current decoding position
        """
        if not self.enable_ai_pruning or self.ai_model is None or len(self.paths) <= self.L:
            return
        
        try:
            self.ai_calls += 1

            # Quick heuristic pruning (cheap): compute simple score per path
            # using current_llr or mean |intermediate_llr| to remove obviously bad paths
            quick_scores = []
            for path in self.paths:
                try:
                    cllr = getattr(path, 'current_llr', None)
                    if isinstance(cllr, (int, float)):
                        score = abs(cllr)
                    else:
                        # Try flattening current_llr or intermediate_llr
                        try:
                            arr = np.array(cllr)
                            score = float(np.mean(np.abs(arr).flatten())) if arr.size > 0 else 0.0
                        except Exception:
                            # fallback to intermediate_llr
                            lst = []
                            if hasattr(path, 'intermediate_llr'):
                                for a in path.intermediate_llr:
                                    try:
                                        ar = np.array(a)
                                        lst.extend(np.abs(ar).flatten().tolist())
                                    except Exception:
                                        continue
                            score = float(np.mean(lst)) if lst else 0.0
                except Exception:
                    score = 0.0
                quick_scores.append(score)

            # Keep top portion by quick score as a cheap reduction (if that leaves > L)
            try:
                cutoff_quick = float(np.percentile(np.array(quick_scores), self.quick_percentile))
            except Exception:
                cutoff_quick = 0.0

            quick_mask = np.array(quick_scores) >= cutoff_quick
            if np.sum(quick_mask) > self.L and np.sum(quick_mask) < len(self.paths):
                reduced_paths = [p for i, p in enumerate(self.paths) if quick_mask[i]]
            else:
                reduced_paths = self.paths

            # Also ensure we keep top-by-metric candidates: union of quick and metric top-k
            try:
                metric_scores = np.array([getattr(p, '_path_metric', 0.0) for p in self.paths], dtype=float)
                # Determine k as a multiplier of L (at least L+1)
                k = max(int(self.topk_multiplier * self.L), self.L + 1)
                k = min(k, len(metric_scores))
                top_metric_idxs = np.argpartition(-metric_scores, kth=k - 1)[:k]
                # Build combined keep mask
                combined_mask = np.array(quick_mask, dtype=bool)
                combined_mask[top_metric_idxs] = True
            except Exception:
                combined_mask = quick_mask

            # If configured to force quick pruning (fast path) or no AI model, apply combined quick/metric pruning
            if self.force_quick_pruning or self.ai_model is None:
                if np.sum(combined_mask) >= self.L and np.sum(combined_mask) < len(self.paths):
                    old_count = len(self.paths)
                    self.paths = [p for i, p in enumerate(self.paths) if combined_mask[i]]
                    self.ai_pruned_count += old_count - len(self.paths)
                return

            # Extract features from reduced set and run NN in batch
            features = self.feature_extractor.extract_batch_features(reduced_paths, position)
            features = self.feature_extractor.normalize_features(features)

            with torch.no_grad():
                predictions = self.ai_model.predict(features)  # Shape: (num_reduced,)

            ai_mask = predictions >= self.ai_threshold

            # Map ai_mask back to original paths
            if reduced_paths is not self.paths:
                # build final keep mask over original paths
                final_keep = []
                ri = 0
                for i in range(len(self.paths)):
                    if quick_mask[i]:
                        final_keep.append(bool(ai_mask[ri]))
                        ri += 1
                    else:
                        # path was removed by quick pruning
                        final_keep.append(False)
                final_keep = np.array(final_keep, dtype=bool)
            else:
                final_keep = np.array(ai_mask, dtype=bool)

            remaining = np.sum(final_keep)
            if remaining < self.L:
                # fallback to metric-only: do nothing
                return

            # Apply final pruning
            old_count = len(self.paths)
            self.paths = [p for i, p in enumerate(self.paths) if final_keep[i]]
            self.ai_pruned_count += old_count - len(self.paths)

        except Exception:
            # Fall back to metric-only on any error
            return
    
    def reset_statistics(self):
        """Reset AI statistics counters."""
        self.ai_pruned_count = 0
        self.ai_calls = 0
    
    def get_statistics(self) -> dict:
        """Return AI pruning statistics."""
        return {
            'ai_calls': self.ai_calls,
            'ai_pruned_count': self.ai_pruned_count,
            'avg_pruned_per_call': (self.ai_pruned_count / self.ai_calls 
                                   if self.ai_calls > 0 else 0),
        }
