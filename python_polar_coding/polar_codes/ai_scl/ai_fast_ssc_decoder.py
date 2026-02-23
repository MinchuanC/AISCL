"""
AI-guided FastSSC decoder: uses fast SSC as the baseline decoder.
For now, simply wraps FastSSC for speed evaluation.
"""
import numpy as np
import torch
from python_polar_coding.polar_codes.base.decoder import BaseDecoder
from python_polar_coding.polar_codes.fast_ssc.decoder import FastSSCDecoder


class AIGuidedFastSSCDecoder(BaseDecoder):
    """
    Wraps FastSSC decoder for speed evaluation.
    AI-guided branching logic can be added later if needed.
    """
    
    def __init__(self, n: int, mask: np.array, is_systematic: bool = True,
                 L: int = 1, ai_model=None, branch_threshold: float = 0.3):
        """
        Parameters
        ----------
        n : int
            log2(N)
        mask : np.array
            Polar code mask
        is_systematic : bool
            Not used with FastSSC, kept for API compatibility
        L : int
            List size (not used with FastSSC, kept for API compatibility)
        ai_model : object
            AI model (not used yet, kept for API compatibility)
        branch_threshold : float
            Branching threshold (not used yet, kept for API compatibility)
        """
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self.L = L
        self.ai_model = ai_model
        self.branch_threshold = branch_threshold
        
        # Use FastSSC as base decoder (BaseTreeDecoder doesn't take is_systematic)
        self.base_decoder = FastSSCDecoder(n=n, mask=mask)
    
    def decode_internal(self, received_llr: np.array) -> np.array:
        """Decode using FastSSC."""
        return self.base_decoder.decode(received_llr)
