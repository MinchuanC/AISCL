"""
Optimized AI-augmented SCL codec using the optimized decoder.
"""
import numpy as np
from typing import Union
from ..base.codec import BasePolarCodec
from .optimized_decoder import OptimizedAISCLDecoder


class OptimizedAISCLPolarCodec(BasePolarCodec):
    """Optimized AISCL codec using batch operations and lightweight paths."""
    decoder_class = OptimizedAISCLDecoder
    
    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasePolarCodec.BHATTACHARYYA,
                 L: int = 1, 
                 ai_model=None):
        """
        Parameters
        ----------
        N : int
            Block length
        K : int
            Information length
        design_snr : float
            Design SNR (unused in AISCL)
        is_systematic : bool
            Systematic encoding
        mask : str or None
            Frozen bit mask
        pcc_method : str
            PCC method
        L : int
            List size
        ai_model : nn.Module
            AI path pruning model (torch)
        """
        self.L = L
        self.ai_model = ai_model
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method)
    
    def init_decoder(self):
        """Initialize optimized AISCL decoder."""
        return self.decoder_class(
            n=self.n,
            mask=self.mask,
            is_systematic=self.is_systematic,
            L=self.L,
            ai_model=self.ai_model
        )
    
    def to_dict(self):
        """Serialize codec state."""
        d = super().to_dict()
        d.update({'L': self.L})
        return d
