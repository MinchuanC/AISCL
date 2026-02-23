"""
Vectorized AISCL codec that uses lightweight numpy-based path states.
"""
from typing import Union
from ..base.codec import BasePolarCodec
from .vectorized_decoder import VectorizedAISCLDecoder


class VectorizedAISCLCodec(BasePolarCodec):
    """Ultra-lightweight AISCL using numpy path states."""
    decoder_class = VectorizedAISCLDecoder
    
    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasePolarCodec.BHATTACHARYYA,
                 L: int = 1,
                 ai_model=None):
        self.L = L
        self.ai_model = ai_model
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method)
    
    def init_decoder(self):
        """Initialize vectorized AISCL decoder."""
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
