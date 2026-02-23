"""
Selective AISCL codec - only use AI when paths are similar.
"""
from typing import Union
from ..base.codec import BasePolarCodec
from .selective_decoder import SelectiveAISCLDecoder


class SelectiveAISCLCodec(BasePolarCodec):
    """Polar code with selective AI scoring."""
    decoder_class = SelectiveAISCLDecoder

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasePolarCodec.BHATTACHARYYA,
                 L: int = 1,
                 ai_model=None,
                 uncertainty_threshold: float = 0.1):
        self.L = L
        self.ai_model = ai_model
        self.uncertainty_threshold = uncertainty_threshold
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method)

    def init_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic, 
                                  L=self.L, 
                                  ai_model=self.ai_model,
                                  uncertainty_threshold=self.uncertainty_threshold)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d
