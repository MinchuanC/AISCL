"""
Early termination AISCL codec using aggressive path pruning.
"""
from typing import Union
from ..base.codec import BasePolarCodec
from .early_termination_decoder import EarlyTerminationAISCLDecoder


class EarlyTerminationAISCLCodec(BasePolarCodec):
    """Polar code with early termination AI-based SCL decoding."""
    decoder_class = EarlyTerminationAISCLDecoder

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasePolarCodec.BHATTACHARYYA,
                 L: int = 1,
                 ai_model=None,
                 dominance_threshold: float = 2.0):
        self.L = L
        self.ai_model = ai_model
        self.dominance_threshold = dominance_threshold
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
                                  dominance_threshold=self.dominance_threshold)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d
