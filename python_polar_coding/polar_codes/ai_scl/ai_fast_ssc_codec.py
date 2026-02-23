"""
AI-guided FastSSC codec.
"""
from typing import Union
from ..base.codec import BasePolarCodec
from .ai_fast_ssc_decoder import AIGuidedFastSSCDecoder


class AIGuidedFastSSCCodec(BasePolarCodec):
    """Polar code with AI-guided FastSSC decoding."""
    decoder_class = AIGuidedFastSSCDecoder

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasePolarCodec.BHATTACHARYYA,
                 L: int = 1,
                 ai_model=None,
                 branch_threshold: float = 0.3):
        self.L = L
        self.ai_model = ai_model
        self.branch_threshold = branch_threshold
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
                                  branch_threshold=self.branch_threshold)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d
