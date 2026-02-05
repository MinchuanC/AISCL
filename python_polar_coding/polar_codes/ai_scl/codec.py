from typing import Union
from ..base.codec import BasePolarCodec
from .decoder import AISCLDecoder

class AISCLPolarCodec(BasePolarCodec):
    """Polar code with AI-based SCL decoding algorithm."""
    decoder_class = AISCLDecoder

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
        return self.decoder_class(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic, L=self.L, ai_model=self.ai_model)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d
