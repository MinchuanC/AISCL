"""Codec wrapper for AI-guided FastSSC v2."""
from typing import Union
from python_polar_coding.polar_codes.base.codec import BasePolarCodec
from python_polar_coding.polar_codes.ai_scl.ai_fast_ssc_decoder_v2 import AIGuidedFastSSCDecoderV2


class AIGuidedFastSSCCodecV2(BasePolarCodec):
    """AI-guided FastSSC codec v2 with intelligent SCL fallback."""
    
    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasePolarCodec.BHATTACHARYYA,
                 L: int = 4,
                 ai_model=None,
                 scl_threshold: float = 0.5):
        """
        Parameters
        ----------
        scl_threshold : float
            AI confidence threshold for switching to SCL. Default 0.5.
        """
        self.L = L
        self.ai_model = ai_model
        self.scl_threshold = scl_threshold
        
        # Call parent __init__ to set up codec
        super().__init__(N=N, K=K,
                         design_snr=design_snr,
                         is_systematic=is_systematic,
                         mask=mask,
                         pcc_method=pcc_method)
    
    def init_decoder(self):
        """Instantiate decoder with AI guiding."""
        return AIGuidedFastSSCDecoderV2(
            n=self.n,
            mask=self.mask,
            is_systematic=self.is_systematic,
            L=self.L,
            ai_model=self.ai_model,
            scl_threshold=self.scl_threshold
        )
