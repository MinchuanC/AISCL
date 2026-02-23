"""Codec for AI-guided Fast SCL decoding."""

from typing import Union

from python_polar_coding.polar_codes.base.codec import BasePolarCodec
from .decoder import AIFastSCLDecoder
from .nn import PathPruningNN


class AIFastSCLPolarCodec(BasePolarCodec):
    """
    Polar code with AI-guided Fast SCL decoding.
    
    Extends standard SCL with neural network-based path pruning
    to reduce computational complexity while maintaining BER performance.
    """
    
    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasePolarCodec.BHATTACHARYYA,
                 L: int = 4,
                 ai_model: PathPruningNN = None,
                 ai_threshold: float = 0.05,
                 enable_ai_pruning: bool = True):
        """
        Initialize AI-guided Fast SCL codec.
        
        Parameters
        ----------
        N : int
            Code length (power of 2)
        K : int
            Information bit length
        design_snr : float
            Design SNR in dB
        is_systematic : bool
            Whether to use systematic encoding
        mask : str or None
            Polar code mask (None for auto-generation)
        pcc_method : str
            PCC method for code construction
        L : int
            List size for SCL
        ai_model : PathPruningNN or None
            Pre-trained neural network for path pruning
        ai_threshold : float
            AI pruning threshold (paths with prob < threshold discarded)
        enable_ai_pruning : bool
            Whether to enable AI pruning
        """
        self.L = L
        self.ai_model = ai_model
        self.ai_threshold = ai_threshold
        self.enable_ai_pruning = enable_ai_pruning
        # Quick-pruning tunables passed to decoder
        self.quick_percentile = 50.0
        self.topk_multiplier = 2.0
        
        super().__init__(N=N, K=K,
                         design_snr=design_snr,
                         is_systematic=is_systematic,
                         mask=mask,
                         pcc_method=pcc_method)
    
    def init_decoder(self):
        """Create AI-guided SCL decoder instance."""
        return AIFastSCLDecoder(
            n=self.n,
            mask=self.mask,
            is_systematic=self.is_systematic,
            L=self.L,
            ai_model=self.ai_model,
            ai_threshold=self.ai_threshold,
            enable_ai_pruning=self.enable_ai_pruning
            , quick_percentile=self.quick_percentile
            , topk_multiplier=self.topk_multiplier
        )
    
    def to_dict(self) -> dict:
        """Serialize codec configuration."""
        config = super().to_dict()
        config.update({
            'L': self.L,
            'ai_threshold': self.ai_threshold,
            'enable_ai_pruning': self.enable_ai_pruning,
            'quick_percentile': self.quick_percentile,
            'topk_multiplier': self.topk_multiplier,
            'ai_model': 'PathPruningNN' if self.ai_model is not None else None,
        })
        return config
