import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from python_polar_coding.polar_codes.base.decoding_path import DecodingPathMixin
from python_polar_coding.polar_codes.sc.decoder import SCDecoder


class PathPruningNet(nn.Module):
    """MLP for path pruning in AISCL."""
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.fc1 = nn.Linear(2 * N, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x

    def score(self, llr, bits):
        """Score a single path."""
        llr = np.asarray(llr, dtype=np.float32)
        bits = np.asarray(bits, dtype=np.float32)
        N = len(llr)
        bits_pad = np.pad(bits, (0, N - len(bits)), 'constant')
        X = np.concatenate([llr, bits_pad])[None, :]
        X = torch.tensor(X)
        with torch.no_grad():
            score = self.forward(X).squeeze().item()
        return score

    def score_batch(self, X):
        """Score a batch of input vectors."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            scores = self.forward(X).squeeze(-1)
        return scores


class AIPath(DecodingPathMixin, SCDecoder):
    """Decoding path of AI-based SCL decoder."""
    def __init__(self, ai_model=None, **kwargs):
        super().__init__(**kwargs)
        self.ai_model = ai_model

    def score_ai(self):
        """Score path using AI model."""
        if self.ai_model is not None:
            try:
                llr_vec = self.intermediate_llr[0]
                bits_vec = self.intermediate_bits[-1]
                if llr_vec is not None and bits_vec is not None:
                    return self.ai_model.score(llr_vec, bits_vec)
            except Exception:
                pass
        return self._path_metric
