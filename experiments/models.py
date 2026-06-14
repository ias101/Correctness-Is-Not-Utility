"""
MLP Failure Predictor for Hazard-Based Adaptive Early Stopping.

Architecture (from FINAL_PROPOSAL.md v3):
  - Input: hidden_state (4096) + stage_embedding (16)
  - Hidden: [256, 128] with ReLU + Dropout(0.1)
  - Output: P(failure | h_t, stage) ∈ [0, 1]

Ablation variants (Block 3):
  - Stage-specific: separate MLP per stage (no shared embedding)
  - +LSTM: recurrent layer over stage sequence
  - No stage embedding: hidden_state only
  - +Retrieval features: hidden_state + stage_emb + bm25_score + passage_count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FailurePredictor(nn.Module):
    """
    Single MLP with learned stage embedding.
    Maps (hidden_state, stage_index) → P(failure).

    This is the primary model (v3) from FINAL_PROPOSAL.md.
    ~10M parameters with LLaMA-8B hidden dim (4096):
      4096 → 256: 4096*256 + 256 = 1,049,088
      256 → 128:  256*128 + 128 = 32,896
      128 → 1:    128*1 + 1 = 129
      Stage emb (4 × 16): 64
      Total: ~1.08M (well under 10M target; but with bias terms closer to ~1.1M)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        stage_emb_dim: int = 16,
        num_stages: int = 4,
        mlp_hidden: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stage_emb_dim = stage_emb_dim
        self.num_stages = num_stages

        # Learned stage embedding
        self.stage_embedding = nn.Embedding(num_stages, stage_emb_dim)

        # Build MLP layers
        input_dim = hidden_dim + stage_emb_dim
        layers = []
        for h in mlp_hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))  # scalar logit

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, hidden_states: torch.Tensor, stage_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, hidden_dim) — LLM hidden states at a given stage
            stage_indices: (B,) — integer stage index [0, num_stages)
        Returns:
            logits: (B, 1) — raw logits (use sigmoid for probability)
        """
        stage_emb = self.stage_embedding(stage_indices)  # (B, stage_emb_dim)
        combined = torch.cat([hidden_states, stage_emb], dim=-1)  # (B, hidden_dim + stage_emb_dim)
        return self.mlp(combined)

    def predict_proba(
        self, hidden_states: torch.Tensor, stage_indices: torch.Tensor
    ) -> torch.Tensor:
        """Return P(correct | h_t, stage) ∈ [0, 1] — probability the final answer WILL be correct.

        NOTE: The model is trained with label = final_correctness (1=correct, 0=wrong).
        sigmoid(logits) therefore gives P(correct), NOT P(failure).
        """
        logits = self.forward(hidden_states, stage_indices)
        return torch.sigmoid(logits).squeeze(-1)

    def predict_failure_proba(
        self, hidden_states: torch.Tensor, stage_indices: torch.Tensor
    ) -> torch.Tensor:
        """Return P(failure | h_t, stage) = 1 - P(correct | h_t, stage)."""
        return 1.0 - self.predict_proba(hidden_states, stage_indices)

    def should_stop(
        self,
        hidden_states: torch.Tensor,
        stage_indices: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """Return bool tensor: True = stop execution at this stage.

        Stop when P(correct | h_t) >= tau — i.e., model is confident the answer is correct,
        so continuing would waste compute.
        """
        return self.predict_proba(hidden_states, stage_indices) >= tau


# ── Ablation Variants (Block 3) ────────────────────────────────────


class StageSpecificPredictor(nn.Module):
    """
    Block 3.1: Separate MLP per stage, no shared parameters.
    Tests whether stage-specific specialization helps.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_stages: int = 4,
        mlp_hidden: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stages = num_stages

        self.per_stage_mlps = nn.ModuleList()
        for _ in range(num_stages):
            layers = []
            input_dim = hidden_dim
            for h in mlp_hidden:
                layers.append(nn.Linear(input_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = h
            layers.append(nn.Linear(input_dim, 1))
            self.per_stage_mlps.append(nn.Sequential(*layers))

    def forward(
        self, hidden_states: torch.Tensor, stage_indices: torch.Tensor
    ) -> torch.Tensor:
        """Route each sample to its stage-specific MLP."""
        outputs = torch.zeros(
            hidden_states.size(0), 1, device=hidden_states.device
        )
        for s in range(self.num_stages):
            mask = (stage_indices == s)
            if mask.any():
                outputs[mask] = self.per_stage_mlps[s](hidden_states[mask])
        return outputs

    def predict_proba(
        self, hidden_states: torch.Tensor, stage_indices: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden_states, stage_indices)).squeeze(-1)


class RecurrentPredictor(nn.Module):
    """
    Block 3.2: LSTM over stage sequence before final prediction.
    Tests whether modeling inter-stage dependencies helps.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        stage_emb_dim: int = 16,
        num_stages: int = 4,
        lstm_hidden: int = 128,
        mlp_hidden: List[int] = [128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.hidden_dim = hidden_dim

        self.stage_embedding = nn.Embedding(num_stages, stage_emb_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim + stage_emb_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        layers = []
        input_dim = lstm_hidden
        for h in mlp_hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.output_mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        stage_indices: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, hidden_dim)
            stage_indices: (B,)
            sequence_mask: optional, for padded sequences
        Returns:
            logits: (B, 1)
        """
        stage_emb = self.stage_embedding(stage_indices)
        combined = torch.cat([hidden_states, stage_emb], dim=-1)
        # Reshape to (B, 1, input_dim) for single-timestep LSTM
        combined = combined.unsqueeze(1)
        lstm_out, _ = self.lstm(combined)
        lstm_out = lstm_out.squeeze(1)  # (B, lstm_hidden)
        return self.output_mlp(lstm_out)

    def predict_proba(
        self, hidden_states: torch.Tensor, stage_indices: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden_states, stage_indices)).squeeze(-1)


class NoStageEmbeddingPredictor(nn.Module):
    """
    Block 3.3: MLP without stage embedding.
    Tests whether stage information is necessary.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        mlp_hidden: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        input_dim = hidden_dim
        for h in mlp_hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)

    def predict_proba(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(hidden_states)).squeeze(-1)


class RetrievalFeaturePredictor(nn.Module):
    """
    Block 3.4: MLP with additional retrieval-specific features.
    Input: hidden_state + stage_emb + bm25_score + passage_count + avg_passage_length.
    Tests whether retrieval-quality signals improve prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        stage_emb_dim: int = 16,
        num_stages: int = 4,
        num_retrieval_features: int = 3,  # bm25, passage_count, avg_passage_len
        mlp_hidden: List[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, stage_emb_dim)

        input_dim = hidden_dim + stage_emb_dim + num_retrieval_features
        layers = []
        for h in mlp_hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        stage_indices: torch.Tensor,
        retrieval_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, hidden_dim)
            stage_indices: (B,)
            retrieval_features: (B, num_retrieval_features)
        """
        stage_emb = self.stage_embedding(stage_indices)
        combined = torch.cat([hidden_states, stage_emb, retrieval_features], dim=-1)
        return self.mlp(combined)

    def predict_proba(
        self,
        hidden_states: torch.Tensor,
        stage_indices: torch.Tensor,
        retrieval_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(hidden_states, stage_indices, retrieval_features)
        ).squeeze(-1)


# ── Factory ─────────────────────────────────────────────────────────


def build_predictor(
    variant: str = "mlp",
    hidden_dim: int = 4096,
    stage_emb_dim: int = 16,
    num_stages: int = 4,
    mlp_hidden: List[int] = [256, 128],
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    Build a failure predictor by variant name.

    Variants:
        - "mlp": default single MLP + stage embedding (v3)
        - "stage_specific": separate MLP per stage (Block 3.1)
        - "lstm": LSTM over stage sequence (Block 3.2)
        - "no_stage_emb": MLP without stage embedding (Block 3.3)
        - "retrieval_features": MLP + retrieval features (Block 3.4)
    """
    variant = variant.lower()
    if variant == "mlp":
        return FailurePredictor(
            hidden_dim=hidden_dim,
            stage_emb_dim=stage_emb_dim,
            num_stages=num_stages,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
    elif variant == "stage_specific":
        return StageSpecificPredictor(
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
    elif variant == "lstm":
        return RecurrentPredictor(
            hidden_dim=hidden_dim,
            stage_emb_dim=stage_emb_dim,
            num_stages=num_stages,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
    elif variant == "no_stage_emb":
        return NoStageEmbeddingPredictor(
            hidden_dim=hidden_dim,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
    elif variant == "retrieval_features":
        return RetrievalFeaturePredictor(
            hidden_dim=hidden_dim,
            stage_emb_dim=stage_emb_dim,
            num_stages=num_stages,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown predictor variant: {variant}")
