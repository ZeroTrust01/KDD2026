"""
DIN Attention - Target Attention for behavior sequence modeling
Reference: FuxiCTR fuxictr/pytorch/layers/attentions/target_attention.py
Paper: DIN (Alibaba, KDD 2018)
"""
import torch
from torch import nn
from .mlp import MLP


class DINAttention(nn.Module):
    """
    DIN-style target attention.
    Computes attention weight for each history item w.r.t. the target item.
    Input: concat(target, history, target-history, target*history) -> MLP -> weight
    """

    def __init__(self,
                 embedding_dim,
                 attention_units=(64, 16),
                 activation="Dice",
                 use_softmax=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax
        self.attention_mlp = MLP(
            input_dim=4 * embedding_dim,
            hidden_units=attention_units,
            activations=activation,
            output_dim=1,
        )

    def forward(self, target_emb, sequence_emb, mask=None):
        """
        Args:
            target_emb:   [B, D] - target item embedding
            sequence_emb: [B, L, D] - behavior sequence embedding
            mask:         [B, L] - 1 for valid, 0 for padding
        Returns:
            output: [B, D] - attention-weighted pooling of sequence
        """
        seq_len = sequence_emb.size(1)
        target_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]

        # Concat 4 interaction features
        att_input = torch.cat([
            target_emb,
            sequence_emb,
            target_emb - sequence_emb,
            target_emb * sequence_emb,
        ], dim=-1)  # [B, L, 4D]

        # Compute attention scores
        B, L, D4 = att_input.shape
        att_score = self.attention_mlp(att_input.view(-1, D4))  # [B*L, 1]
        att_score = att_score.view(B, L)  # [B, L]

        if mask is not None:
            att_score = att_score * mask.float()

        if self.use_softmax:
            if mask is not None:
                att_score = att_score + (-1e9) * (1 - mask.float())
            att_score = att_score.softmax(dim=-1)

        # Weighted sum
        output = (att_score.unsqueeze(-1) * sequence_emb).sum(dim=1)  # [B, D]
        return output
