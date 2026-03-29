"""
DIN + DCN V2 Baseline Model
Combines:
  - DIN target attention for behavior sequence modeling
  - DCN V2 cross network for explicit feature interaction
  - Parallel MLP for implicit feature interaction

Architecture:
  Input Features
       │
  FeatureEmbedding
       │
  ┌────┴────┐
  │  DIN    │  (sequence features: target attention pooling)
  └────┬────┘
       │
  All Embeddings (flat)
       │
  ┌────┴────────────────┐
  │  CrossNetV2         │  MLP (parallel)
  └────┬────────────────┘    │
       │                     │
       └────── concat ───────┘
              │
           FC → sigmoid
"""
import torch
from torch import nn
from src.models.layers import FeatureEmbedding, DINAttention, CrossNetV2, MLP


class DIN_DCN(nn.Module):
    """DIN + DCN V2 baseline for CTR prediction."""

    def __init__(self, feature_config, model_config):
        super().__init__()
        self.feature_config = feature_config
        emb_dim = model_config.get("embedding_dim", 16)

        # --- Embedding layer ---
        self.embedding = FeatureEmbedding(feature_config, default_emb_dim=emb_dim)

        # --- DIN Attention for behavior sequences ---
        # target_field -> sequence_field mapping
        self.din_pairs = model_config.get("din_pairs", [])
        # e.g. [("ad_cate_id", "cate_seq"), ("ad_brand", "brand_seq")]
        self.attention_layers = nn.ModuleDict()
        for target_field, seq_field in self.din_pairs:
            self.attention_layers[seq_field] = DINAttention(
                embedding_dim=emb_dim,
                attention_units=model_config.get("attention_hidden_units", [64, 16]),
                activation=model_config.get("attention_activation", "Dice"),
                use_softmax=model_config.get("din_use_softmax", False),
            )

        # --- Compute total input dim ---
        input_dim = self.embedding.total_dim

        # --- CrossNet V2 ---
        num_cross_layers = model_config.get("num_cross_layers", 3)
        self.cross_net = CrossNetV2(input_dim, num_cross_layers)

        # --- Parallel DNN ---
        dnn_hidden = model_config.get("dnn_hidden_units", [256, 128])
        dnn_dropout = model_config.get("dnn_dropout", 0.0)
        self.parallel_dnn = MLP(
            input_dim=input_dim,
            hidden_units=dnn_hidden,
            activations="ReLU",
            dropout=dnn_dropout,
            batch_norm=model_config.get("batch_norm", False),
        )

        # --- Output layer ---
        final_dim = input_dim + dnn_hidden[-1]  # cross_out + dnn_out
        self.fc = nn.Linear(final_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, inputs):
        """
        Args:
            inputs: dict of {feature_name: tensor}
        Returns:
            dict with 'y_pred': [B, 1] predicted probabilities
        """
        # 1. Get all embeddings
        emb_dict = self.embedding(inputs)

        # 2. Apply DIN attention for sequence features
        for target_field, seq_field in self.din_pairs:
            if target_field in emb_dict and seq_field in emb_dict:
                target_emb = emb_dict[target_field]       # [B, D]
                sequence_emb = emb_dict[seq_field]         # [B, L, D]
                mask = inputs[seq_field].long() != 0       # [B, L]
                # Replace sequence embedding with attention-pooled result
                pooled = self.attention_layers[seq_field](
                    target_emb, sequence_emb, mask
                )  # [B, D]
                emb_dict[seq_field] = pooled  # now [B, D] instead of [B, L, D]

        # 3. Flatten all embeddings
        flat_emb = self._flatten_emb_dict(emb_dict)  # [B, total_dim]

        # 4. CrossNet + parallel DNN
        cross_out = self.cross_net(flat_emb)           # [B, total_dim]
        dnn_out = self.parallel_dnn(flat_emb)          # [B, dnn_hidden[-1]]
        combined = torch.cat([cross_out, dnn_out], dim=-1)

        # 5. Output
        y_pred = self.output_activation(self.fc(combined))  # [B, 1]
        return {"y_pred": y_pred}

    def _flatten_emb_dict(self, emb_dict):
        """Concatenate all embeddings in feature_config order."""
        emb_list = []
        for feat_name in self.feature_config:
            if feat_name not in emb_dict:
                continue
            emb = emb_dict[feat_name]
            if emb.dim() == 3:
                # Sequence not yet pooled (shouldn't happen after DIN), mean pool
                mask = (emb.abs().sum(-1) > 0).float()
                mask_sum = mask.sum(-1, keepdim=True).clamp(min=1)
                emb = (emb * mask.unsqueeze(-1)).sum(1) / mask_sum
            emb_list.append(emb)
        return torch.cat(emb_list, dim=-1)
