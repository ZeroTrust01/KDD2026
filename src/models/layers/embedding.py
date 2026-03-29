"""
Feature Embedding Layer
Reference: FuxiCTR fuxictr/pytorch/layers/embeddings/feature_embedding.py
Handles categorical (Embedding), numeric (Linear), and sequence (Embedding + pooling) features.
"""
import torch
from torch import nn


class FeatureEmbedding(nn.Module):
    """
    Unified feature embedding layer.
    
    feature_config format:
    {
        "feature_name": {
            "type": "categorical" | "numeric" | "sequence",
            "vocab_size": int,          # for categorical/sequence
            "embedding_dim": int,
            "max_len": int,             # for sequence
        },
        ...
    }
    """

    def __init__(self, feature_config, default_emb_dim=16):
        super().__init__()
        self.feature_config = feature_config
        self.embeddings = nn.ModuleDict()
        self.total_dim = 0

        for feat_name, spec in feature_config.items():
            ftype = spec["type"]
            emb_dim = spec.get("embedding_dim", default_emb_dim)

            if ftype == "categorical":
                vocab_size = spec["vocab_size"]
                self.embeddings[feat_name] = nn.Embedding(
                    vocab_size, emb_dim, padding_idx=0
                )
                self.total_dim += emb_dim

            elif ftype == "numeric":
                self.embeddings[feat_name] = nn.Linear(1, emb_dim, bias=False)
                self.total_dim += emb_dim

            elif ftype == "sequence":
                vocab_size = spec["vocab_size"]
                self.embeddings[feat_name] = nn.Embedding(
                    vocab_size, emb_dim, padding_idx=0
                )
                # sequence features contribute emb_dim after pooling
                self.total_dim += emb_dim

        self._init_weights()

    def _init_weights(self):
        for name, module in self.embeddings.items():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=1e-4)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1e-4)

    def forward(self, inputs):
        """
        Args:
            inputs: dict of {feature_name: tensor}
                - categorical: [B] LongTensor
                - numeric: [B] FloatTensor
                - sequence: [B, L] LongTensor
        Returns:
            emb_concat: [B, total_dim] - all embeddings concatenated (sequences mean-pooled)
            emb_dict: dict of {feature_name: embedding tensor}
        """
        emb_dict = {}
        for feat_name, spec in self.feature_config.items():
            if feat_name not in inputs:
                continue
            ftype = spec["type"]
            x = inputs[feat_name]

            if ftype == "categorical":
                emb_dict[feat_name] = self.embeddings[feat_name](x.long())  # [B, D]

            elif ftype == "numeric":
                emb_dict[feat_name] = self.embeddings[feat_name](
                    x.float().unsqueeze(-1)
                )  # [B, D]

            elif ftype == "sequence":
                seq_emb = self.embeddings[feat_name](x.long())  # [B, L, D]
                # Store full sequence embedding for DIN attention
                emb_dict[feat_name] = seq_emb  # [B, L, D]

        return emb_dict

    def get_flat_embedding(self, emb_dict, sequence_pooling="mean"):
        """Flatten all embeddings into a single vector per sample."""
        emb_list = []
        for feat_name, spec in self.feature_config.items():
            if feat_name not in emb_dict:
                continue
            emb = emb_dict[feat_name]
            if spec["type"] == "sequence":
                if sequence_pooling == "mean":
                    # Mean pooling with mask (non-zero positions)
                    mask = (emb.abs().sum(dim=-1) > 0).float()  # [B, L]
                    mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
                    emb = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask_sum  # [B, D]
                elif sequence_pooling == "sum":
                    emb = emb.sum(dim=1)
            emb_list.append(emb)
        return torch.cat(emb_list, dim=-1)  # [B, total_dim]
