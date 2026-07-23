"""
=============================================================
Attention-Based ESD Risk Prediction Model  (Member 2)
=============================================================
Architecture:
  Input  →  Feature Encoder  →  Positional Encoding
         →  Multi-Head Temporal Attention (x2)
         →  Context Fusion (categorical embeddings)
         →  Feed-Forward Head  →  Risk Class (0/1/2)

Key design choices:
  - Multi-head self-attention captures temporal dependencies
    across the sensor window (e.g. rising E-field over 2s)
  - Categorical context (activity, environment, fabric) is
    embedded and fused after attention layers
  - Dropout + LayerNorm for regularisation
  - Output: 3-class softmax (Low / Medium / High)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Positional Encoding ───────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Attention Block ───────────────────────────────────────────
class AttentionBlock(nn.Module):
    """Multi-head self-attention + Feed-forward sublayer."""
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-forward with residual
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, attn_weights


# ── Main Model ────────────────────────────────────────────────
class ESDAttentionModel(nn.Module):
    """
    Explainable Attention-Based ESD Risk Predictor.

    Args:
        n_numerical  : number of continuous sensor features
        cat_vocab    : dict {feature_name: vocab_size}
        cat_emb_dim  : embedding dim for each categorical
        window_size  : number of time steps in input window
        d_model      : transformer hidden dim
        n_heads      : number of attention heads
        n_layers     : number of attention blocks
        ff_dim       : feed-forward inner dim
        n_classes    : output classes (3: Low/Medium/High)
        dropout      : dropout rate
    """
    def __init__(
        self,
        n_numerical : int   = 5,
        cat_vocab   : dict  = None,
        cat_emb_dim : int   = 8,
        window_size : int   = 50,
        d_model     : int   = 64,
        n_heads     : int   = 4,
        n_layers    : int   = 2,
        ff_dim      : int   = 128,
        n_classes   : int   = 3,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.window_size  = window_size
        self.d_model      = d_model

        # ── categorical embeddings ────────────────────────────
        if cat_vocab is None:
            cat_vocab = {}
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size + 1, cat_emb_dim)
            for name, vocab_size in cat_vocab.items()
        })
        total_cat_dim = len(cat_vocab) * cat_emb_dim

        # ── numerical input projection ─────────────────────────
        self.input_proj = nn.Linear(n_numerical, d_model)

        # ── positional encoding ───────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=window_size + 10, dropout=dropout)

        # ── attention blocks ──────────────────────────────────
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        # ── context fusion ────────────────────────────────────
        fusion_dim = d_model + total_cat_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # ── classification head ───────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def forward(self, numerical, categoricals=None, return_attention=False):
        """
        Args:
            numerical    : (B, T, n_numerical)  float tensor
            categoricals : dict {name: (B,) int tensor}  — per-window context
            return_attention : if True, return attention weight maps

        Returns:
            logits            : (B, n_classes)
            attention_weights : list of (B, T, T) tensors  [if return_attention]
        """
        B, T, _ = numerical.shape

        # Project numerical inputs to d_model
        x = self.input_proj(numerical)           # (B, T, d_model)
        x = self.pos_enc(x)

        # Run through attention blocks
        all_attn_weights = []
        for block in self.attention_blocks:
            x, attn_w = block(x)
            all_attn_weights.append(attn_w)      # (B, T, T)

        # Global average pooling over time
        x = x.mean(dim=1)                        # (B, d_model)

        # Embed and fuse categorical context
        if categoricals:
            cat_embs = [
                self.cat_embeddings[name](cat_tensor)
                for name, cat_tensor in categoricals.items()
            ]
            cat_concat = torch.cat(cat_embs, dim=-1)   # (B, total_cat_dim)
            x = torch.cat([x, cat_concat], dim=-1)     # (B, d_model + total_cat_dim)
            x = self.fusion_norm(x)

        logits = self.classifier(x)              # (B, n_classes)

        if return_attention:
            return logits, all_attn_weights
        return logits

    def predict_proba(self, numerical, categoricals=None):
        """Returns softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(numerical, categoricals)
            return F.softmax(logits, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model factory ─────────────────────────────────────────────
def build_model(window_size=50):
    cat_vocab = {
        "activity":    4,   # sitting/walking/running/handling
        "environment": 4,   # office/lab/outdoor/home
        "fabric_type": 4,   # cotton/polyester/wool/synthetic
    }
    model = ESDAttentionModel(
        n_numerical  = 5,
        cat_vocab    = cat_vocab,
        cat_emb_dim  = 8,
        window_size  = window_size,
        d_model      = 64,
        n_heads      = 4,
        n_layers     = 2,
        ff_dim       = 128,
        n_classes    = 3,
        dropout      = 0.1,
    )
    return model


if __name__ == "__main__":
    model = build_model(window_size=50)
    print(f"Model parameters: {model.count_parameters():,}")

    # Dummy forward pass
    B, T = 8, 50
    num  = torch.randn(B, T, 5)
    cats = {
        "activity":    torch.randint(0, 4, (B,)),
        "environment": torch.randint(0, 4, (B,)),
        "fabric_type": torch.randint(0, 4, (B,)),
    }
    logits, attn = model(num, cats, return_attention=True)
    print(f"Output shape   : {logits.shape}")
    print(f"Attention maps : {len(attn)} layers, each {attn[0].shape}")
