"""
Model
======
GPT-style decoder-only Transformer for beat generation.

Architecture:
  - Token embedding + learned positional embedding
  - N Transformer decoder layers (causal self-attention)
  - Linear projection to vocabulary

Designed to be small enough to train on a single GPU with ~200 beats,
but scalable for larger datasets.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply padding mask
        if attention_mask is not None:
            # attention_mask: (B, T), 1 = attend, 0 = ignore
            pad_mask = (1 - attention_mask).bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Transformer decoder block: attention → FFN with residuals + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BeatTransformer(nn.Module):
    """
    GPT-style model for beat generation.

    Takes token sequences → predicts next token at each position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding weights with output head
        self.head.weight = self.token_emb.weight

        # Init weights
        self.apply(self._init_weights)
        logger.info(
            f"BeatTransformer: {vocab_size} vocab, {d_model}d, "
            f"{n_heads}h, {n_layers}L, {self._count_params():.1f}M params"
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _count_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) 1=attend, 0=pad
            targets: (B, T) target token indices for loss computation

        Returns:
            {"logits": (B, T, V), "loss": scalar if targets provided}
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        # Position indices
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        # Embeddings
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        result = {"logits": logits}

        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_token_id,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 0.85,
        top_k: int = 50,
        top_p: float = 0.92,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-k/top-p sampling.

        Args:
            prompt: (1, T) initial token sequence
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop when this token is generated

        Returns:
            (1, T+generated) full sequence
        """
        self.eval()
        device = prompt.device
        generated = prompt.clone()

        for _ in range(max_length):
            # Crop to max_seq_len if needed
            input_ids = generated[:, -self.max_seq_len :]

            output = self.forward(input_ids)
            logits = output["logits"][:, -1, :]  # Last position

            # Temperature
            logits = logits / temperature

            # Top-k
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated
