"""Random-initialized byte-level causal Transformer.

This is an educational language-model baseline, not a production assistant.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.max_seq_len, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding width must be divisible by attention heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        z = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        z = z.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(z))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd * 4),
            nn.GELU(),
            nn.Linear(cfg.n_embd * 4, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ByteTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must have shape (batch, time), dtype torch.long")
        batch, length = tokens.shape
        if not 1 <= length <= self.cfg.max_seq_len:
            raise ValueError("Sequence length outside configured context")
        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) + self.position(positions)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("Targets must have the same shape as tokens")
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        if new_tokens < 0 or temperature <= 0:
            raise ValueError("new_tokens must be non-negative and temperature positive")
        output = prefix
        for _ in range(new_tokens):
            logits, _ = self(output[:, -self.cfg.max_seq_len:])
            next_token = torch.multinomial(F.softmax(logits[:, -1, :] / temperature, dim=-1), 1)
            output = torch.cat((output, next_token), dim=1)
        return output
