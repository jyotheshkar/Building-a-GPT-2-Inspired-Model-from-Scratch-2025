"""Small GPT-2-inspired decoder: pre-LN, causal attention, GELU and tied weights.

This is newly maintained educational code. Character tokenization and tiny
configuration differ from OpenAI GPT-2; no pretrained weights are included.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)), persistent=False)

    def forward(self, inputs):
        batch, length, width = inputs.shape
        q, k, v = self.qkv(inputs).view(batch, length, 3, self.n_head, self.head_size).unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        probabilities = self.attn_dropout(F.softmax(scores, dim=-1))
        attended = (probabilities @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.out_proj(attended))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attention = CausalAttention(config)
        self.ffwd = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd),
                                  nn.GELU(approximate="tanh"),
                                  nn.Linear(4 * config.n_embd, config.n_embd),
                                  nn.Dropout(config.dropout))

    def forward(self, inputs):
        inputs = inputs + self.attention(self.ln1(inputs))
        return inputs + self.ffwd(self.ln2(inputs))


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError("GPTLanguageModel requires a validated ModelConfig.")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ffwd.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids, targets=None):
        if token_ids.dtype != torch.long or token_ids.ndim != 2:
            raise ValueError("Model inputs must be a two-dimensional torch.long tensor.")
        batch, length = token_ids.shape
        if not 1 <= batch <= 128 or not 1 <= length <= self.config.block_size:
            raise ValueError("Input batch/context exceeds the configured model limits.")
        if bool(((token_ids < 0) | (token_ids >= self.config.vocab_size)).any()):
            raise ValueError("Input token IDs are outside the configured vocabulary.")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.embedding_dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
        logits = self.lm_head(self.final_norm(self.blocks(hidden)))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape or targets.dtype != torch.long:
                raise ValueError("Training targets must match the input tensor shape and type.")
            if bool(((targets < 0) | (targets >= self.config.vocab_size)).any()):
                raise ValueError("Target token IDs are outside the configured vocabulary.")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
