import torch
import torch.nn as nn

from config import *
from transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            N_EMBD
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD
        )

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(N_EMBD, N_HEAD, DROPOUT)
                for _ in range(N_LAYER)
            ]
        )

        # Final normalization
        self.ln_f = nn.LayerNorm(N_EMBD)

        # Predict next token
        self.lm_head = nn.Linear(
            N_EMBD,
            vocab_size
        )

    def forward(self, idx):

        B, T = idx.shape

        # Token positions
        positions = torch.arange(T, device=idx.device)

        # Embeddings
        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb

        # Transformer blocks
        x = self.blocks(x)

        # Final layer normalization
        x = self.ln_f(x)

        # Vocabulary scores
        logits = self.lm_head(x)

        return logits