import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, n_head):

        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Query, Key and Value projections
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Final output projection
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):

        B, T, C = x.shape

        # Project input into Q, K and V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal attention
        scores = Q @ K.transpose(-2, -1)

        scores = scores / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))

        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)

        out = weights @ V

        # Combine attention heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        out = self.proj(out)

        return out