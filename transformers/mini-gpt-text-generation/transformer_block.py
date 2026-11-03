import torch.nn as nn
from attention import MultiHeadAttention


class FeedForward(nn.Module):

    def __init__(self, n_embd):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(n_embd, 4 * n_embd),

            nn.ReLU(),

            nn.Linear(4 * n_embd, n_embd)

        )

    def forward(self, x):

        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(self, n_embd, n_head, dropout):

        super().__init__()

        self.attention = MultiHeadAttention(n_embd, n_head)

        self.feed_forward = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)

        self.ln2 = nn.LayerNorm(n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.dropout(self.attention(self.ln1(x)))

        # Feed-forward + residual connection
        x = x + self.dropout(self.feed_forward(self.ln2(x)))

        return x