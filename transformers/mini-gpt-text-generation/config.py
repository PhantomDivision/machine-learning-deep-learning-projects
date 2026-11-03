import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPT model configuration
VOCAB_SIZE = 5000  # Actual vocabulary size comes from tokenizer
BLOCK_SIZE = 16

N_EMBD = 128
N_HEAD = 4
N_LAYER = 2

DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 10