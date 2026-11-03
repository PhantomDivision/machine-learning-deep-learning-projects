import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, block_size):

        tokens = tokenizer.encode(text)

        self.data = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):

        x = self.data[idx:idx + self.block_size]

        y = self.data[idx + 1:idx + self.block_size + 1]

        return x, y