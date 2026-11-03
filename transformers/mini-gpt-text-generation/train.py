import torch
import torch.nn as nn
import os
import json

from torch.utils.data import DataLoader

from config import *
from tokenizer import SimpleTokenizer
from dataset import GPTDataset
from model import GPTLanguageModel


# 1. Load training text
with open("data/input.txt", "r") as f:
    text = f.read()


# 2. Create tokenizer
tokenizer = SimpleTokenizer(text)
vocab_size = len(tokenizer.word_to_id)

print("Vocabulary size:", vocab_size)


# 3. Create dataset
dataset = GPTDataset(text, tokenizer, BLOCK_SIZE)

# 4. Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# 5. Create GPT model
model = GPTLanguageModel(vocab_size).to(DEVICE)


# 6. Loss function
criterion = nn.CrossEntropyLoss()


# 7. Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# 8. Model information
print("Dataset samples:", len(dataset))
print("Model parameters:", sum(p.numel() for p in model.parameters()))
print("Device:", DEVICE)


# 9. Training loop
print("\nStarting GPT training...\n")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        B, T, C = logits.shape

        logits = logits.reshape(B * T, C)
        targets = y.reshape(B * T)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}")


# 10. Save trained model
os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/gpt_model.pth"
)


# 11. Save tokenizer vocabulary
with open("models/tokenizer.json", "w") as f:
    json.dump(tokenizer.word_to_id, f, indent=4)


print("\nGPT training completed!")
print("Model saved: models/gpt_model.pth")
print("Tokenizer saved: models/tokenizer.json")