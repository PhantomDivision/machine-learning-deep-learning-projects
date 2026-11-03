import torch
import json

from config import *
from model import GPTLanguageModel


# ==========================================
# 1. Load tokenizer vocabulary
# ==========================================

with open("models/tokenizer.json", "r") as f:
    word_to_id = json.load(f)

id_to_word = {
    int(idx): word
    for word, idx in word_to_id.items()
}


# ==========================================
# 2. Tokenizer functions
# ==========================================

def encode(text):

    words = text.split()

    return [
        word_to_id[word]
        for word in words
    ]


def decode(token_ids):

    return " ".join(
        id_to_word[token_id]
        for token_id in token_ids
    )


# ==========================================
# 3. Load trained GPT model
# ==========================================

vocab_size = len(word_to_id)

model = GPTLanguageModel(vocab_size).to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/gpt_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("GPT model loaded successfully!")


# ==========================================
# 4. Text generation function
# ==========================================

def generate(prompt, max_new_tokens=10):

    # Convert prompt into token IDs
    tokens = encode(prompt)

    idx = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    # Generate tokens one by one
    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the available context
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Predict next token
            logits = model(idx_cond)

            logits = logits[:, -1, :]

            # Select the most likely token
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

            # Append predicted token
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

    return decode(idx[0].tolist())


# ==========================================
# 5. Generate text
# ==========================================

prompt = "GPT"

output = generate(
    prompt,
    max_new_tokens=10
)

print("\nPrompt:", prompt)
print("Generated text:", output)