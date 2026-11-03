import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with attention weights enabled
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager"
)

# Example sentence
text = "I absolutely love this product!"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Get predictions and attention weights
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Extract attention weights
attentions = outputs.attentions

print("Number of Transformer layers:", len(attentions))

# Inspect first layer
first_layer = attentions[0]

print("First layer attention shape:", first_layer.shape)

# Display tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print("\nTokens:")
print(tokens)

# Display attention from first head
first_head = first_layer[0, 0]

print("\nAttention weights from first head:")
print(first_head)

# Compare two attention heads
for head_number in [0, 1]:

    print(f"\nAttention Head {head_number}")

    head_attention = first_layer[0, head_number]

    for token, weights in zip(tokens, head_attention):
        print(f"\nToken: {token}")

        for target_token, weight in zip(tokens, weights):
            print(f"  {target_token:12s}: {weight.item():.4f}")


print("\nAttention across all layers:")

for layer_number, layer_attention in enumerate(attentions):

    print(f"\nLayer {layer_number + 1}")

    # Average attention across all 12 heads
    average_attention = layer_attention[0].mean(dim=0)

    # Show attention received by each token from the CLS query
    cls_attention = average_attention[0]

    for token, weight in zip(tokens, cls_attention):
        print(f"{token:12s}: {weight.item():.4f}")