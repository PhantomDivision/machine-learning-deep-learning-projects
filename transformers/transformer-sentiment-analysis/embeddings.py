import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "I love this product!"

encoded = tokenizer(text, return_tensors="pt")

# Get embeddings before Transformer encoder layers
with torch.no_grad():
    embeddings = model.distilbert.embeddings(
        input_ids=encoded["input_ids"]
    )

print("Input IDs shape:", encoded["input_ids"].shape)
print("Embedding shape:", embeddings.shape)

# Display embedding for the word 'love'
love_embedding = embeddings[0, 2]

print("\nEmbedding of 'love':")
print(love_embedding)

print("\nEmbedding dimensions:", love_embedding.shape)