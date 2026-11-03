from transformers import AutoTokenizer

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "I love this product!"

# Convert text into tokens
tokens = tokenizer.tokenize(text)

print("Original text:", text)
print("Tokens:", tokens)

# Convert tokens into numerical IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Token IDs:", token_ids)

# Complete model input
encoded = tokenizer(text, return_tensors="pt")

print("Encoded input:", encoded)