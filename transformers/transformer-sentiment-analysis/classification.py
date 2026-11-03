import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

text = "I absolutely love this product!"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Raw classification scores
logits = outputs.logits

# Convert scores into probabilities
probabilities = torch.softmax(logits, dim=1)

# Find predicted class
predicted_class = torch.argmax(probabilities, dim=1).item()

print("Input:", text)
print("\nLogits:", logits)
print("\nProbabilities:", probabilities)

print("\nPredicted sentiment:",
      model.config.id2label[predicted_class])

print("Confidence:",
      probabilities[0][predicted_class].item() * 100, "%")