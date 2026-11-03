from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Load dataset
dataset = load_dataset("stanfordnlp/imdb")

# Prepare subsets
train_data = dataset["train"].shuffle(seed=42).select(range(2000))

split = train_data.train_test_split(
    test_size=0.2,
    seed=42
)

train_data = split["train"]
val_data = split["test"]

test_data = dataset["test"].shuffle(seed=42).select(range(500))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Tokenize datasets
train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Remove original text
train_data = train_data.remove_columns(["text"])
val_data = val_data.remove_columns(["text"])
test_data = test_data.remove_columns(["text"])

# Convert to PyTorch format
train_data.set_format("torch")
val_data.set_format("torch")
test_data.set_format("torch")

print("\nTokenization completed!")

print("Training dataset:", train_data)
print("Validation dataset:", val_data)
print("Testing dataset:", test_data)

print("\nFirst training example:")
print(train_data[0])