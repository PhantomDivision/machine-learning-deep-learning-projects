from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset("stanfordnlp/imdb")

# Shuffle training data
train_data = dataset["train"].shuffle(seed=42)

# Select small training subset
train_data = train_data.select(range(2000))

# Split into training and validation
split = train_data.train_test_split(
    test_size=0.2,
    seed=42
)

train_data = split["train"]
val_data = split["test"]

# Keep a small test subset for our first experiment
test_data = dataset["test"].shuffle(seed=42).select(range(500))

print("Training samples:", len(train_data))
print("Validation samples:", len(val_data))
print("Testing samples:", len(test_data))

print("\nTraining example:")
print(train_data[0])