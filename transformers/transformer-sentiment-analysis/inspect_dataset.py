from datasets import load_dataset

# Download IMDb dataset
dataset = load_dataset("stanfordnlp/imdb")

print("\nDataset information:")
print(dataset)

print("\nTraining example:")
print(dataset["train"][0])

print("\nTest example:")
print(dataset["test"][0])

print("\nLabels:")
print(dataset["train"].features["label"])