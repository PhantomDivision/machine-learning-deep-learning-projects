from transformers import pipeline

MODEL_PATH = "./fine_tuned_model"

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=-1
)

reviews = [
    "This movie was absolutely fantastic. I loved every moment!",
    "The movie was boring, predictable, and a complete waste of time.",
    "The acting was good, but the story was disappointing."
]

for review in reviews:
    result = classifier(review)[0]

    print("\nReview:", review)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"] * 100, 2), "%")