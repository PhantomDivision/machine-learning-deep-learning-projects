from transformers import pipeline

def main():

    print("TRANSFORMER SENTIMENT ANALYSIS")

    print("\nLoading DistilBERT model...")

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    print("Model loaded successfully!")

    reviews = [
        "I absolutely love this product!",
        "The service was very disappointing.",
        "Amazing experience. Highly recommended!",
        "Worst purchase I have ever made.",
        "The product is good but delivery was slow."
    ]

    results = classifier(reviews)

    for review, result in zip(reviews, results):

        print("\nReview:", review)
        print("Sentiment:", result["label"])
        print("Confidence:", round(result["score"] * 100, 2), "%")

if __name__ == "__main__":
    main()