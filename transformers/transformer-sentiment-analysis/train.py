import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 1. Load dataset
dataset = load_dataset("stanfordnlp/imdb")

train_data = dataset["train"].shuffle(seed=42).select(range(2000))

split = train_data.train_test_split(
    test_size=0.2,
    seed=42
)

train_data = split["train"]
val_data = split["test"]

test_data = dataset["test"].shuffle(seed=42).select(range(500))

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Remove text column
train_data = train_data.remove_columns(["text"])
val_data = val_data.remove_columns(["text"])
test_data = test_data.remove_columns(["text"])

# 4. Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# 5. Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 7. Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=25,
    report_to="none",
    use_cpu=True
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 9. Train
print("\nStarting fine-tuning...\n")

trainer.train()

# 10. Evaluate on untouched test subset
print("\nFinal test evaluation:")

test_results = trainer.evaluate(
    eval_dataset=test_data,
    metric_key_prefix="test"
)

print(test_results)

# 11. Save model and tokenizer
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("\nModel saved successfully!")