"""Train and evaluate a small spam classifier on labeled example emails.

Run from this lesson directory:
    python train_spam_classifier.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).resolve().parent / "data" / "emails.csv"
VALID_LABELS = {"spam", "ham"}
VALID_SPLITS = {"train", "test"}


def load_examples(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load and validate a fixed train/test split from a CSV file."""
    examples: dict[str, list[tuple[str, str]]] = {
        "train": [],
        "test": [],
    }

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["split", "label", "text"]:
            raise ValueError("CSV columns must be: split,label,text")

        for line_number, row in enumerate(reader, start=2):
            split = row["split"].strip()
            label = row["label"].strip()
            message = row["text"].strip()

            if split not in VALID_SPLITS or label not in VALID_LABELS or not message:
                raise ValueError(f"Invalid example at CSV line {line_number}")
            examples[split].append((message, label))

    if not examples["train"] or not examples["test"]:
        raise ValueError("Both train and test sets must contain examples")

    training_texts = {text for text, _ in examples["train"]}
    testing_texts = {text for text, _ in examples["test"]}
    if training_texts & testing_texts:
        raise ValueError("Training and test sets contain duplicate messages")

    return examples


def main() -> None:
    examples = load_examples(DATA_PATH)
    train_texts, train_labels = zip(*examples["train"])
    test_texts, test_labels = zip(*examples["test"])

    # The vectorizer learns a vocabulary from training text only.
    # The classifier learns label-related patterns from those features.
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(train_texts, train_labels)

    # The held-out test messages were not used during training.
    predictions = model.predict(test_texts)

    print(f"Training examples: {len(train_texts)}")
    print(f"Test examples: {len(test_texts)}")
    print(f"Test accuracy: {accuracy_score(test_labels, predictions):.2%}")

    # Rows = actual labels; columns = predicted labels.
    labels = ["ham", "spam"]
    matrix = confusion_matrix(test_labels, predictions, labels=labels)
    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print("              predicted ham  predicted spam")
    for label, counts in zip(labels, matrix):
        print(f"actual {label:<5} {counts[0]:>13} {counts[1]:>15}")

    print("\nHeld-out predictions:")
    for message, actual, predicted in zip(test_texts, test_labels, predictions):
        status = "correct" if actual == predicted else "incorrect"
        print(f"- [{status}] actual={actual}, predicted={predicted}: {message}")

    # These are extra demonstrations, not part of the accuracy calculation.
    new_messages = [
        "Claim your prize today",
        "Collect your reward today",
        "Please send the project notes",
        "A totally unfamiliar suspicious phrase",
    ]
    print("\nAdditional predictions (no ground-truth labels):")
    for message, predicted in zip(new_messages, model.predict(new_messages)):
        print(f"- {predicted}: {message}")


if __name__ == "__main__":
    main()
