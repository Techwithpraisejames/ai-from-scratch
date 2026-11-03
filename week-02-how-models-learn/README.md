# How Machine Learning Models Learn — Spam Classification

This lesson builds on the [Week 01 rule-based spam filter](../week-01-ai-fundamentals/README.md). Instead of writing a new `if` rule for each phrase, we train a **real machine-learning classifier** on labeled email examples.

**Goal:** Understand labeled data, numerical text features, training, inference (prediction), and evaluation on held-out examples.

## Files

- [`data/emails.csv`](data/emails.csv): 44 fictional, labeled emails; 32 training examples and 12 held-out test examples.
- [`train_spam_classifier.py`](train_spam_classifier.py): Loads data, validates the split, trains a model, evaluates it, and predicts new messages.
- [`requirements.txt`](requirements.txt): The Python dependency.
- [`exercises.md`](exercises.md): Practice tasks.

## Requirements and setup

Install Python **3.10 or later**. Run these commands from the `week-02-how-models-learn` folder.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_spam_classifier.py
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_spam_classifier.py
```

If PowerShell prevents activation, you can invoke `.venv\Scripts\python.exe` directly for installation and execution.

## How the demonstration works

1. **Examples and labels:** Each CSV row has a message, a label (`spam` or `ham`, meaning legitimate email), and a fixed `train` or `test` split.
2. **Features:** `CountVectorizer` converts text into numerical word counts. It learns its vocabulary from training messages, not from test messages.
3. **Training:** `MultinomialNB` (multinomial Naive Bayes) learns statistical associations between word counts and labels.
4. **Inference:** `predict()` assigns labels to messages using the fitted pipeline.
5. **Evaluation:** Accuracy and a confusion matrix compare predictions with the known labels of held-out messages. A false positive flags legitimate email as spam; a false negative misses spam.

The pipeline fits the vectorizer and classifier together. Keeping the test set out of `fit()` prevents the most obvious form of test-data leakage.

## Reproducibility and limitations

- The split is explicitly recorded in the CSV, so it does not change between runs. No random seed is needed for this deterministic example.
- The dataset is **small, fictional, and deliberately simple**. High accuracy on 12 test messages does not demonstrate real-world spam detection performance.
- This model uses word counts. It does not understand meaning, guarantee synonym recognition, or generalize reliably to unfamiliar wording.
- The script prints the **actual output on your machine**. No result or accuracy figure is assumed in advance.
- The additional predictions at the end are illustrations only; they are **not included** in the test accuracy.
- Do not use this demonstration as a production spam or security filter.

## Try it yourself

Run the script, record the output, and then complete [the exercises](exercises.md). For the video walkthrough, we'll use the actual observed results rather than a fabricated example.
