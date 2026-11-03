# Exercises — How Models Learn

Work through these after running `train_spam_classifier.py`.

1. **Predict before running:** Read the four messages in `new_messages`. Write down whether you expect each to be classified as spam or ham. Run the script and compare.
2. **Inspect mistakes:** Look at the held-out predictions and confusion matrix. How many legitimate messages were flagged as spam? How many spam messages were missed?
3. **Add training examples:** Add at least four *new* fictional messages with `split=train` to `data/emails.csv`. Keep the test rows unchanged. Rerun the script and record what changed.
4. **Test generalization:** Change only the four `new_messages` examples in the script. Try wording that does not occur in the training data. Explain why the model may struggle.
5. **Explain the pipeline:** In your own words, describe what `CountVectorizer`, `fit()`, and `predict()` each do.

## Record your findings

- Original test accuracy:
- Original confusion matrix:
- Test accuracy after adding training examples:
- One prediction that surprised you:
- What this experiment does **not** prove:

**Important:** Do not edit the held-out test rows while comparing the original and modified models. Otherwise, you are changing the benchmark and cannot make a direct comparison. If you repeatedly tune the model based on the test set, it stops being an unbiased final evaluation; this exercise is for learning, not a rigorous benchmark.
