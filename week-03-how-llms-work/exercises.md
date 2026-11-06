# Exercises — How LLMs Work

Run the default Python script and save its actual terminal output before changing settings.

1. Compare the number of tokens and words in the prompt. What does the GPT-2 space marker mean?
2. Record the five highest-probability next tokens. Why don't their probabilities necessarily total 100%?
3. Compare greedy generation with sampling using the same prompt.
4. Run the script with `--temperature 0.5` and `--temperature 1.2`, holding the prompt and seed constant. Compare sampled outputs. Does the greedy output change?
5. Try `--prompt "Once upon a time"`. Record how the next-token candidates change.

## Experiment log

- Python version and operating system:
- Installed package versions (`python -m pip freeze`):
- Prompt:
- Five next-token candidates:
- Greedy output:
- Sampled output (seed and temperature):
- One surprising result:

## Further question

Why is this demonstration called inference rather than training? What would training require?

Do not treat a generated continuation as a verified fact.
