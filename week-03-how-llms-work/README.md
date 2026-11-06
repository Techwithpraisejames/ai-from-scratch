# How Large Language Models Work: Tokens and Text Generation

This lesson follows the [spam-classifier experiment](../week-02-how-models-learn/README.md). The previous classifier predicted **spam** or **ham**. Here, a pretrained *causal language model* predicts tokens and generates a sequence of text.

You will run **HuggingFaceTB/SmolLM2-135M**, a compact pretrained causal language model released in 2024, locally on your laptop using Python and VS Code. It is useful for demonstrating the mechanics of generation, but its small size means it is **not** representative of the capabilities of modern chat assistants.

## Files

- [`explore_language_model.py`](explore_language_model.py) — token inspection, next-token probabilities, greedy generation, and seeded sampling.
- [`requirements.txt`](requirements.txt) — pinned Python dependencies.
- [`exercises.md`](exercises.md) — reproducible experiments and reflection questions.
- [`.gitignore`](.gitignore) — excludes local virtual environments, caches, and Python artifacts.

## What you'll learn

1. **Tokens:** A tokenizer turns text into token IDs. Tokens can represent words, word fragments, or punctuation.
2. **Next-token prediction:** The model produces scores for possible next tokens. Softmax converts those scores into a probability distribution.
3. **Autoregressive generation:** The model repeatedly selects a next token and uses the growing sequence as context.
4. **Decoding:** Greedy decoding picks the highest-scoring token at each step; sampling selects tokens probabilistically. Temperature changes the sampling distribution.
5. **Limits:** Fluent generated text does not guarantee factual accuracy. This model is pretrained; this script does not train it.

Transformers use attention to process relationships among tokens in context. This script lets you observe tokenization and generation, **not** inspect attention weights or prove how the model reasons.

## Prerequisites

- Python **3.10–3.12**, 64-bit, and VS Code with the Python extension.
- Internet access **on the first run** to download the public model/tokenizer files (several hundred MB in total, depending on the downloaded files); later runs reuse the local Hugging Face cache.
- Sufficient free disk space and RAM for Python, PyTorch, and the model. CPU execution is intentional, so the first load and generation may take some time.

> The pinned versions are for reproducing this lesson. Installation depends on your OS and Python version. We will validate the actual output on your laptop before writing the video script.

## Setup in VS Code

Clone or download the repository, open the `week-03-how-llms-work` folder in VS Code, and open **Terminal → New Terminal**. All commands below assume your terminal is inside this folder.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe explore_language_model.py
```

If you have Python 3.10 or 3.12 instead of 3.11, replace `py -3.11` with your installed version (for example `py -3.12`). These commands do not require PowerShell script activation.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python explore_language_model.py
```

If `python3` points to an unsupported version, use an installed Python 3.10–3.12 executable to create the environment.

### VS Code interpreter

Use **Ctrl+Shift+P** (or **Cmd+Shift+P** on macOS) → **Python: Select Interpreter** → choose the Python executable inside `.venv`. Run the script from the integrated terminal. Avoid installing dependencies globally.

## Expected *sections* of output

The exact generated text must come from your run; we have **not** invented or prefilled results.

1. **TOKENIZATION:** Token IDs and readable token pieces for the prompt.
2. **NEXT-TOKEN CANDIDATES:** Five next-token candidates and their probabilities, calculated before generation.
3. **GREEDY GENERATION:** A deterministic continuation for the given model and prompt.
4. **SAMPLED GENERATION:** A continuation using `seed=42` and `temperature=0.8` by default.

A fixed seed makes sampling repeatable in the same supported CPU environment and software configuration, but outputs can differ across library versions or platforms. Generation may be grammatically awkward or factually wrong; that is part of the lesson.

## Run a second experiment

Windows:

```powershell
.venv\Scripts\python.exe explore_language_model.py --prompt "Once upon a time" --seed 7 --temperature 1.2
```

macOS / Linux (with the virtual environment active):

```bash
python explore_language_model.py --prompt "Once upon a time" --seed 7 --temperature 1.2
```

Other supported flags: `--max-new-tokens 40`, `--seed 123`, `--temperature 0.5`. Higher temperature generally flattens the sampling distribution; lower positive temperature sharpens it. **Temperature has no effect on the greedy run.**

## Troubleshooting

- **`No module named transformers` / `torch`:** Select the `.venv` interpreter and install from `requirements.txt` using that interpreter.
- **`py -3.11` not found:** Install Python 3.11 (64-bit) or use an installed supported Python version.
- **Download or connection error:** Check your internet connection and retry. The first run must download model files from Hugging Face.
- **Low disk space / memory:** Close memory-heavy apps and ensure you have room for the model and Python packages.
- **Unexpected text:** Generated text can be repetitive, incomplete, or inaccurate. Preserve it as actual evidence rather than editing the output for the tutorial.
- **Unsupported environment:** The standard installation may not work on every processor/OS combination. Report your OS, Python version, and complete error before changing dependencies.

## Reproducibility and safety

The model identifier and Python package versions are specified, but the model repository's default revision can change. For strict long-term reproducibility, record the downloaded model revision and installed package versions alongside your terminal output; a commit-pinned model revision can be added after verification. The script uses CPU only and downloads model files through Hugging Face Transformers. Token pieces use model-specific conventions; do not assume a particular visible space marker or one-token-per-word mapping.

Do not treat model output as verified facts or instructions. Do not use this small demo model for production advice or high-stakes decisions.

Complete [the exercises](exercises.md) after you have captured the first run.
