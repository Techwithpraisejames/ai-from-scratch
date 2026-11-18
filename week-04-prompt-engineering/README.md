# Prompt Engineering: Design, Test, and Evaluate AI Instructions

**AI From Scratch — Week 4**

In [Week 3](../week-03-how-llms-work/README.md), we explored tokens, next-token predictions, and decoding. This week, we use the OpenAI API to see how changing instructions affects an instruction-tuned language model's output. We evaluate the same **50 labeled customer support tickets** under four configurations.

## What you'll learn

- **Prompt structure:** task, context, category definitions, constraints, and expected output.
- **Zero-shot prompting:** instructions without labeled examples.
- **Few-shot prompting:** including labeled examples that are separate from the evaluation dataset.
- **API instructions:** how developer-provided instructions differ from user input.
- **Structured outputs:** enforcing a JSON schema via an API, rather than merely requesting JSON in text.
- **Evaluation:** accuracy, exact-format compliance, failures, and token usage.

## The experiment

The task is to classify support tickets into **Billing**, **Technical Issue**, **Account Access**, or **Feature Request**.

The dataset at [data/support_tickets.csv](data/support_tickets.csv) contains 50 manually labeled, authored tickets: 13 Billing, 13 Technical Issue, 12 Account Access, and 12 Feature Request. Some examples are deliberately ambiguous. These labels are educational ground truth based on the category definitions, not externally validated judgments.

We run the **same 50 tickets** against four configurations:

| Version | Instructions | Examples | Output |
| --- | --- | --- | --- |
| `basic` | "Classify this customer support ticket." | None | Unconstrained text |
| `structured` | Defined categories, rules, exact label requirement | None | Unconstrained text |
| `few_shot` | Same rules | Four separate labeled examples | Unconstrained text |
| `schema` | Same rules and examples | Four separate labeled examples | Strict API JSON schema |

**Important:** The fourth version changes the API's output constraint in addition to the prompt. Its format compliance is not evidence that wording alone improved performance.

## Prerequisites

- Python 3.10 or newer
- An OpenAI API key with available API credits
- Network access to the OpenAI API

The experiment uses the OpenAI Python SDK and defaults to `gpt-6-luna`. You can set `OPENAI_MODEL` to another model supporting the Responses API and structured outputs. Model support and prices may change; verify current availability in the [OpenAI API documentation](https://platform.openai.com/docs/).

## Run locally

From the repository root:

### Windows PowerShell

```powershell
cd week-04-prompt-engineering
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and replace the placeholder with your actual key
.\.venv\Scripts\python.exe evaluate_prompts.py
```

### macOS / Linux

```bash
cd week-04-prompt-engineering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and replace the placeholder with your actual key
python evaluate_prompts.py
```

Never commit API keys, `.env` files, or exported secrets. The script loads a local `.env` file with `python-dotenv`. Copy `.env.example` to `.env` and replace the placeholder with your own API key. `.env` is ignored by Git.

**Cost warning:** The complete experiment sends **200 API requests** (50 tickets × 4 configurations), which incur token-based API charges. Inspect the code and current API pricing before running it.

## What the script records

The script prints a summary and writes `results.csv` locally with one row per request. For each row it records expected label, predicted label, exact-match correctness, format compliance, latency, input/output tokens, raw response, and any error.

**Accuracy** is the percentage of all 50 tickets where the predicted category exactly matches the manually assigned label. **Format compliance** checks whether the model returned the exact category label (or a valid category in the enforced schema). These are separate measures: a perfectly formatted prediction can still be incorrect.

Errors count as failures. The script does not set `temperature` because GPT-6 Luna rejects that parameter. Identical outputs are not guaranteed across runs or model updates. The 50 authored examples are a small teaching dataset, not a production benchmark. Few-shot examples are **not drawn from these 50 evaluation rows**.

We have not run the API experiment in this repository and make no claims about which version will win. Run it and examine the failures before drawing conclusions.

## Files

- [`evaluate_prompts.py`](evaluate_prompts.py) — API calls and evaluation
- [`data/support_tickets.csv`](data/support_tickets.csv) — 50 labeled examples
- [`requirements.txt`](requirements.txt) — Python dependency
- [`exercises.md`](exercises.md) — follow-up experiments

**Next:** [Week 5 — Neural Networks and Backpropagation](../week-05-neural-networks-backpropagation/README.md).
