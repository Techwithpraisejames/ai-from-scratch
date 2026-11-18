"""Compare four prompt designs on 50 labeled support tickets.

Setup: pip install -r requirements.txt; set OPENAI_API_KEY in your environment.
Usage: python evaluate_prompts.py
This script makes 200 API requests and incurs API usage charges.
"""

import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

load_dotenv()

CATEGORIES = ("Billing", "Technical Issue", "Account Access", "Feature Request")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
DATA = Path(__file__).parent / "data" / "support_tickets.csv"
OUTPUT = Path(__file__).parent / "results.csv"

CATEGORY_RULES = """Classify each customer support ticket into exactly one category:
- Billing: charges, invoices, refunds, pricing, payment, subscriptions, and receipts.
- Technical Issue: broken or incorrect product behavior, bugs, crashes, and outages.
- Account Access: login, passwords, authentication, invitations, and permissions.
- Feature Request: a request for a new capability that does not currently exist.
If a ticket mentions a paid plan but describes an app bug, prefer Technical Issue.
Return only the category name, exactly as spelled above."""

FEW_SHOT = """Examples:
Ticket: I need a copy of my receipt for last month.
Category: Billing
Ticket: My paid account is active but the app still displays the trial badge.
Category: Technical Issue
Ticket: I cannot sign in because my verification code never arrives.
Category: Account Access
Ticket: Please let us schedule automatic data exports.
Category: Feature Request
"""

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": list(CATEGORIES)}},
    "required": ["category"],
    "additionalProperties": False,
}


def classify(client, ticket, version):
    if version == "basic":
        instructions = "Classify this customer support ticket."
    else:
        instructions = CATEGORY_RULES
        if version in ("few_shot", "schema"):
            instructions += "\n\n" + FEW_SHOT

    arguments = {
        "model": MODEL,
        "instructions": instructions,
        "input": ticket,
    }
    if version == "schema":
        arguments["text"] = {
            "format": {
                "type": "json_schema",
                "name": "ticket_category",
                "schema": SCHEMA,
                "strict": True,
            }
        }

    response = client.responses.create(**arguments)
    raw = response.output_text.strip()
    if version == "schema":
        parsed = json.loads(raw)["category"]
        compliant = parsed in CATEGORIES
    else:
        parsed = raw
        compliant = raw in CATEGORIES
    return parsed, raw, compliant, response.usage


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this script.")
    with DATA.open(encoding="utf-8", newline="") as file:
        tickets = list(csv.DictReader(file))
    assert len(tickets) == 50, "Expected exactly 50 labeled tickets."
    assert set(row["expected_category"] for row in tickets) == set(CATEGORIES)
    assert len({row["ticket_id"] for row in tickets}) == 50

    client = OpenAI()
    versions = ("basic", "structured", "few_shot", "schema")
    results = []
    for version in versions:
        print(f"\nTesting {version} on {len(tickets)} tickets with {MODEL}...")
        for row in tickets:
            started = time.perf_counter()
            try:
                predicted, raw, compliant, usage = classify(
                    client, row["message"], version
                )
                error = ""
                input_tokens = usage.input_tokens if usage else ""
                output_tokens = usage.output_tokens if usage else ""
            except BadRequestError as exc:
                raise SystemExit(
                    f"API rejected the request for {version} / {row['ticket_id']}: {exc}\n"
                    "Stopping to avoid repeating a configuration error across all tickets."
                ) from exc
            except Exception as exc:
                predicted, raw, compliant = "", "", False
                input_tokens, output_tokens = "", ""
                error = f"{type(exc).__name__}: {exc}"
                print(f"  {row['ticket_id']}: {error}")

            results.append({
                "version": version,
                "ticket_id": row["ticket_id"],
                "expected": row["expected_category"],
                "predicted": predicted,
                "correct": int(predicted == row["expected_category"] and not error),
                "format_compliant": int(compliant and not error),
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "raw_output": raw,
                "error": error,
            })

    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\nResults (accuracy and format compliance are separate measures)")
    print(f"{'Prompt':<14} {'Accuracy':>10} {'Format':>10} {'Errors':>8}")
    for version in versions:
        subset = [r for r in results if r["version"] == version]
        correct = sum(r["correct"] for r in subset)
        compliant = sum(r["format_compliant"] for r in subset)
        errors = sum(bool(r["error"]) for r in subset)
        print(f"{version:<14} {correct:>3}/50      {compliant:>3}/50      {errors:>3}")
    print(f"\nDetailed per-ticket results saved to {OUTPUT}")
    print("This is a small, authored dataset; results are illustrative, not a benchmark.")


if __name__ == "__main__":
    main()
