"""Explore tokenization, next-token probabilities, and text generation.

Run from any working directory:
    python explore_language_model.py

The first run downloads the public SmolLM2-135M model and tokenizer.
This script performs inference; it does not train a language model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_PROMPT = "The future of artificial intelligence is"
DEFAULT_SEED = 42
DEFAULT_NEW_TOKENS = 24
TOP_K = 5


@dataclass(frozen=True)
class Settings:
    prompt: str
    seed: int
    max_new_tokens: int
    temperature: float


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("--prompt must contain text")
    if not 1 <= args.max_new_tokens <= 100:
        parser.error("--max-new-tokens must be between 1 and 100")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero")

    return Settings(
        prompt=args.prompt,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


def show_tokens(tokenizer, prompt: str) -> torch.Tensor:
    """Display token pieces and their integer IDs."""
    encoded = tokenizer(prompt, return_tensors="pt")
    token_ids = encoded["input_ids"][0]
    pieces = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    print("\nTOKENIZATION")
    print(f"Prompt: {prompt!r}")
    print(f"Number of tokens: {len(token_ids)}")
    for token_id, piece in zip(token_ids.tolist(), pieces):
        print(f"  {token_id:>6}  {piece!r}")
    print("Token pieces use tokenizer-specific notation; decoded text may differ.")
    return encoded["input_ids"]


@torch.inference_mode()
def show_next_token_candidates(model, tokenizer, input_ids: torch.Tensor) -> None:
    """Show the highest-probability next tokens for this exact prompt."""
    outputs = model(input_ids=input_ids)
    next_token_logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=TOP_K)

    print("\nNEXT-TOKEN CANDIDATES (before generating text)")
    for token_id, probability in zip(top_ids.tolist(), top_probabilities.tolist()):
        text = tokenizer.decode([token_id])
        print(f"  {text!r:<24} {probability:.2%}")
    print("These are the top five candidates, not all possible tokens.")


@torch.inference_mode()
def generate_examples(model, tokenizer, input_ids: torch.Tensor, settings: Settings) -> None:
    """Compare greedy decoding with seeded probabilistic sampling."""
    print("\nGREEDY GENERATION (chooses the highest-scoring next token)")
    greedy_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=settings.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(greedy_ids[0], skip_special_tokens=True))

    # Set the random seed immediately before sampling for repeatable CPU runs.
    torch.manual_seed(settings.seed)
    print(
        f"\nSAMPLED GENERATION "
        f"(seed={settings.seed}, temperature={settings.temperature})"
    )
    sampled_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=settings.max_new_tokens,
        do_sample=True,
        temperature=settings.temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(sampled_ids[0], skip_special_tokens=True))


def main() -> None:
    settings = parse_args()
    print(f"Model: {MODEL_ID}")
    print("Device: CPU")
    print("This is pretrained-model inference, not training.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to("cpu")
    model.eval()

    input_ids = show_tokens(tokenizer, settings.prompt)
    show_next_token_candidates(model, tokenizer, input_ids)
    generate_examples(model, tokenizer, input_ids, settings)


if __name__ == "__main__":
    main()
