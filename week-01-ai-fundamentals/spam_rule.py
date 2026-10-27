"""Toy exact-phrase spam filter for Week 01. Not for production use."""


def is_spam(message: str) -> bool:
    """Return True only if the configured phrase occurs in the message."""
    return "claim your prize" in message.lower()


if __name__ == "__main__":
    examples = [
        "Claim your prize today!",
        "Collect your reward today!",
        "Your meeting is scheduled for tomorrow.",
    ]
    for message in examples:
        print(f"{'SPAM' if is_spam(message) else 'NOT FLAGGED'}: {message}")
