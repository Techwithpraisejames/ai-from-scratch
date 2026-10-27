# Week 01 Exercise: Design a Spam Detector

**Time:** 10–15 minutes. **Prerequisite:** Basic programming logic.

1. **Rule-based approach:** Write pseudocode that flags an email containing `claim your prize`. Explain how it handles capitalization.
2. **Training examples:** Invent six short, fictional email messages. Label three SPAM and three NOT SPAM. Do not use real personal emails or sensitive information.
3. **Failure analysis:** Give one message the rule would miss. Explain one way an ML model trained on your six examples could make a mistake.
4. **Model selection:** Would a deep neural network be your first choice for a tiny dataset? Explain your reasoning.

## Reflection questions

- What information must you define manually in the rule-based version?
- What information must you supply to train an ML classifier?
- Why is evaluation on unseen examples important?

## Optional challenge

Run `python spam_rule.py`, then add a second phrase and observe how the rule changes. Remember: this toy script is not a production spam filter.
