# Week 01: AI vs. Machine Learning vs. Deep Learning

**Objective:** Explain the relationship between AI, ML, and deep learning, and decide when simple rules may be sufficient.

## The hierarchy

- **Artificial intelligence (AI):** The broad field of systems performing tasks associated with intelligence, including rule-based approaches, search, and learning-based systems.
- **Machine learning (ML):** A subset of AI where models learn patterns from examples rather than relying exclusively on manually written rules.
- **Deep learning:** A subset of ML that uses multilayer neural networks.
- **Generative AI:** Systems that generate text, images, audio, or other content. Many modern generative models use deep learning; generative AI is a capability category, not another nested layer of the AI/ML/DL hierarchy.

## Practical example: Spam detection

A basic rule-based detector can flag emails containing the exact phrase `claim your prize`. It will miss semantically similar messages using different words. See [spam_rule.py](spam_rule.py).

An ML-based approach would require labeled examples (spam/not spam), a training procedure, and evaluation on held-out messages. A model might generalize beyond exact phrases, but it can still make mistakes. This week introduces the idea; Week 02 demonstrates actual training.

## Exercise

Complete [exercises.md](exercises.md). No libraries, API keys, or paid services are required.

**YouTube lesson:** Link to be added after publication.
