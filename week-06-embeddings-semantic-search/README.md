# Week 6: Embeddings and Semantic Search

Use Google's EmbeddingGemma 2 locally to inspect sentence embeddings and compare keyword overlap with cosine-similarity retrieval. No vector database or paid inference API required.

## Setup

Python 3.11+ recommended. If you already have a working EmbeddingGemma 2 environment, reuse it rather than replacing its dependencies. Otherwise create a virtual environment and run `python -m pip install -r requirements.txt`. The first model download requires internet access and any applicable Hugging Face access approval.

## Run

From this folder:

```powershell
python semantic_search.py
python semantic_search.py --query "How do I get my money back?"
python semantic_search.py --plot
```

The full evaluation uses 24 synthetic support document titles, 13 answerable queries with one labeled target each, and two unanswerable queries. Hit@1 and Hit@3 measure whether the labeled target appears in the respective result positions. Keyword overlap is a deliberately simple baseline, not BM25. Similarity scores are not probabilities. Results are written to results.json; the plot is optional.

For optional image retrieval, provide your own folder of images:

```powershell
python image_search.py ./photos "a person at the beach"
```

The scripts follow the [EmbeddingGemma 2 documentation](https://huggingface.co/docs/transformers/model_doc/embedding_gemma2). Runtime results have not yet been verified.

## Exercises

- Try a paraphrase and compare both rankings.
- Search for an exact error code and explain the result.
- Examine the two unanswerable questions. Why does semantic retrieval still rank documents?
- Inspect the PCA plot and explain why a two-dimensional projection cannot preserve all original relationships.
- Try truncating embeddings to 128 dimensions, renormalize them and compare retrieval quality.
