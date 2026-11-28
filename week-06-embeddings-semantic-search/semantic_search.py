import argparse
import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODEL = "google/embeddinggemma-2"

def words(text):
    return set(re.findall(r"\w+", text.lower()))

def keyword_scores(query, documents):
    return np.array([len(words(query) & words(d["text"])) for d in documents])

def rank(scores, k=3):
    return np.argsort(-scores, kind="stable")[:k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    data = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
    docs = data["documents"]
    print("Loading", MODEL, flush=True)
    model = SentenceTransformer(MODEL, device=args.device)
    matrix = model.encode_document([d["text"] for d in docs], normalize_embeddings=True, batch_size=4, show_progress_bar=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    print("Document matrix shape:", matrix.shape)
    print("First vector, first 8 values:", matrix[0, :8])
    print("First vector norm:", np.linalg.norm(matrix[0]))
    phrases = ["How do I get my money back?", "Requesting a refund for my subscription", "How do I reset my password?"]
    vectors = np.asarray(model.encode(phrases, prompt_name="STS", normalize_embeddings=True, batch_size=4), dtype=np.float32)
    print("\nSentence similarity experiment:")
    for i in range(len(phrases)):
        for j in range(i+1,len(phrases)):
            print(f"{phrases[i]} | {phrases[j]}: {vectors[i] @ vectors[j]:.6f}")

    queries = data["queries"] if args.query is None else [{"text":args.query,"relevant":[]}]
    qmatrix = np.asarray(model.encode_query([q["text"] for q in queries], normalize_embeddings=True, batch_size=4), dtype=np.float32)
    results = []
    metrics = {"keyword":[0,0], "semantic":[0,0]}
    count = 0
    for q, vector in zip(queries,qmatrix):
        print("\nQUERY:", q["text"])
        relevant = set(q["relevant"])
        if relevant:
            count += 1
        row = {"query":q["text"],"relevant":q["relevant"]}
        for label, scores in [("keyword",keyword_scores(q["text"],docs)),("semantic",matrix @ vector)]:
            indices = rank(scores)
            if label == "keyword":
                indices = [i for i in indices if scores[i] > 0]
            hits = [{**docs[i],"score":float(scores[i])} for i in indices]
            row[label] = hits
            print(label.upper())
            for hit in hits:
                print(f"  {hit['id']} {hit['score']:.4f} {hit['text']}")
            if not hits:
                print("  No keyword match")
            if relevant:
                metrics[label][0] += bool(hits and hits[0]["id"] in relevant)
                metrics[label][1] += bool(relevant & {h["id"] for h in hits})
        results.append(row)
    if count:
        print("\nANSWERABLE QUERIES:",count)
        for label,(top1,top3) in metrics.items():
            print(f"{label}: Hit@1 {top1}/{count} ({top1/count:.1%}); Hit@3 {top3}/{count} ({top3/count:.1%})")
        print("Unanswerable queries are excluded from hit-rate metrics; inspect their results separately.")
    (ROOT / "results.json").write_text(json.dumps({"model":MODEL,"shape":list(matrix.shape),"pairwise_similarity":(vectors @ vectors.T).tolist(),"results":results,"answerable":count,"hits":metrics},indent=2),encoding="utf-8")
    if args.plot:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(matrix)
        plt.scatter(coords[:,0],coords[:,1])
        for (x,y),doc in zip(coords,docs):
            plt.annotate(doc["id"],(x,y))
        plt.title("Document embeddings: 2D PCA projection")
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.tight_layout()
        plt.savefig(ROOT / "embedding_projection.png",dpi=180)
        plt.close()
    print("Saved results.json")

if __name__ == "__main__":
    main()
