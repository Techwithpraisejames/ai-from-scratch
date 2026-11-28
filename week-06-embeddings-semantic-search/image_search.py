"""Optional text-to-image retrieval with local photos."""
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("query")
    args = parser.parse_args()
    paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
    if not paths:
        raise SystemExit("Provide a folder containing photos.")
    model = SentenceTransformer("google/embeddinggemma-2")
    images = model.encode([{"image":str(p.resolve())} for p in paths], normalize_embeddings=True, batch_size=1)
    query = model.encode_query([args.query], normalize_embeddings=True)
    scores = np.asarray(images) @ np.asarray(query)[0]
    for i in np.argsort(-scores)[:3]:
        print(f"{scores[i]:.6f} {paths[i]}")

if __name__ == "__main__":
    main()
