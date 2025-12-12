import os
from sklearn.cluster._hdbscan import hdbscan
import json
import numpy as np
from umap import UMAP
import util.database as db
from util import paths
from util.language_model import openai_client
from util.whisper import transcribe
from datetime import datetime

np.random.seed(42)

def clustering(embeddings, min_cluster_size=3, reduced_dimensions=5):
    if reduced_dimensions > 0:
        umap = UMAP(
            n_neighbors=2,
            n_components=reduced_dimensions,
            metric='cosine')

        reduced_embeddings = umap.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings

    print("Clustering...")
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean')

    return clustering.fit_predict(reduced_embeddings)

def main():
    database = db.load(paths.DATABASE)

    metadata = {
        "audio_file": input("Audio file: ").strip(),
        "author": input("Author: ").strip(),
        "category": input("Category: ").strip(),
        "date": datetime.today().strftime('%Y-%m-%d')
    }
    database = db.add_metadata(database, metadata["category"], metadata)

    windows = transcribe(metadata["audio_file"], database, metadata)

    open_ai = openai_client()
    embeddings = open_ai.get_embeddings(windows)

    window_data = []

    for text, embedding in zip(windows, embeddings):
        window_data.append({
            "text": text,
            "embedding": embedding.tolist(),
            "source_file": metadata["audio_file"],
            "author": metadata["author"],
            "date": metadata["date"]
        })

    existing = db.get_existing_windows(database, metadata["category"])
    for win in existing:
        window_data += win

    embeddings = [w["embedding"] for w in window_data]

    labels = clustering(embeddings)

    clusters = {}
    for window, label in zip(window_data, labels):
        category_name = str(label)
        clusters.setdefault(category_name, {
                "windows": [],
            })
        clusters[str(label)]["windows"].append(window)

    print("Clustering done.")
    countdown = 0
    categories = {}
    print("Generating category names and summaries..")
    for label, win in clusters.items():
        if label == "-1":
            categories["Uncategorized"] = win
            continue
        countdown += 1
        print(f"{countdown}/{len(clusters)}")
        window_data = [w["text"] for w in win["windows"]]
        category_name = open_ai.get_category(window_data)
        categories[category_name] = win
        win["summary"] = open_ai.get_summary(window_data)

    database["data"][metadata["category"]] = categories

    print("Saving database...")
    db.save(database, paths.DATABASE)

    # Create human readable version
    for win in categories.values():
        for window in win["windows"]:
            window.pop("embedding")
    with open("./data/categories.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
