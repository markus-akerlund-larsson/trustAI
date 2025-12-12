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

def main():
    database = db.load_database(paths.DATABASE)

    filename = input("Audio file: ").strip()

    metadata = {
        "name": input("Author: ").strip(),
        "category": input("Category: ").strip(),
        "date": datetime.today().strftime('%Y-%m-%d')
    }

    windows, person_name, date = transcribe(filename, database, metadata)

    open_ai = openai_client()
    embeddings = open_ai.get_embeddings(windows)

    window_data = []

    for text, embedding in zip(windows, embeddings):
        window_data.append({
            "text": text,
            "embedding": embedding.tolist(),
            "source_file": filename,
            "author": person_name,
            "date": date
        })

    database.setdefault("metadata", {})
    database.setdefault("data", {})
    database["data"].setdefault(metadata["category"], {})
    existing = database["data"][metadata["category"]]

    for data in existing.values():
        window_data += data["windows"]

    embeddings = []

    for w in window_data:
        embeddings.append(w["embedding"])

    umap = UMAP(
        n_neighbors=2,
        n_components=5,
        metric='cosine')

    reduced_embeddings = umap.fit_transform(embeddings)

    print("Clustering...")
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=3,
        metric='euclidean')


    labels = clustering.fit_predict(reduced_embeddings)


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
    for label, data in clusters.items():
        if label == "-1":
            categories["Uncategorized"] = data
            continue
        countdown += 1
        print(f"{countdown}/{len(clusters)}")
        window_data = [w["text"] for w in data["windows"]]
        category_name = open_ai.get_category(window_data)
        categories[category_name] = data
        data["summary"] = open_ai.get_summary(window_data)

    print("Saving database...")
    database["data"][metadata["category"]] = categories
    with open("./data/database.json", "w", encoding="utf-8") as f:
        json.dump(database, f, ensure_ascii=False, indent=2)

    for data in categories.values():
        for window in data["windows"]:
            window.pop("embedding")
    with open("./data/database_human_readable.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
