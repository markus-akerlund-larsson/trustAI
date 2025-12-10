import os
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from umap import UMAP
import seaborn as sns

import util.windowing
from util.language_model import openai_client

np.random.seed(42)

def text_placeholder(path):
    result = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                result.append(f.read())
    return result

def load_file(category, filename):
    # windows = whisper(filename)
    windows = util.windowing.split_all(text_placeholder("./texter_python_control"))

    open_ai = openai_client()
    embeddings = open_ai.get_embeddings(windows)

    texts = []

    for text, embedding in zip(windows, embeddings):
        texts.append({
            "text": text,
            "embedding": embedding,
            "source_file": filename,
            #"category": open_ai.get_category(text),
            #"summary": open_ai.get_summary([text]),
            "timestamp": "00:00:00 (placeholder)",
        })

    with open("/data/database.json", "r") as f:
        database = json.load(f)

    database["data"].setdefault(category, {})
    existing = database["data"][category]

    for data in existing.values():
        texts += data

    umap = UMAP(
        n_neighbors=2,
        n_components=5,
        metric='cosine')

    reduced_embeddings = umap.fit_transform(embeddings)

    clustering = hdbscan.HDBSCAN(
        min_cluster_size=5,
        metric='cosine')

    labels = clustering.fit_predict(reduced_embeddings)

    clusters = {}
    for window, label in zip(texts, labels):
        category_name = str(label)
        clusters.setdefault(category_name, {
                "windows": [],
            })
        clusters[str(label)]["windows"].append(window)

    countdown = 0
    categories = {}
    for label, data in clusters.items():
        if label == "-1":
            categories["Uncategorized"] = data
            continue
        countdown += 1
        print(f"{countdown}/{len(clusters)}")
        texts = [w["text"] for w in data["windows"]]
        category_name = openai_client.get_category(texts)
        categories[category_name] = data
        data["summary"] = openai_client.get_summary(texts)

    with open("database2.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

