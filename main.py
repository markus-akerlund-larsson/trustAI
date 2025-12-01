import os
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


load_dotenv()
OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI()
def load_files(path):
    result = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                result.append(f.read())
    return result

def split_sliding_chars(str, winsize=300, overlap=40):
    return [str[i:i + winsize] for i in range(len(str) - overlap)]

def split_sliding_words(text, winsize=160, overlap=80):
    words = text.split()
    step = winsize - overlap
    windows = []

    for i in range(0, len(words), step):
        chunk = words[i:i + winsize]
        if chunk:
            windows.append(" ".join(chunk))

    if  len(windows[-1].split()) < winsize:
        last_win = " ".join(words[-winsize:])
        windows[-1] = last_win
    return windows

def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
     )
    return np.array([item.embedding for item in response.data])

def split_all(strs):
    res = []
    for str in strs:
        res += split_sliding_words(str)
    return res

def save_clusters(texts, labels, filename="res.json"):
    clusters = {}

    for text, label in zip(texts, labels):
        if str(label) not in clusters:
            clusters[str(label)] = []
        clusters[str(label)].append(text)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

texter_python = load_files("./texter_python_control")
texter_gen = load_files("./texter_gen")

wins = split_all(texter_python)
embeddings_map = {}
embeddings = []

countdown = len(wins)

embeddings = get_embeddings(wins)
embeddings = normalize(embeddings)

tsne = TSNE(n_components=2, metric='cosine').fit_transform(embeddings)
plt.figure(figsize=(6, 6))
plt.scatter(tsne[:, 0], tsne[:, 1])
plt.title("t-SNE")
plt.show()

pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(embeddings)

cosine_hdbscan = hdbscan.HDBSCAN(
    min_cluster_size=2,
    metric='cosine',
)

euclidean_hdbscan = hdbscan.HDBSCAN(
    min_cluster_size=2,
    metric='euclidean',
)

cs_labels = cosine_hdbscan.fit_predict(embeddings)
eu_labels = euclidean_hdbscan.fit_predict(embeddings)
reduced_labels = cosine_hdbscan.fit_predict(reduced_embeddings)
tsne_labels = cosine_hdbscan.fit_predict(tsne)

save_clusters(wins, cs_labels, "cosine.json")
save_clusters(wins, eu_labels, "euclidean.json")
save_clusters(wins, reduced_labels, "reduced.json")
save_clusters(wins, tsne_labels, "tsne.json")