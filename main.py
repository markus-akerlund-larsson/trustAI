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
sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

np.random.seed(42)
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

def split_sliding_words(text, winsize=160, overlap=40):
    words = text.split()
    if(len(words) <= winsize):
        return [text]

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

def get_embeddings_batched(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
       batch = texts[i:i + batch_size]
       all_embeddings.extend(get_embeddings(batch))
    return np.array(all_embeddings)

def get_summary(texts):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Rewrite the contents of these texts into one coherent text of similar length. No information not contained in the original is allowed in the summary. And all information in the texts must be included in the summary. The texts to summarize:"},
            {"role": "user", "content": "\n---\n".join(texts)}
        ]
    )
    return response.choices[0].message.content

def get_category(texts):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Create a 1 to 3 word name for the topic of these texts. Fewer words is better. The texts to:"},
            {"role": "user", "content": "\n---\n".join(texts)}
        ]
    )
    return response.choices[0].message.content

def split_all(strs, winsize=160, overlap=40):
    res = []
    for str in strs:
        res += split_sliding_words(str, winsize, overlap)
    return res

def group_clusters(texts, labels):
    clusters = {}
    for text, label in zip(texts, labels):
        if str(label) not in clusters:
            clusters[str(label)] = {
                "texts": [],
            }
        clusters[str(label)]["texts"].append(text)
    return clusters

def save_clusters(clusters, filename="res.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

def add_summary(clusters):
    countdown = 0
    for cluster, contents in clusters.items():
        if cluster == "-1":
            continue
        countdown += 1
        print(f"{countdown}/{len(clusters)}")
        clusters[cluster]["summary"] = get_summary(contents["texts"])
        clusters[cluster]["category"] = get_category(contents["texts"])
    return clusters

texter_python = load_files("./texter_python_control")
texter_gen = load_files("./texter_gen")

pwins = split_all(texter_python)
gwins = split_all(texter_python, 40, 10)

wins = gwins

embeddings_map = {}
embeddings = []


embeddings = get_embeddings(wins)
embeddings = normalize(embeddings)


euclidean_hdbscan = hdbscan.HDBSCAN(
    min_cluster_size=3,
    metric='euclidean',
)
# silhouette score?

#cs_labels = cosine_hdbscan.fit_predict(embeddings)
#cs_clusters = group_clusters(wins, cs_labels)
#add_summary(cs_clusters)
#save_clusters(cs_clusters, "cosine.json")

#pca = PCA(n_components=3)
#reduced_embeddings = pca.fit_transform(embeddings)
#pca_labels = euclidean_hdbscan.fit_predict(reduced_embeddings)
#pca_clusters = group_clusters(wins, pca_labels)
#add_summary(pca_clusters)
#save_clusters(pca_clusters, "pca.json")

best_unclustered = 900000
best_val = 0
best_umap_clusters = {}
best_cosine_hdbscan = None
no_tests = 0
if(no_tests != 0):
    for v in range(5, 5, 5):
        umap = UMAP(
            n_neighbors=2,
            n_components=5,
            metric='cosine')
        cosine_hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric='cosine',
        )
        tests = []
        for i in range(no_tests):
            umap_embeddings = umap.fit_transform(embeddings)
            umap_labels = cosine_hdbscan.fit_predict(umap_embeddings)
            tests.append(len(group_clusters(wins, umap_labels)["-1"]["texts"]))
        unclustered = sum(tests) / no_tests
        print("Value: " + str(v) + "\tunclustered: " + str(unclustered))

#plt.scatter(umap_embeddings[:,0], umap_embeddings[:,1])
#plt.show()

umap = UMAP(
    n_neighbors=2,
    n_components=5,
    metric='cosine')
cosine_hdbscan = hdbscan.HDBSCAN(
    min_cluster_size=5,
    metric='cosine')

umap_embeddings=umap.fit_transform(embeddings)

umap_labels = cosine_hdbscan.fit_predict(umap_embeddings)
umap_clusters = group_clusters(wins, umap_labels)
print(len(umap_clusters["-1"]["texts"]))
add_summary(umap_clusters)
save_clusters(umap_clusters, "umap.json")
