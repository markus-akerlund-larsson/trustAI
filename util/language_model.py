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



class openai_client:

    def __init__(self):
        load_dotenv()
        OpenAI.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI()

    def get_embeddings(self, texts):
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return np.array([item.embedding for item in response.data])

    def get_embeddings_batched(self, texts, batch_size=100):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self.get_embeddings(batch))
        return np.array(all_embeddings)

    def get_summary(self, texts):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Rewrite the contents of these texts into one coherent text of similar length. No information not contained in the original is allowed in the summary. And all information in the texts must be included in the summary. The texts to summarize:"},
                {"role": "user", "content": "\n---\n".join(texts)}
            ]
        )
        return response.choices[0].message.content

    def get_category(self, texts):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Create a 1 to 3 word name for the topic of these texts. Fewer words is better. The texts to:"},
                {"role": "user", "content": "\n---\n".join(texts)}
            ]
        )
        return response.choices[0].message.conten