import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
import numpy as np
import underthesea
from sklearn.cluster import KMeans
import re
import json

import uvicorn
from pydantic import BaseModel, conlist
from fastapi import FastAPI
from typing import List

from sklearn.metrics.pairwise import cosine_similarity


index_name = "demo_simcse"
path_index = "config/index.json"
model_embedding = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

def embed_text(text):
    emb = model_embedding.encode(text)
    return emb

def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\,\?]+$-()!*=._", "", row)
    row = row.replace(",", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("*", " ")\
        .replace("=", " ").replace("(", " ")\
        .replace(")", " ").replace("_", " ").replace(".", " ")
    row = row.strip().lower()
    return row

def vector_emb(text):
    stand = standardize_data(text)
    titles = tokenize(stand)
    title_vectors = embed_text(titles)
    return title_vectors

list_text = []
list_id_text = []


def load_data(path_data):
    df = pd.read_csv(path_data).fillna(' ')
    list_id = []
    save_data = []
    cnt = 0

    for index, row in tqdm(df.iterrows()):
        item = {
            'id': row['id'],
            'text': row['text']
        }
        list_text.append(item["text"])
        emb = vector_emb(item["text"])
        list_id.append(cnt)
        save_data.append(emb)
        cnt+=1


    save_data = np.array(save_data)
    list_id = np.array(list_id)
    return save_data, list_id
save_data, list_id = load_data("dataset.csv")

app = FastAPI(
    title="Text New Similarity",
    description="A simple API that use NLP model check Similarity",
    version="0.1",
)


class text_sample(BaseModel):
    id : str
    text: str

class batch(BaseModel):
    #batchs: List[conlist(item_type=float, min_items=1, max_items=20)]
    list_item : List[text_sample]

@app.post("/predict-batch")
async def predict_batch(item: batch, num):
    list_data = item.list_item
    list_text = []
    list_id = []
    for data in list_data:
        id = data.id
        text = data.text
        list_id.append(id)
        list_text.append(text)
    emb_vec = [vector_emb(list_text[i]) for i in range(len(list_text))]
    cluster = KMeans(init="k-means++", n_clusters=5, n_init=5)
    cluster.fit(emb_vec)
    y_hat = cluster.predict(emb_vec)
    pair = zip(y_hat, list_id)

    values, counts = np.unique(y_hat, return_counts=True)
    counts = np.argsort(counts)
    counts = counts[::-1]
    find = counts[0]

    list_id_output = []
    for y_hat, id in pair:
        if y_hat == values[find]:
            list_id_output.append(id)
    k = int(num)
    if len(list_id_output) >= k:
        out = [[{"id": i}, {"text": list_text[int(i)]}]  for i in list_id_output[:k]]
    else:
        out = [[{"id": i}, {"text": list_text[int(i)]}]  for i in list_id_output]

    output = json.loads(json.dumps(out))
    return output

def emb_text(text):
    stand = standardize_data(text)
    titles = tokenize(stand)
    title_vectors = embed_text(titles)
    return title_vectors



@app.get("/predict-text")
async def predict_text(text: str, num: str):
    vec = emb_text(text)
    vec = np.array(vec)
    vec = vec.reshape(1,-1)
    sim_scores = cosine_similarity(vec, save_data)
    score = sim_scores[0]
    pair = zip(list_id, score)                                                          

    sim_scores = sorted(pair, key=lambda x: x[1], reverse=True)
    k = int(num)
    sim_scores = sim_scores[0:k]

    list_out = []
    for id in sim_scores:
        list_out.append(list_text[id[0]])
    return str(list_out)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4500)

    

    