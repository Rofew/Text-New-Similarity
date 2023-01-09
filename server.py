import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
import numpy as np
import underthesea
import re

import uvicorn
from pydantic import BaseModel, conlist
from fastapi import FastAPI
from typing import List

from sklearn.metrics.pairwise import cosine_similarity


index_name = "demo_simcse"
path_index = "config/index.json"
model_embedding = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
path_data = "data/data_title.csv"

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
    stand = standardize_data(text["title"])
    titles = tokenize(stand)
    title_vectors = embed_text(titles)
    # titles = [tokenize(doc["title"]) for doc in docs]
    # title_vectors = embed_text(titles)

    # request = text
    # request["_op_type"] = "index"
    # request["_index"] = index_namez
    # request["title_vector"] = title_vectors

    return title_vectors



df = pd.read_csv(path_data).fillna(' ')

list_id = []
save_data = []

list_text = []
list_id_text = []
cnt = 0
for index, row in tqdm(df.iterrows()):
    cnt+=1
    if cnt == 1000:
        break
    item = {
        'id': row['id'],
        'title': row['title']
    }
    list_text.append(item["title"])
    emb = vector_emb(item)
    list_id.append(item["id"])
    save_data.append(emb)

# intput = {'id': "1",
#           'title':"ai se mai em anh"}

# id, pro = emb(intput)

save_data = np.array(save_data)
list_id = np.array(list_id)







# sim_scores = sorted(pair, key=lambda x: x[1], reverse=True)
# sim_scores = sim_scores[1:11]
# for id in sim_scores:
#     print(list_text[id[0]])


app = FastAPI(
    title="Text New Similarity",
    description="A simple API that use NLP model check Similarity",
    version="0.1",
)


def emb_text(text):
    stand = standardize_data(text)
    titles = tokenize(stand)
    print(titles)
    title_vectors = embed_text(titles)
    # titles = [tokenize(doc["title"]) for doc in docs]
    # title_vectors = embed_text(titles)

    # request = text
    # request["_op_type"] = "index"
    # request["_index"] = index_name
    # request["title_vector"] = title_vectors

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