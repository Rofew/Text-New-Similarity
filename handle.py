import os
import pandas as pd
from tqdm import tqdm

with open("data/dataset.txt", encoding="utf-8") as f:
    data = f.readlines()

texts = []
ids = []
for line in range(len(data)):
    if line:
        id,text = data[line].split(",",1)
        texts.append(text)
        ids.append(id)
        print(line)
    else:
        continue

import pandas as pd
df=pd.DataFrame.from_dict({
    "id": ids,
    "text": texts
})

df.to_csv("dataset.csv", index=False)