import numpy as np

def retrieve(query, index, chunks, model, k=3):
    q_emb = model.encode([query])
    _, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]
