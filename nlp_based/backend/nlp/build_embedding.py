# backend/nlp/build_embeddings.py
import os
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import preprocess_text  # your preprocess function

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faq_cleaned.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
EMB_PATH = os.path.join(OUT_DIR, "faq_embeddings.pkl")
ANS_PATH = os.path.join(OUT_DIR, "answer_list.pkl")
Q_PATH = os.path.join(OUT_DIR, "canonical_questions.pkl")

def load_faqs(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df[["question", "answer"]].dropna().reset_index(drop=True)
    df["clean_q"] = df["question"].astype(str).apply(preprocess_text)
    return df

def build_and_save():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    df = load_faqs()
    model = SentenceTransformer(MODEL_NAME)
    
    embeddings = model.encode(df["clean_q"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-10)
    
    # Save embeddings only
    with open(EMB_PATH, "wb") as f:
        pickle.dump(embeddings_norm, f)
    
    # Save answers list
    with open(ANS_PATH, "wb") as f:
        pickle.dump(df["answer"].tolist(), f)
    
    # Save canonical questions
    with open(Q_PATH, "wb") as f:
        pickle.dump(df["question"].tolist(), f)
    
    print(f"✅ Saved embeddings → {EMB_PATH}")
    print(f"✅ Saved answers → {ANS_PATH}")
    print(f"✅ Saved canonical questions → {Q_PATH}")

if __name__ == "__main__":
    build_and_save()
