import json
import pickle
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from preprocess import preprocess_for_embeddings

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "faq_canonical.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")

EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "faq_embeddings.pkl")
ANSWERS_PATH = os.path.join(MODEL_DIR, "answer_list.pkl")
INTENTS_PATH = os.path.join(MODEL_DIR, "intent_list.pkl")
QUESTIONS_PATH = os.path.join(MODEL_DIR, "question_list.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- LOAD DATA ----------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    faq_data = json.load(f)   # <-- THIS IS A LIST

questions = []
answers = []
intents = []

for item in faq_data:
    intent = item.get("intent", "")
    answer = item.get("answer", "")
    question_list = item.get("questions", [])

    for q in question_list:
        if not q or not isinstance(q, str):
            continue

        questions.append(preprocess_for_embeddings(q))
        answers.append(answer)
        intents.append(intent)

print(f"Total questions to embed: {len(questions)}")

# ---------- LOAD MiniLM MODEL ----------
print("Loading MiniLM model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- CREATE EMBEDDINGS ----------
print("Creating embeddings...")
embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    show_progress_bar=True
)

# ---------- SAVE FILES ----------
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(embeddings, f)

with open(ANSWERS_PATH, "wb") as f:
    pickle.dump(answers, f)

with open(INTENTS_PATH, "wb") as f:
    pickle.dump(intents, f)

with open(QUESTIONS_PATH, "wb") as f:
    pickle.dump(questions, f)

print("Embeddings saved successfully")
print(f"Files saved in: {MODEL_DIR}")
