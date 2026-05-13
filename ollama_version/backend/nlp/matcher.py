import os
import pickle
import numpy as np
import sqlite3

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import preprocess_for_embeddings
from .rules import check_rules

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "faq_embeddings.pkl")
ANSWERS_PATH = os.path.join(MODEL_DIR, "answer_list.pkl")
INTENTS_PATH = os.path.join(MODEL_DIR, "intent_list.pkl")
QUESTIONS_PATH = os.path.join(MODEL_DIR, "question_list.pkl")

# =====================================================
# LOAD DATA (ONCE AT STARTUP)
# =====================================================
with open(EMBEDDINGS_PATH, "rb") as f:
    FAQ_EMBEDDINGS = pickle.load(f)

with open(ANSWERS_PATH, "rb") as f:
    ANSWERS = pickle.load(f)

with open(INTENTS_PATH, "rb") as f:
    INTENTS = pickle.load(f)

with open(QUESTIONS_PATH, "rb") as f:
    QUESTIONS = pickle.load(f)

# =====================================================
# LOAD MiniLM MODEL (ONCE)
# =====================================================
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# CONFIDENCE THRESHOLDS
# =====================================================
HIGH_CONFIDENCE = 0.65
LOW_CONFIDENCE = 0.40

FOLLOW_UP_PHRASES = ["still", "again", "but", "does not work", "doesn't work", "cannot", "can't", "what now", "then what", "next"]

def is_follow_up(text: str) -> bool:
    text = text.lower()
    return any(phrase in text for phrase in FOLLOW_UP_PHRASES)

def find_best_match(user_query: str, memory: list | None = None) -> dict:
    if not user_query or not isinstance(user_query, str):
        return {"type": "fallback", "answer": None, "confidence": 0.0, "intent": None, "answer_id": None}

    # RULE-BASED FAST PATH
    rule_response = check_rules(user_query)
    if rule_response:
        return rule_response

    # FOLLOW-UP LINKING
    if memory and is_follow_up(user_query):
        last = memory[-1]
        return {"type": "answer", "answer": last["answer"], "confidence": last["confidence"], "intent": last["intent"], "answer_id": last["answer_id"]}

    # SEMANTIC MATCHING
    processed_query = preprocess_for_embeddings(user_query)
    query_embedding = MODEL.encode([processed_query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, FAQ_EMBEDDINGS)[0]

    # RL BOOST
    try:
        backend_dir = os.path.dirname(BASE_DIR) 
        db_path = os.path.join(backend_dir, "feedback.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT answer_id, AVG(rating) FROM feedback GROUP BY answer_id')
            ratings = dict(cursor.fetchall())
            conn.close()
            for idx, avg_rating in ratings.items():
                if idx is not None and idx < len(similarities):
                    if avg_rating >= 4.0: similarities[idx] += 0.1
                    elif avg_rating < 3.0: similarities[idx] -= 0.1
    except Exception as e:
        print(f"⚠️ RL Boost failed: {e}")

    top_indices = similarities.argsort()[::-1][:4]
    top_scores = similarities[top_indices]
    top_answers = [ANSWERS[i] for i in top_indices]
    top_intents = [INTENTS[i] for i in top_indices]
    top_questions = [QUESTIONS[i] for i in top_indices]

    best_score = float(top_scores[0])
    best_index = int(top_indices[0])

    # REPETITION DETECTION
    if memory:
        last = memory[-1]
        if last["intent"] == top_intents[0]:
            return {"type": "repeat", "answer": "You already asked this. Would you like more details?", "confidence": best_score, "intent": top_intents[0], "answer_id": best_index}

    suggestions = top_questions[1:]

    if best_score >= HIGH_CONFIDENCE:
        return {"type": "answer", "answer": top_answers[0], "confidence": best_score, "intent": top_intents[0], "answer_id": best_index, "suggestions": suggestions}

    if LOW_CONFIDENCE <= best_score < HIGH_CONFIDENCE:
        return {"type": "clarify", "answer": None, "confidence": best_score, "intent": top_intents[0], "answer_id": best_index, "suggestions": suggestions}

    return {"type": "fallback", "answer": None, "confidence": best_score, "intent": top_intents[0], "answer_id": best_index, "suggestions": suggestions}
