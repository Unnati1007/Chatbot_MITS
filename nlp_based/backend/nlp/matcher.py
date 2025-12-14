# backend/nlp/matcher.py
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from .preprocess import preprocess_text  # ensure preprocess.py is in the same folder
import os

# ---------- File paths & model ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # backend/
EMB_PATH = os.path.join(BASE_DIR, "models", "faq_embeddings.pkl")
ANS_PATH = os.path.join(BASE_DIR, "models", "answer_list.pkl")
Q_PATH = os.path.join(BASE_DIR, "models", "canonical_questions.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- Rule-based intents ----------
GREETINGS = ["hi", "hello", "hey"]

def handle_greeting(text):
    if any(word in text.lower() for word in GREETINGS):
        return "Hello! How can I help you with Moodle, IMS, or registration?"
    return None

def handle_password_issue(text):
    text_lower = text.lower()
    if "forgot" in text_lower and "password" in text_lower:
        return "It seems you forgot your password. Please visit the Moodle password reset page to recover it."
    return None

def handle_registration_issue(text):
    text_lower = text.lower()
    if "registration" in text_lower or "email not received" in text_lower:
        return ("If you are having registration issues, please check your email for confirmation "
                "or contact support at registration@example.com.")
    return None

def handle_emotional_intent(text):
    text_lower = text.lower()
    if "thanks" in text_lower or "thank you" in text_lower:
        return "Glad to help!"
    elif text_lower in ["ok", "hmm", "again"]:
        return "Sure, take your time. Let me know if you need anything."
    return None

def check_rule_based_intents(text):
    """Return a response if any rule-based intent matches."""
    for func in [handle_greeting, handle_password_issue, handle_registration_issue, handle_emotional_intent]:
        response = func(text)
        if response:
            return {"answer": response, "score": 1.0, "rule_based": True}
    return None

# ---------- Semantic matcher ----------
class FAQMatcher:
    def __init__(self, emb_path=EMB_PATH, ans_path=ANS_PATH, q_path=Q_PATH, model_name=MODEL_NAME):
        # Load embeddings, answers, questions
        with open(emb_path, "rb") as f:
            self.embeddings = pickle.load(f)  # shape (N, dim)
        with open(ans_path, "rb") as f:
            self.answers = pickle.load(f)
        with open(q_path, "rb") as f:
            self.questions = pickle.load(f)
        
        # Load SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        self.dim = self.embeddings.shape[1]

    def find_top_k(self, query, k=3):
        """Return top-k matches with cosine similarity."""
        q_clean = preprocess_text(query)
        q_emb = self.model.encode([q_clean], convert_to_numpy=True)
        # normalize for cosine similarity
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        sims = self.embeddings @ q_emb[0]
        idxs = np.argsort(-sims)[:k]
        top_matches = [(self.questions[i], self.answers[i], float(sims[i])) for i in idxs]
        return top_matches

    def get_best(self, query, threshold=0.60):
        """Return best answer using rule-based or semantic matching."""
        # 1️⃣ Check rule-based intents first
        rule_response = check_rule_based_intents(query)
        if rule_response:
            return rule_response

        # 2️⃣ Semantic matching
        top = self.find_top_k(query, k=1)
        question, answer, score = top[0]

        if score >= threshold:
            return {"answer": answer, "score": score, "question": question, "rule_based": False}
        elif score >= 0.40:
            suggestions = self.find_top_k(query, k=3)
            return {"answer": None, "score": score, "suggestions": suggestions, "rule_based": False}
        else:
            return {
                "answer": "I’m not sure, please ask in another way.",
                "score": score,
                "suggestions": [],
                "rule_based": False
            }

# ---------- Quick test ----------
if __name__ == "__main__":
    matcher = FAQMatcher()
    test_queries = [
        "Hi there",
        "I forgot my password",
        "My registration email didn't arrive",
        "Thanks for your help",
        "How do I access Moodle?"
    ]
    for q in test_queries:
        result = matcher.get_best(q)
        print(f"Query: {q}")
        print("Result:", result)
        print("-" * 50)
