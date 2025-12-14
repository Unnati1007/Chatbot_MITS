# backend/app.py
from flask import Flask, request, jsonify, session, send_from_directory
from nlp.matcher import FAQMatcher
import random
import csv
import os

# ---------- Flask app ----------
app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = "your_secret_key_here"  # required for session

# ---------- Initialize matcher ----------
matcher = FAQMatcher()

# ---------- Config ----------
MAX_MEMORY = 3
FALLBACK_MESSAGE = "I’m not sure about this. Maybe try asking in a different way?"

# ---------- Polite formatting ----------
POLITE_PREFIXES = [
    "Sure! ",
    "I can help with that. ",
    "Don’t worry, here’s what you can do: "
]

def polite_format(answer_text):
    prefix = random.choice(POLITE_PREFIXES)
    return f"{prefix}{answer_text}"

# ---------- Logging ----------
LOG_FILE = os.path.join(os.path.dirname(__file__), "chat_log.csv")

def log_interaction(user_query, answer, confidence):
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_query", "answer", "confidence"])
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_query, answer, confidence])

# ---------- Helper functions ----------
def update_memory(query):
    if "recent_queries" not in session:
        session["recent_queries"] = []
    memory = session["recent_queries"]
    memory.append(query)
    if len(memory) > MAX_MEMORY:
        memory.pop(0)
    session["recent_queries"] = memory

def check_repetition(query):
    return query in session.get("recent_queries", [])

def update_last_intent(query_result):
    session["last_intent"] = query_result.get("question", None)

def get_last_intent():
    return session.get("last_intent", None)

# ---------- Serve frontend ----------
@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

# ---------- Chatbot API ----------
@app.route("/get_answer", methods=["POST"])
def get_answer():
    data = request.get_json()
    user_query = data.get("query", "").strip()
    
    if not user_query:
        return jsonify({
            "answer": "Please enter a question.",
            "confidence": 0.0,
            "intent": None,
            "suggestions": []
        })

    # 1️⃣ Check repetition
    if check_repetition(user_query):
        update_memory(user_query)
        response = {
            "answer": "You already asked this. Would you like more details?",
            "confidence": 1.0,
            "intent": "repetition",
            "suggestions": []
        }
        log_interaction(user_query, response["answer"], response["confidence"])
        return jsonify(response)

    # 2️⃣ Check vague follow-ups
    follow_up_phrases = ["but i still cannot login", "it didn’t work", "same issue", "still not working"]
    if user_query.lower() in follow_up_phrases:
        last_intent = get_last_intent()
        if last_intent:
            idx = matcher.questions.index(last_intent)
            update_memory(user_query)
            formatted_answer = polite_format(matcher.answers[idx])
            response = {
                "answer": formatted_answer,
                "confidence": 1.0,
                "intent": last_intent,
                "suggestions": []
            }
            log_interaction(user_query, formatted_answer, 1.0)
            return jsonify(response)

    # 3️⃣ Rule-based + semantic matching
    result = matcher.get_best(user_query)

    # 4️⃣ Polite formatting
    if result.get("answer") and result.get("answer") != FALLBACK_MESSAGE:
        result["answer"] = polite_format(result["answer"])
        confidence = result.get("score", 0.0)
        intent = result.get("question", "unknown")
        suggestions = result.get("suggestions", [])
    else:
        result["answer"] = polite_format(FALLBACK_MESSAGE)
        confidence = 0.0
        intent = None
        suggestions = []

    # 5️⃣ Update memory and last intent
    update_memory(user_query)
    if intent not in [None, "unknown"]:
        update_last_intent({"question": intent})

    # 6️⃣ Log interaction
    log_interaction(user_query, result["answer"], confidence)

    # 7️⃣ Return JSON
    return jsonify({
        "answer": result["answer"],
        "confidence": confidence,
        "intent": intent,
        "suggestions": suggestions
    })

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
