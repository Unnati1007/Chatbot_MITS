import os
# 🤫 Silence TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, session
from flask_cors import CORS

import requests
import json

from nlp.matcher import find_best_match

# =====================================================
# FLASK APP SETUP
# =====================================================
app = Flask(__name__)
CORS(app)

# 🔐 Required for session memory
app.secret_key = "drive_safe_chatbot_secret_key"

# Limit session memory
MAX_MEMORY = 3


# =====================================================
# HELPERS — SESSION MEMORY
# =====================================================
def get_memory():
    """Return last interactions from session"""
    return session.get("memory", [])


def update_memory(entry: dict):
    """Store last MAX_MEMORY interactions"""
    memory = get_memory()
    memory.append(entry)
    session["memory"] = memory[-MAX_MEMORY:]


def clear_memory():
    session["memory"] = []


# =====================================================
# OLLAMA HELPERS
# =====================================================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3"

def call_ollama(user_query, context):
    """Call local Ollama server for conversational response"""
    system_prompt = (
        "You are the MITS AI Assistant, a helpful and professional university companion. "
        "Use the provided context to answer the user's question accurately. "
        "If the context is irrelevant, answer based on your general knowledge but mention you are an AI. "
        "Keep the tone friendly, concise, and professional."
    )
    
    full_prompt = f"System: {system_prompt}\nContext: {context}\nUser: {user_query}\nAssistant:"
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return None
    except Exception as e:
        print(f"Ollama Error: {e}")
        return None


# =====================================================
# CHAT ENDPOINT
# =====================================================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({
            "reply": "Invalid request",
            "type": "error"
        }), 400

    user_message = data["message"].strip()

    if not user_message:
        return jsonify({
            "reply": "Please enter a message.",
            "type": "error"
        })

    # ---------- GET MEMORY ----------
    memory = get_memory()

    # ---------- MATCH ----------
    result = find_best_match(user_message, memory)

    response = {
        "type": result["type"],
        "confidence": result.get("confidence", 0.0),
    }

    # ---------- GENERATE CONVERSATIONAL RESPONSE ----------
    context = result.get("context", "General university help")
    
    # Use Ollama for a conversational touch
    ollama_reply = call_ollama(user_message, context)
    
    if ollama_reply:
        response["reply"] = ollama_reply
        response["type"] = "ollama"
    else:
        # Fallback to hardcoded logic if Ollama is unavailable
        if result["type"] == "clarify":
            response["reply"] = "Did you mean one of these?"
            response["suggestions"] = result.get("suggestions", [])
        elif result.get("answer"):
            response["reply"] = result["answer"]
        else:
            response["reply"] = (
                "Sorry, I couldn't find a specific answer for that. "
                "Please try asking about Moodle, IMS, or registration."
            )

    # ---------- UPDATE MEMORY ----------
    update_memory({
        "intent": result.get("intent", "unknown"),
        "answer": response["reply"],
        "confidence": result.get("confidence", 0.0),
        "answer_id": result.get("answer_id", None),
    })

    return jsonify(response)


# =====================================================
# CLEAR MEMORY (OPTIONAL ENDPOINT)
# =====================================================
@app.route("/reset", methods=["POST"])
def reset_chat():
    clear_memory()
    return jsonify({"message": "Chat memory cleared"})


# =====================================================
# HEALTH CHECK
# =====================================================
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "DriveSafe Chatbot Backend Running"})


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
