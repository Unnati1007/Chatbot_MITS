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
MODEL_NAME = "llama3.2"

def check_ollama_status():
    """Verify if Ollama is running and Llama 3.2 is available"""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            if any("llama3.2" in m.lower() for m in models):
                print(f"✅ Ollama is Online. Found models: {', '.join(models)}")
                return True
            else:
                print(f"⚠️ Ollama is running but '{MODEL_NAME}' was not found. Please run 'ollama pull llama3.2'.")
        else:
            print(f"❌ Ollama server responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not connect to Ollama server at http://127.0.0.1:11434. Make sure the Ollama app is open.")
    return False

def call_ollama(user_query, context):
    """Call local Ollama server for conversational response"""
    system_prompt = (
        "You are the 'MITS AI Assistant', a premium university companion for Madhav Institute of Technology & Science. "
        "INSTRUCTIONS:\n"
        "1. Use the provided Context to answer the user's question accurately.\n"
        "2. Format your response using Markdown (bolding, lists, etc.) to make it highly readable.\n"
        "3. **CRITICAL:** Only provide URLs that are explicitly mentioned in the provided Context. DO NOT invent or assume any website addresses.\n"
        "4. If the context is about a technical issue (Moodle, IMS), provide step-by-step instructions.\n"
        "5. **STRICT POLICY:** If the user's question is completely unrelated to university life, academics, or MITS (e.g., cooking, movies, sports), "
        "politely inform them that you are specifically designed to assist with MITS-related queries and cannot provide general life advice or recipes."
    )
    
    full_prompt = f"System: {system_prompt}\n\nContext: {context}\n\nUser: {user_query}\n\nAssistant:"
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"DEBUG: Ollama error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"DEBUG: Ollama connection failed: {e}")
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
    elif result.get("answer"):
        # FALLBACK: If Ollama fails, use the direct database answer
        response["reply"] = result["answer"]
        response["type"] = result["type"]
    else:
        # ABSOLUTE FALLBACK: No match and no Ollama
        if result["type"] == "clarify":
            response["reply"] = "I'm not quite sure, did you mean one of these?"
            response["suggestions"] = result.get("suggestions", [])
        else:
            response["reply"] = (
                "I'm sorry, I couldn't find a specific answer in my knowledge base. "
                "Please try rephrasing your question or check the MITS website."
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
    # use_reloader=False prevents double-loading of the AI model
    print("🚀 MITS AI Backend is starting...")
    
    # Check if local Ollama is ready
    check_ollama_status()
    
    app.run(debug=True, use_reloader=False, port=5000)
