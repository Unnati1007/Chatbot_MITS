from .preprocess import preprocess_for_rules

# =====================================================
# RULE DEFINITIONS
# =====================================================

GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening"
}

THANKS = {
    "thanks",
    "thank you",
    "thx",
    "thankyou"
}

CONFIRMATIONS = {
    "ok",
    "okay",
    "hmm",
    "fine",
    "alright",
    "again"
}

FAREWELLS = {
    "bye",
    "goodbye",
    "see you",
    "take care",
    "exit",
    "quit"
}

PASSWORD_KEYWORDS = {
    "forgot password",
    "lost password",
    "reset password",
    "password problem",
    "cannot login",
    "cant login"
}

REGISTRATION_KEYWORDS = {
    "registration",
    "not registered",
    "email not received",
    "ims registration",
    "registration email"
}


# =====================================================
# MAIN RULE CHECK FUNCTION
# =====================================================
def check_rules(user_query: str) -> dict | None:
    """
    Returns rule-based response if matched.
    Otherwise returns None.
    """

    if not user_query or not isinstance(user_query, str):
        return None

    text = preprocess_for_rules(user_query)

    # ---------- GREETINGS ----------
    if any(greet in text for greet in GREETINGS):
        return {
            "type": "rule",
            "intent": "greeting",
            "answer": "Hello! 👋 How can I help you with Moodle, IMS, or registration?",
            "confidence": 1.0
        }

    # ---------- THANK YOU ----------
    if any(t in text for t in THANKS):
        return {
            "type": "rule",
            "intent": "thanks",
            "answer": "Glad to help! 😊 Let me know if you have more questions.",
            "confidence": 1.0
        }

    # ---------- CONFIRMATION / FILLER ----------
    if any(c in text for c in CONFIRMATIONS):
        return {
            "type": "rule",
            "intent": "confirmation",
            "answer": "Okay 👍 Tell me what you need help with.",
            "confidence": 1.0
        }

    # ---------- FAREWELLS ----------
    if any(f in text for f in FAREWELLS):
        return {
            "type": "rule",
            "intent": "farewell",
            "answer": "Goodbye! 😊 Feel free to ask if you have more questions about Moodle or IMS later. Have a great day!",
            "confidence": 1.0
        }

    # ---------- PASSWORD ISSUES ----------
    if "password" in text and ("forgot" in text or "lost" in text):
        return {
            "type": "rule",
            "intent": "password_reset",
            "answer": (
                "You can reset your password using the **Forgot Password** option "
                "on the Moodle or IMS login page. Enter your university email ID "
                "and follow the instructions sent to your email."
            ),
            "confidence": 0.95
        }

    # ---------- REGISTRATION ISSUES ----------
    if any(k in text for k in REGISTRATION_KEYWORDS):
        return {
            "type": "rule",
            "intent": "registration_issue",
            "answer": (
                "If registration is not open yet, please keep checking the university "
                "website for notifications. Once registrations start, you will receive "
                "official updates."
            ),
            "confidence": 0.90
        }

    return None
