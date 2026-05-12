import pandas as pd
import json
from collections import defaultdict
from preprocess import normalize

# ---------------- PARAPHRASE TEMPLATES ----------------
TEMPLATES = [
    "{q}",
    "how to {q}",
    "i have an issue with {q}",
    "problem related to {q}",
    "can you help with {q}",
    "i am facing {q}",
    "need help regarding {q}"
]

def expand_question(question: str) -> set:
    variations = set()
    for t in TEMPLATES:
        variations.add(t.format(q=question))
    return variations


# ---------------- LOAD CSV SAFELY ----------------
CSV_PATH = "../data/faq_data.csv"
OUTPUT_PATH = "../data/faq_canonical.json"

df = pd.read_csv(CSV_PATH)

faq_groups = defaultdict(set)

for _, row in df.iterrows():
    question_raw = row.get("question")
    answer_raw = row.get("answer")

    # Skip empty or NaN rows
    if pd.isna(question_raw) or pd.isna(answer_raw):
        continue

    question = normalize(str(question_raw))
    answer = str(answer_raw).strip()

    if question and answer:
        faq_groups[answer].add(question)

# ---------------- BUILD CANONICAL DATA ----------------
faq_canonical = []

for idx, (answer, questions) in enumerate(faq_groups.items(), start=1):
    expanded_questions = set()

    for q in questions:
        expanded_questions.update(expand_question(q))

    faq_canonical.append({
        "intent": f"intent_{idx}",
        "questions": sorted(expanded_questions),
        "answer": answer
    })

# ---------------- SAVE JSON ----------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(faq_canonical, f, indent=2, ensure_ascii=False)

print(f"✅ faq_canonical.json generated successfully")
print(f"📌 Total intents: {len(faq_canonical)}")
print(f"📌 Total questions: {sum(len(i['questions']) for i in faq_canonical)}")
