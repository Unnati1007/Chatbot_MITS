# MITS AI Assistant 

A premium, full-stack AI chatbot designed specifically for students and faculty of **Madhav Institute of Technology & Science (MITS)**. This assistant helps with university-related queries like Moodle, IMS, Registration, and Academics using a local LLM for privacy and speed.

---

## 🌟 Key Features

- **Full-Screen Interactive UI**: A modern, immersive chat experience that fills the entire viewport.
- **Light & Dark Theme**: Built-in theme toggle with persistent storage (remembers your choice).
- **Real-time AI Streaming**: Watch the AI "think" and generate responses word-by-word.
- **Thinking Indicator**: Visual animated feedback while the model processes your query.
- **Voice Recognition**: Support for voice input optimized for Indian English.
- **Copy to Clipboard**: One-click button to copy AI responses.
- **Admin Analytics Dashboard**: Monitor student feedback, top queries, and AI learning performance.
- **Rule-Based & AI Hybrid**: Instantly answers common FAQs while using Llama 3.2 for complex queries.

---

## 🛠️ Technology Stack

- **Frontend**: React.js, Vite, Lucide React, Framer Motion, CSS3 (Vanilla)
- **Backend**: Python, Flask, Flask-CORS
- **Database**: SQLite (for feedback and RL stats)
- **AI Engine**: Ollama (Running Llama 3.2)
- **NLP**: Sentence-Transformers for semantic matching

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **Ollama** (Download from [ollama.com](https://ollama.com))

### 2. Model Installation
The assistant uses the **Llama 3.2** model. Once Ollama is installed, run the following command in your terminal:
```bash
ollama pull llama3.2
```

### 3. Backend Setup
1. Navigate to the `backend` directory:
```bash
cd backend
```
2. Install dependencies:
```bash
pip install flask flask-cors requests sentence-transformers
```
3. Run the Flask server:
```bash
python app.py
```
*The backend will run on `http://127.0.0.1:5000`*

### 4. Frontend Setup
1. Navigate to the `frontend` directory:
```bash
cd frontend
```
2. Install dependencies:
```bash
npm install
```
3. Run the development server:
```bash
npm run dev
```
*The frontend will run on `http://localhost:5173`*

---

## 📊 Admin Dashboard
To access the analytics dashboard, navigate to:
`http://localhost:5173/admin`

From here, you can monitor:
- Student feedback and ratings.
- AI Learning impact and rewards.
- FAQ performance metrics.

---

## 📝 License
This project is developed for institutional use at Madhav Institute of Technology & Science.

---

**Developed with ❤️ by Unnati Jadon**
