from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained model, tokenizer, and label mappings
model = tf.keras.models.load_model('best_faq_model.keras')
# model = tf.keras.models.load_model('faq_model.keras')  # Updated to use .keras file
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('answer_labels.pkl', 'rb') as handle:
    answer_labels = pickle.load(handle)

reverse_answer_labels = {v: k for k, v in answer_labels.items()}

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

app = Flask(__name__)

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # This serves the HTML file

# API endpoint for answering queries
@app.route('/get_answer', methods=['POST'])
def get_answer():
    question = request.json['question']
    processed_question = preprocess_text(question)
    seq = tokenizer.texts_to_sequences([processed_question])
    padded_seq = pad_sequences(seq, maxlen=model.input_shape[1])
    prediction = model.predict(padded_seq)
    predicted_label = prediction.argmax(axis=1)[0]
    answer = reverse_answer_labels[predicted_label]
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
