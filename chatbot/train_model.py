import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pickle
import re

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
punctuation = string.punctuation
lemmatizer = WordNetLemmatizer()

# Enhanced text preprocessing function
def preprocess_text(text):
    # Remove punctuation and lower the text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split words and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join back the words into a single string
    return ' '.join(words)

# Load the FAQ dataset
df = pd.read_csv('faq_data.csv')

# Preprocess the questions
df['processed_question'] = df['question'].apply(preprocess_text)

# Tokenize the questions
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # Adding out-of-vocabulary token handling
tokenizer.fit_on_texts(df['processed_question'].values)
sequences = tokenizer.texts_to_sequences(df['processed_question'].values)
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Map answers to numeric labels
answer_labels = {answer: idx for idx, answer in enumerate(df['answer'].unique())}
y_train = df['answer'].map(answer_labels).values

# Define the improved model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_sequence_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),  # Dropout to prevent overfitting
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(answer_labels), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Add EarlyStopping and ModelCheckpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=69, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_faq_model.keras', save_best_only=True)

# Split into training and validation sets (e.g., 80% train, 20% validation)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y_train, test_size=0.2, random_state=42)

# Train the model with validation
model.fit(X_train, y_train, epochs=70, batch_size=1, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# Save the final model and necessary files
model.save('final_faq_model.keras')

# Save the tokenizer and label mappings for later use
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('answer_labels.pkl', 'wb') as handle:
    pickle.dump(answer_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and necessary files saved!")