import streamlit as st
import re
import nltk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pickle

from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


# Load model and tokenizer
model = tf.keras.models.load_model("hate_speech_model.h5", custom_objects={"F1Score": F1Score()})

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Download required nltk resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add("rt")

# Preprocessing functions
def remove_entity(raw_text):
    return re.sub(r"&[^\s;]+;", "", raw_text)

def change_user(raw_text):
    return re.sub(r"@([^ ]+)", "user", raw_text)

def remove_url(raw_text):
    return re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', raw_text)

def remove_noise_symbols(raw_text):
    return raw_text.replace('"', '').replace("'", '').replace("!", '').replace("`", '').replace("..", '')

def remove_stopwords(raw_text):
    tokens = nltk.word_tokenize(raw_text)
    return " ".join([word for word in tokens if word.lower() not in stop_words])

def preprocess_single_tweet(tweet):
    tweet = change_user(tweet)
    tweet = remove_entity(tweet)
    tweet = remove_url(tweet)
    tweet = remove_noise_symbols(tweet)
    tweet = remove_stopwords(tweet)
    return tweet

# Streamlit UI
st.title("🔥 Hate Speech Detection Module")
tweet = st.text_area("Enter a caption/comment to analyze:")

if st.button("Classify"):
    cleaned = preprocess_single_tweet(tweet)
    seq = tokenizer.texts_to_sequences([cleaned])
    max_len = 100  # Use the same max_length as training
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    class_id = np.argmax(pred)
    class_labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}

    st.markdown(f"### Prediction: `{class_labels[class_id]}`")
    st.markdown(f"#### Confidence: {(pred[0])* 100:.2f}%")
