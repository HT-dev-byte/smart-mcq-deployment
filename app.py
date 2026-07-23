import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# Configuration
# ==========================
REPO_ID = "HT-Dev-Byte/smart-mcq-cnn"
MAX_LEN = 256
LABELS = ["A", "B", "C", "D", "E"]

# ==========================
# Load model and tokenizer
# ==========================
@st.cache_resource
def load_resources():
    token = st.secrets["HF_TOKEN"]

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="multikernel_cnn.keras",
        token=token
    )

    tokenizer_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="tokenizer.pkl",
        token=token
    )

    model = tf.keras.models.load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_resources()

# ==========================
# Prediction Function
# ==========================
def predict_answer(prompt, option_a, option_b, option_c, option_d, option_e):

    text = (
    f"Question: {prompt}\n"
    f"A: {option_a}\n"
    f"B: {option_b}\n"
    f"C: {option_c}\n"
    f"D: {option_d}\n"
    f"E: {option_e}"
)

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probs = model.predict(padded, verbose=0)[0]

    return probs


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Smart MCQ Solver")
st.write("Predict the correct answer using the trained Multi-Kernel CNN model.")

prompt = st.text_area("Question")

option_a = st.text_input("Option A")
option_b = st.text_input("Option B")
option_c = st.text_input("Option C")
option_d = st.text_input("Option D")
option_e = st.text_input("Option E")

if st.button("Predict"):

    if not all([prompt, option_a, option_b, option_c, option_d, option_e]):
        st.warning("Please fill in all fields.")

    else:

        probs = predict_answer(
            prompt,
            option_a,
            option_b,
            option_c,
            option_d,
            option_e
        )

        predicted = np.argmax(probs)

        st.success(f"Predicted Answer: {LABELS[predicted]}")

        st.subheader("Confidence Scores")

        top3 = np.argsort(probs)[::-1][:3]

        for idx in top3:
            st.write(f"**{LABELS[idx]}** : {probs[idx]:.4f}")
            st.progress(float(probs[idx]))
