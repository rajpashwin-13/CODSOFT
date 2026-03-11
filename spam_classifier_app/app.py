from importlib.resources import path
import streamlit as st
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "spam.csv")
    df = pd.read_csv(path, encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    df['message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = load_model()

st.title("SMS Spam Classifier")
st.write("Enter a message to check whether it is Spam or Ham")

st.markdown("Try Example Messages")
example = st.selectbox(
    "Choose an example",
    [
        "Congratulations! You won a free lottery ticket",
        "Your OTP is required to verify bank account",
        "Hey bro are we meeting today?",
        "Free entry in a prize draw claim now"
    ]
)

user_input = st.text_area("Enter SMS here", value=example)
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = re.sub(r'[^a-z0-9\s]', '', user_input.lower())
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        prob = model.predict_proba(vectorized)

        if prediction[0] == 1:
            st.error("This is a SPAM message!")
            st.write(f"Spam Probability: {prob[0][1]*100:.2f}%")

        else:
            st.success("This is a HAM message!")
            st.write(f"Ham Probability: {prob[0][0]*100:.2f}%")