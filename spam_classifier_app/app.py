import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


@st.cache_resource
def load_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")
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

user_input = st.text_area("Enter SMS here")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = re.sub(r'[^a-z0-9\s]', '', user_input.lower())
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.error("This is a SPAM message!")
        else:
            st.success("This is a HAM message!")