import streamlit as st
from model import train_model
from preprocess import clean_text

@st.cache_resource
def load_model():
    return train_model("dataset.csv")

model, vectorizer = load_model()

st.title("📧 Spam Email Detector")

user_input = st.text_area("Enter email text:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        text = clean_text(user_input)
        vec = vectorizer.transform([text])
        result = model.predict(vec)

        if result[0] == 1:
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM")