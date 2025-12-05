import streamlit as st
import joblib

# Load Model & Vectorizer
model = joblib.load("spam_bnb_model.pkl")  # BernoulliNB
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page Config
st.set_page_config(page_title="Spam Detector App", page_icon="📩", layout="centered")

# Custom CSS
st.markdown("""
<style>
.stTextInput>div>div>input {
    border-radius: 10px;
}
.result-box {
    padding: 18px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 18px;
    margin-top: 15px;
}
.spam {
    background-color: #ffcccc;
    color: #b30000;
    border-left: 8px solid #b30000;
}
.ham {
    background-color: #d1ffd1;
    color: #006600;
    border-left: 8px solid #006600;
}
.prob {
    font-size: 14px;
    color: #333333;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📩 SMS Spam Detector")
st.write("Enter a message below and detect whether it's Spam or Ham.")

# Input Text
user_input = st.text_area("✍️ Write your message:", height=120)

# Predict Button
if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        # Transform input text
        input_vector = vectorizer.transform([user_input])

        # Spam probability
        prob_spam = model.predict_proba(input_vector)[0][1]  # Probability of Spam
        threshold = 0.4  # You can tune this (0.4–0.5)
        prediction = 1 if prob_spam > threshold else 0

        # Display Result
        if prediction == 1:
            st.markdown(
                f"<div class='result-box spam'>🚫 Spam Message Detected!</div>"
                f"<div class='prob'>Spam Probability: {prob_spam:.2f}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box ham'>✔️ Safe! This is a Ham Message.</div>"
                f"<div class='prob'>Spam Probability: {prob_spam:.2f}</div>",
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.caption("🔎 Powered by Bernoulli Naive Bayes + TF-IDF | Built with ❤️ using Streamlit")



