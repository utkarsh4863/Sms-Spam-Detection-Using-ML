# 📩 SMS Spam Detector

A machine learning-based web application to detect whether a text message is **Spam** or **Ham (Safe)** using **Bernoulli Naive Bayes** and **TF-IDF Vectorization**. Built with **Streamlit** for an interactive UI.

---

## 🔹 Features

- 🚀 Detects spam messages in real-time.
- ✨ Modern and clean UI using Streamlit.
- 🔔 Highlights spam and ham messages with different styles.
- 🌐 Easy to deploy and use on any browser.

---

## 🛠 Technology Stack

- **Python 3.10+**
- **Machine Learning:** Bernoulli Naive Bayes
- **Text Vectorization:** TF-IDF (scikit-learn)
- **Web App Framework:** Streamlit
- **Libraries:** pandas, numpy, nltk, joblib, matplotlib, wordcloud

---

## 💾 Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/<your-username>/sms-spam-detection.git
    cd sms-spam-detection
    ```

2. **Create a virtual environment and activate:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Run the App Locally

```bash
streamlit run app.py
```
Open [https://sms-spam-detection-using-ml-6p2ppkzigwo9pdmrqig93m.streamlit.app/) in your browser to use the app.

---

## 🗂 File Structure

```
sms-spam-detection/
│
├── app.py                    # Streamlit web app
├── spam_bernoulli_model.pkl  # Trained BernoulliNB model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 How It Works

1. User enters a message in the input box.
2. The app vectorizes the message using the saved TF-IDF vectorizer.
3. Bernoulli Naive Bayes model predicts Spam or Ham.
4. Result is displayed with styled feedback:

   - 🚫 **SPAM Message Detected!**
   - ✔️ **Safe! This is a HAM Message.**

---

## ⚖️ Model Performance

**BernoulliNB Metrics (Test Data):**
- **Accuracy:** 0.9845
- **Precision:** 1.000
- **Recall:** 0.8779
- **F1 Score:** 0.9350

> High precision ensures almost no false positives (safe messages misclassified as spam).

---

## ⚙️ Notes

- Dataset preprocessing included lowercasing, removing special characters, stopwords removal, and stemming.
- Imbalanced dataset handled during model training with techniques like oversampling (SMOTE).
- Model and vectorizer saved using pickle for deployment.

---

## 💻 License

This project is open-source and available under the MIT License.

---

## ❤️ Acknowledgements

- Scikit-learn
- Streamlit
- NLTK
