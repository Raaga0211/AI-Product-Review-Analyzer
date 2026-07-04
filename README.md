# 🛍️ AI Product Review Analyzer

An AI-powered web application that analyzes product reviews using Machine Learning and Natural Language Processing (NLP). The system predicts whether a review is **Positive**, **Neutral**, or **Negative** and enhances sentiment analysis using emoji-based sentiment scoring.

---

## 📌 Project Title

**Multi-Stage Sentiment Analysis of Twitter Product Reviews using Machine Learning with Emoji-Based Sentiment Enhancement**

---

## 🚀 Features

- Real-time Product Review Analysis
- Sentiment Classification (Positive, Neutral, Negative)
- Emoji-Based Sentiment Enhancement
- Intelligent Product Recommendation
- Interactive Streamlit Web Application
- Machine Learning Model Comparison
- User-Friendly Interface

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- SciPy
- Joblib
- TF-IDF (NLP)
- Logistic Regression

---

## 📂 Dataset

- **Dataset:** Women Clothing Reviews with Emojis
- **Size:** 19,675 Reviews
- **Format:** CSV

---

## 🤖 Machine Learning Models

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **98.81%** |
| Support Vector Machine (SVM) | **96.59%** |
| Multinomial Naive Bayes | **79.85%** |

**Final Model Selected:** Logistic Regression

---

## 📊 Workflow

1. User enters a product review.
2. Text is preprocessed using NLP techniques.
3. TF-IDF converts the text into numerical features.
4. Logistic Regression predicts the sentiment.
5. Emoji sentiment score is calculated.
6. The application displays:
   - Predicted Sentiment
   - Emoji Score
   - Product Recommendation

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Product-Review-Analyzer.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
AI-Product-Review-Analyzer/
│
├── app.py
├── train_model.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── women_reviews_with_emojis.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👨‍💻 Team Members

- Puligilla Sri Raaga
- G. Upasana
- Shreyas Menon

**Guide:** Mr. P. Ravindra

---

## 📄 License

This project was developed as a mini-project for academic purposes.