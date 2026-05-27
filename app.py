import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="E-Commerce NLP Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #08111f 0%,
            #102a43 45%,
            #1b4965 100%
        );
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    h1 {
        text-align: center;
        color: #38F8D4 !important;
        font-size: 58px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 18px rgba(56,248,212,0.45);
    }

    h2, h3 {
        color: #38F8D4 !important;
        font-weight: 800 !important;
    }

    p, li, label {
        color: #E2E8F0 !important;
        font-size: 17px !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #020617 0%,
            #0F172A 100%
        );
        border-right: 1px solid rgba(56,248,212,0.25);
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.06);
        padding: 12px;
        border-radius: 14px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    div[role="radiogroup"] label:hover {
        background: rgba(56,248,212,0.15);
        border: 1px solid rgba(56,248,212,0.55);
    }

    .stButton>button {
        background: linear-gradient(
            90deg,
            #38F8D4,
            #38BDF8
        );
        color: #020617 !important;
        border: none;
        border-radius: 16px;
        padding: 0.9rem;
        font-size: 19px;
        font-weight: 900;
        width: 100%;
        box-shadow: 0 0 20px rgba(56,248,212,0.35);
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(56,248,212,0.65);
    }

    textarea {
        background-color: white !important;
        color: black !important;
        border-radius: 14px !important;
        border: 3px solid #38F8D4 !important;
        font-size: 18px !important;
    }

    textarea::placeholder {
        color: gray !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.16);
        padding: 18px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }

    .positive-card {
        background: linear-gradient(
            90deg,
            #00C853,
            #64DD17
        );
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        color: white;
        box-shadow: 0 0 25px rgba(0,255,0,0.45);
    }

    .negative-card {
        background: linear-gradient(
            90deg,
            #D50000,
            #FF1744
        );
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        color: white;
        box-shadow: 0 0 25px rgba(255,0,0,0.45);
    }

    .neutral-card {
        background: linear-gradient(
            90deg,
            #2962FF,
            #00B0FF
        );
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        color: white;
        box-shadow: 0 0 25px rgba(0,150,255,0.45);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <h1>🛒 E-Commerce NLP Sentiment Analyzer</h1>

    <p style='text-align:center;
              font-size:22px;
              color:#E2E8F0;'>

    AI-Powered Customer Review Sentiment Analysis
    using NLP + Machine Learning 🚀

    </p>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DOWNLOAD NLTK
# ============================================================

@st.cache_resource
def download_nltk():

    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")

download_nltk()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "final_nlp_sentiment_eda_fe_dataset.csv"
    )

    return df

df = load_data()

# ============================================================
# LOAD MODEL + TFIDF
# ============================================================

@st.cache_resource
def load_pickle_files():

    with open(
        "best_sentiment_model.pkl",
        "rb"
    ) as model_file:

        model = pickle.load(model_file)

    with open(
        "tfidf_vectorizer.pkl",
        "rb"
    ) as tfidf_file:

        tfidf = pickle.load(tfidf_file)

    return model, tfidf

model, tfidf = load_pickle_files()

# ============================================================
# TEXT CLEANING
# ============================================================

stop_words = set(
    stopwords.words("english")
)

ps = PorterStemmer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    words = word_tokenize(text)

    words = [

        word for word in words

        if word not in stop_words

    ]

    words = [

        ps.stem(word)

        for word in words

    ]

    return " ".join(words)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🧭 Navigation"
)

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Home",
        "📊 EDA Dashboard",
        "🔮 Predict Sentiment"

    ]

)

# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.header("📌 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reviews",
        df.shape[0]
    )

    col2.metric(
        "Total Columns",
        df.shape[1]
    )

    col3.metric(
        "ML Model",
        type(model).__name__
    )

    st.markdown("---")

    st.subheader("📖 About Project")

    st.write(
        """
        This NLP application predicts customer review sentiment
        using Machine Learning and TF-IDF Vectorization.

        ### Features

        ✅ Sentiment Prediction  
        ✅ NLP Text Cleaning  
        ✅ Interactive EDA Dashboard  
        ✅ Word Cloud Visualization  
        ✅ TF-IDF + ML Model  
        ✅ Streamlit Deployment  

        """
    )

# ============================================================
# EDA PAGE
# ============================================================

elif page == "📊 EDA Dashboard":

    st.header("📊 Exploratory Data Analysis")

    # ========================================================
    # SENTIMENT DISTRIBUTION
    # ========================================================

    if "sentiment" in df.columns:

        st.subheader("Sentiment Distribution")

        fig, ax = plt.subplots(
            figsize=(7,5)
        )

        sns.countplot(
            data=df,
            x="sentiment",
            ax=ax
        )

        st.pyplot(fig)

    # ========================================================
    # RATING DISTRIBUTION
    # ========================================================

    if "rating" in df.columns:

        st.subheader("Rating Distribution")

        fig, ax = plt.subplots(
            figsize=(7,5)
        )

        sns.countplot(
            data=df,
            x="rating",
            ax=ax
        )

        st.pyplot(fig)

    # ========================================================
    # WORD COUNT
    # ========================================================

    if "word_count" not in df.columns:

        df["word_count"] = df["review"].apply(
            lambda x: len(str(x).split())
        )

    st.subheader(
        "Review Length Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    sns.histplot(
        df["word_count"],
        bins=50,
        ax=ax
    )

    st.pyplot(fig)

    # ========================================================
    # WORD CLOUD
    # ========================================================

    st.subheader("Word Cloud")

    try:

        clean_reviews = (

            df["clean_review"]

            .fillna("")

            .astype(str)

        )

        clean_reviews = clean_reviews[
            clean_reviews.str.strip() != ""
        ]

        all_words = " ".join(
            clean_reviews.tolist()
        )

        if len(all_words.strip()) > 0:

            wordcloud = WordCloud(

                width=1200,
                height=500,
                background_color="white"

            ).generate(all_words)

            fig, ax = plt.subplots(
                figsize=(15,7)
            )

            ax.imshow(wordcloud)

            ax.axis("off")

            st.pyplot(fig)

        else:

            st.warning(
                "No valid text available for Word Cloud."
            )

    except Exception as e:

        st.error(
            "Word Cloud generation failed."
        )

        st.write(e)

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔮 Predict Sentiment":

    st.header(
        "🔮 Predict Customer Review Sentiment"
    )

    user_review = st.text_area(

        "Enter Customer Review",

        height=200,

        placeholder="Example: Product quality is amazing and delivery was very fast..."

    )

    if st.button("🚀 Predict Sentiment"):

        if user_review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            cleaned_review = clean_text(
                user_review
            )

            vector_input = tfidf.transform(
                [cleaned_review]
            )

            prediction = model.predict(
                vector_input
            )[0]

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == "positive":

                st.markdown(

                    '<div class="positive-card">😊 POSITIVE REVIEW</div>',

                    unsafe_allow_html=True

                )

            elif prediction == "negative":

                st.markdown(

                    '<div class="negative-card">😡 NEGATIVE REVIEW</div>',

                    unsafe_allow_html=True

                )

            else:

                st.markdown(

                    '<div class="neutral-card">😐 NEUTRAL REVIEW</div>',

                    unsafe_allow_html=True

                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("🧹 Cleaned Review")

            st.code(cleaned_review)
