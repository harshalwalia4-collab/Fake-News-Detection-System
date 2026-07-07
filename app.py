# Streamlit app template for Fake News Detection
import streamlit as st
import pandas as pd
import joblib
import os

from utils import (
    preprocess_text,
    predict_news,
    check_required_files
)

st.set_page_config(
    page_title="Fake News Detection",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color:#0E1117;
}
h1,h2,h3{
color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detection System")

status = check_required_files()

if not all(status.values()):
    st.error(
        "Model files missing. Run train.py first."
    )
    st.stop()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "News Detection",
        "Analytics",
        "About"
    ]
)

# HOME PAGE

if menu == "Home":

    st.header("Project Overview")

    st.write("""
    NLP-based Fake News Detection System
    using TF-IDF and Machine Learning.
    """)

    if os.path.exists(
        "combined_news_dataset.csv"
    ):
        df = pd.read_csv(
            "combined_news_dataset.csv"
        )

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Rows",
            len(df)
        )

        c2.metric(
            "Columns",
            len(df.columns)
        )

        c3.metric(
            "Fake News",
            len(df[df["label"] == 0])
        )

# DETECTION PAGE

elif menu == "News Detection":

    st.header("Detect News")

    news = st.text_area(
        "Paste News Article",
        height=250
    )

    if st.button("Predict"):

        if not news.strip():
            st.warning(
                "Please enter article text."
            )

        else:

            result = predict_news(
                news,
                model,
                vectorizer
            )

            if result["prediction"] == 1:

                st.success(
                    f"{result['label']}"
                )

            else:

                st.error(
                    f"{result['label']}"
                )

            st.metric(
                "Confidence",
                f"{result['confidence']}%"
            )

# ANALYTICS PAGE

elif menu == "Analytics":

    st.header("Analytics Dashboard")

    if os.path.exists(
        "combined_news_dataset.csv"
    ):

        df = pd.read_csv(
            "combined_news_dataset.csv"
        )

        st.subheader(
            "Class Distribution"
        )

        chart_data = (
            df["label"]
            .value_counts()
        )

        st.bar_chart(chart_data)

# ABOUT PAGE

elif menu == "About":

    st.header("About Project")

    st.write("""
    Technologies Used:

    - Python
    - NLP
    - TF-IDF
    - Scikit-Learn
    - Streamlit
    - Plotly
    - Pandas
    """)

    st.write("Developer: ")# app.py

import streamlit as st
import pandas as pd
import joblib
import os

from utils import (
    predict_news,
    check_required_files
)

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)

DATASET_FILE = "combined_news_dataset.csv.zip"

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.kpi-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.title("📰 Fake News Detection System")

# ---------------- MODEL CHECK ---------------- #

status = check_required_files()

if not all(status.values()):
    st.error(
        "Required files missing. Upload model.pkl, vectorizer.pkl and model_metadata.pkl"
    )
    st.stop()

try:

    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

except Exception as e:

    st.error(f"Model Loading Error: {e}")
    st.stop()

# ---------------- SIDEBAR ---------------- #

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "News Detection",
        "Analytics",
        "About"
    ]
)

# =====================================================
# HOME PAGE
# =====================================================

if menu == "Home":

    st.header("🏠 Project Overview")

    st.write("""
    Fake News Detection System using:

    - NLP
    - Text Cleaning
    - TF-IDF Vectorization
    - Machine Learning
    - Streamlit Dashboard
    """)

    if os.path.exists(DATASET_FILE):

        try:

            df = pd.read_csv(
                DATASET_FILE,
                compression="zip"
            )

            st.subheader("Dataset Preview")

            st.dataframe(df.head())

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Rows",
                f"{len(df):,}"
            )

            col2.metric(
                "Columns",
                len(df.columns)
            )

            col3.metric(
                "Fake News",
                len(
                    df[df["label"] == 0]
                )
            )

            col4.metric(
                "Real News",
                len(
                    df[df["label"] == 1]
                )
            )

        except Exception as e:

            st.error(
                f"Dataset Error: {e}"
            )

    else:

        st.warning(
            "combined_news_dataset.csv.zip not found"
        )

# =====================================================
# NEWS DETECTION
# =====================================================

elif menu == "News Detection":

    st.header("🔍 Fake News Detection")

    news_text = st.text_area(
        "Paste News Article",
        height=300
    )

    if st.button("Predict"):

        if not news_text.strip():

            st.warning(
                "Please enter news article."
            )

        else:

            try:

                result = predict_news(
                    news_text,
                    model,
                    vectorizer
                )

                if result["prediction"] == 1:

                    st.success(
                        f"Prediction: {result['label']}"
                    )

                else:

                    st.error(
                        f"Prediction: {result['label']}"
                    )

                st.metric(
                    "Confidence",
                    f"{result['confidence']}%"
                )

            except Exception as e:

                st.error(
                    f"Prediction Error: {e}"
                )

# =====================================================
# ANALYTICS
# =====================================================

elif menu == "Analytics":

    st.header("📊 Analytics Dashboard")

    if os.path.exists(DATASET_FILE):

        try:

            df = pd.read_csv(
                DATASET_FILE,
                compression="zip"
            )

            st.subheader(
                "Class Distribution"
            )

            class_counts = (
                df["label"]
                .value_counts()
            )

            st.bar_chart(
                class_counts
            )

            st.subheader(
                "Dataset Statistics"
            )

            st.write(
                df.describe(
                    include="all"
                )
            )

        except Exception as e:

            st.error(
                f"Analytics Error: {e}"
            )

    else:

        st.warning(
            "Dataset file missing."
        )

# =====================================================
# ABOUT
# =====================================================

elif menu == "About":

    st.header("ℹ About Project")

    st.write("""
    ### Fake News Detection System

    This project uses:

    - NLP
    - TF-IDF
    - Machine Learning
    - Streamlit

    Models Trained:

    - Logistic Regression
    - Naive Bayes
    - Random Forest
    - Linear SVM
    - XGBoost

    Output:

    - REAL NEWS
    - FAKE NEWS
    """)

    st.write("### Developer")
    st.write("Harshal")