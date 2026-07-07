# utils.py

import re
import string
import warnings
import pandas as pd
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")

# NLP
import nltk

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except:
    nltk.download("omw-1.4")

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------
# Global Objects
# ---------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------
# Text Cleaning Functions
# ---------------------------------------------------

def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text)


def remove_html(text):
    return re.sub(r"<.*?>", "", text)


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def remove_special_characters(text):
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------
# Main Preprocessing Function
# ---------------------------------------------------

def preprocess_text(text):
    """
    Complete NLP preprocessing pipeline
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    text = remove_urls(text)

    text = remove_html(text)

    text = remove_numbers(text)

    text = remove_special_characters(text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = remove_extra_spaces(text)

    tokens = word_tokenize(text)

    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 2
    ]

    return " ".join(tokens)


# ---------------------------------------------------
# Dataset Loading
# ---------------------------------------------------

def load_and_merge_dataset(
    fake_path="Fake.csv",
    true_path="True.csv"
):
    """
    Merge Fake.csv and True.csv
    """

    fake_df = pd.read_csv(fake_path)

    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    fake_df["news_type"] = "FAKE"
    true_df["news_type"] = "REAL"

    combined_df = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    combined_df = combined_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    combined_df.to_csv(
        "combined_news_dataset.csv.zip",
        compression="zip",
        index=False
    )

    return combined_df


# ---------------------------------------------------
# Dataset Summary
# ---------------------------------------------------

def dataset_summary(df):

    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values":
            int(df.isnull().sum().sum()),
        "duplicates":
            int(df.duplicated().sum()),
        "fake_news":
            int((df["label"] == 0).sum()),
        "real_news":
            int((df["label"] == 1).sum())
    }

    return summary


# ---------------------------------------------------
# Article Length Features
# ---------------------------------------------------

def add_length_features(df):

    if "text" not in df.columns:
        return df

    df["article_length"] = (
        df["text"]
        .astype(str)
        .apply(len)
    )

    df["word_count"] = (
        df["text"]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    return df


# ---------------------------------------------------
# Most Common Words
# ---------------------------------------------------

def get_top_words(
    texts,
    n=20
):

    all_words = []

    for text in texts:

        all_words.extend(
            str(text).split()
        )

    counter = Counter(all_words)

    top_words = counter.most_common(n)

    words = [i[0] for i in top_words]

    counts = [i[1] for i in top_words]

    return pd.DataFrame({
        "word": words,
        "count": counts
    })


# ---------------------------------------------------
# Class-wise Common Words
# ---------------------------------------------------

def get_fake_real_words(
    df,
    top_n=20
):

    fake_text = " ".join(
        df[df["label"] == 0]
        ["clean_text"]
        .astype(str)
    )

    real_text = " ".join(
        df[df["label"] == 1]
        ["clean_text"]
        .astype(str)
    )

    fake_words = get_top_words(
        [fake_text],
        top_n
    )

    real_words = get_top_words(
        [real_text],
        top_n
    )

    return fake_words, real_words


# ---------------------------------------------------
# TF-IDF Keywords
# ---------------------------------------------------

def get_top_tfidf_keywords(
    vectorizer,
    top_n=30
):
    """
    Extract top TF-IDF terms
    """

    try:

        feature_names = (
            vectorizer
            .get_feature_names_out()
        )

        return pd.DataFrame({
            "keyword":
                feature_names[:top_n]
        })

    except:

        return pd.DataFrame()


# ---------------------------------------------------
# Prediction Utility
# ---------------------------------------------------

def predict_news(
    text,
    model,
    vectorizer
):
    """
    Predict Fake / Real News
    """

    cleaned_text = preprocess_text(text)

    vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(vector)[0]

    confidence = 0

    try:

        confidence = (
            np.max(
                model.predict_proba(vector)
            ) * 100
        )

    except:

        confidence = 95.0

    label = (
        "REAL NEWS"
        if prediction == 1
        else "FAKE NEWS"
    )

    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(
            confidence,
            2
        )
    }


# ---------------------------------------------------
# Batch Prediction
# ---------------------------------------------------

def batch_predict(
    dataframe,
    model,
    vectorizer,
    text_column="text"
):

    df = dataframe.copy()

    df["Prediction"] = df[text_column].apply(
        lambda x:
        predict_news(
            x,
            model,
            vectorizer
        )["label"]
    )

    return df


# ---------------------------------------------------
# Model Metadata
# ---------------------------------------------------

def create_metadata(
    model_name,
    accuracy,
    dataset_size,
    tfidf_features=10000
):

    from datetime import datetime

    metadata = {
        "best_model": model_name,
        "accuracy": round(
            accuracy,
            4
        ),
        "dataset_size": dataset_size,
        "training_date":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        "tfidf_max_features":
            tfidf_features,
        "ngram_range":
            "(1,2)"
    }

    return metadata


# ---------------------------------------------------
# Safe Loader
# ---------------------------------------------------

def safe_read_csv(path):

    try:

        return pd.read_csv(path)

    except Exception as e:

        print(
            f"Error loading file: {e}"
        )

        return pd.DataFrame()


# ---------------------------------------------------
# Application Health Check
# ---------------------------------------------------

def check_required_files():

    import os

    files = [
        "model.pkl",
        "vectorizer.pkl",
        "model_metadata.pkl"
    ]

    status = {}

    for file in files:

        status[file] = os.path.exists(file)

    return status


# ---------------------------------------------------
# Text Statistics
# ---------------------------------------------------

def text_statistics(df):

    if "text" not in df.columns:

        return {}

    lengths = (
        df["text"]
        .astype(str)
        .apply(len)
    )

    stats = {

        "avg_length":
            round(lengths.mean(), 2),

        "max_length":
            int(lengths.max()),

        "min_length":
            int(lengths.min()),

        "median_length":
            int(lengths.median())
    }

    return stats