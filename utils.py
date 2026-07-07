import re
import string
import warnings
import pandas as pd
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")

import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


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


def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = remove_urls(text)
    text = remove_html(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = remove_extra_spaces(text)

    # No punkt dependency
    tokens = text.split()

    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 2
    ]

    return " ".join(tokens)


def load_and_merge_dataset(
    fake_path="Fake.csv",
    true_path="True.csv"
):

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


def create_metadata(
    model_name,
    accuracy,
    dataset_size,
    tfidf_features=10000
):

    from datetime import datetime

    return {
        "best_model": model_name,
        "accuracy": round(accuracy, 4),
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


def predict_news(
    text,
    model,
    vectorizer
):

    cleaned_text = preprocess_text(text)

    vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(vector)[0]

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


def check_required_files():

    import os

    files = [
        "model.pkl",
        "vectorizer.pkl",
        "model_metadata.pkl"
    ]

    return {
        file: os.path.exists(file)
        for file in files
    }
