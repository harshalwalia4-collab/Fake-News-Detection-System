import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils import (
    load_and_merge_dataset,
    preprocess_text,
    create_metadata
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }


print("Loading dataset...")

df = load_and_merge_dataset()

df["text"] = df["text"].fillna("")
df["clean_text"] = df["text"].apply(preprocess_text)

X = df["clean_text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Multinomial NB":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Linear SVM":
        LinearSVC()
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        eval_metric="logloss"
    )

best_model = None
best_score = 0
best_name = ""

results = []

for name, model in models.items():

    print(f"Training {name}")

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    metrics["model"] = name

    results.append(metrics)

    if metrics["accuracy"] > best_score:
        best_score = metrics["accuracy"]
        best_model = model
        best_name = name

results_df = pd.DataFrame(results)

print(results_df)

joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

metadata = create_metadata(
    best_name,
    best_score,
    len(df)
)

joblib.dump(
    metadata,
    "model_metadata.pkl"
)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print(f"\nBest Model: {best_name}")
print(f"Accuracy: {best_score:.4f}")
print("\nSaved:")
print("model.pkl")
print("vectorizer.pkl")
print("model_metadata.pkl")