import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import clean_text

def train_model(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Rename columns if needed
    df.columns = ["label", "message"]

    # Convert labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean text
    df["message"] = df["message"].apply(clean_text)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.2f}")

    return model, vectorizer