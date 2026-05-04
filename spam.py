from model import train_model
from preprocess import clean_text

def predict_spam(model, vectorizer, text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)

    return "SPAM" if result[0] == 1 else "NOT SPAM"


if __name__ == "__main__":
    print("Training model...")

    model, vectorizer = train_model("dataset.csv")

    print("\nEnter email text (type 'exit' to quit):")

    while True:
        user_input = input(">> ")

        if user_input.lower() == "exit":
            break

        result = predict_spam(model, vectorizer, user_input)
        print("Prediction:", result)