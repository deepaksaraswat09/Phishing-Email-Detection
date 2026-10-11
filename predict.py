import os
import re
import string
import sys
import warnings
import numpy as np
import joblib

# Suppress harmless scikit-learn cross-version unpickling warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows consoles if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths to trained model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


def clean_text(text: str) -> str:
    """Preprocess email text according to the training pipeline:
    - Lowercase
    - Remove HTML tags
    - Replace URLs with 'url' token
    - Replace email addresses with 'email' token
    - Remove punctuation
    - Normalize whitespace
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Model or Vectorizer file not found in {os.path.join(BASE_DIR, 'models')}."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(email_text: str, model, vectorizer):
    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    # Robust probability estimation across scikit-learn versions
    prob_safe, prob_phishing = None, None
    try:
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(vector)[0])
            prob_phishing = 1.0 / (1.0 + np.exp(-score))
            prob_safe = 1.0 - prob_phishing
    except Exception:
        pass

    # 0 = SAFE, 1 = PHISHING
    is_phishing = bool(prediction == 1)
    label = "PHISHING" if is_phishing else "SAFE"

    return {
        "label": label,
        "is_phishing": is_phishing,
        "probabilities": {
            "safe": prob_safe * 100 if prob_safe is not None else None,
            "phishing": prob_phishing * 100 if prob_phishing is not None else None,
        },
    }


def main():
    print("=" * 60)
    print("      PHISHING EMAIL DETECTION SYSTEM (ML Inference)      ")
    print("=" * 60)

    try:
        model, vectorizer = load_artifacts()
        print("[+] Model and TF-IDF Vectorizer loaded successfully.\n")
    except Exception as e:
        print(f"[-] Error loading model artifacts: {e}")
        sys.exit(1)

    if len(sys.argv) > 1:
        # CLI argument mode
        email_input = " ".join(sys.argv[1:])
        result = predict(email_input, model, vectorizer)
        print(f"Input: {email_input[:100]}...")
        print(f"Prediction: [{result['label']}]")
        if result["probabilities"]["safe"] is not None:
            print(
                f"Confidence -> Safe: {result['probabilities']['safe']:.2f}% | Phishing: {result['probabilities']['phishing']:.2f}%"
            )
        return

    # Interactive loop mode
    print("Type an email subject/body (or 'exit' / 'quit' to stop):\n")
    while True:
        try:
            user_input = input("Enter email text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting. Thank you!")
                break

            res = predict(user_input, model, vectorizer)
            badge = (
                "[!] PHISHING EMAIL DETECTED"
                if res["is_phishing"]
                else "[*] SAFE EMAIL"
            )
            print("-" * 50)
            print(f"Result: {badge}")
            if res["probabilities"]["safe"] is not None:
                print(
                    f"Confidence: Safe = {res['probabilities']['safe']:.2f}% | Phishing = {res['probabilities']['phishing']:.2f}%"
                )
            print("-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Thank you!")
            break


if __name__ == "__main__":
    main()
