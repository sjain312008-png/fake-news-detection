from flask import Flask, request, jsonify, render_template
import os
import pickle

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "fake_news_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)


@app.route("/")
def home():
    return "Fake News Detection Backend is Running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    news = data.get("news", "").strip()

    if not news:
        return jsonify({"error": "Please enter some news text."}), 400

    news_tfidf = tfidf.transform([news])
    prediction = model.predict(news_tfidf)[0]

    result = "FAKE NEWS" if prediction == 0 else "REAL NEWS"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
