from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    warning = None

    if request.method == "POST":
        news = request.form["news"]
        input_vector = vectorizer.transform([news])
        result = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector)[0]
        confidence = round(prob[result] * 100, 2)
        prediction = "REAL" if result == 1 else "FAKE"

        if confidence < 65:
            warning = "⚠️ This news may be very new or unfamiliar to the model."

    return render_template("index.html", prediction=prediction, confidence=confidence, warning=warning)

if __name__ == "__main__":
    app.run(debug=True)
