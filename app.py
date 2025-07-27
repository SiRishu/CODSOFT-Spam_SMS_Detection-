from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained spam detection model and vectorizer
model = joblib.load("notebook/spam_detection_randomforest_model.pkl")
vectorizer = joblib.load("notebook/tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        transformed_message = vectorizer.transform([message])
        prediction = model.predict(transformed_message)[0]
        result = "Spam ❌" if prediction == 1 else "Not Spam ✅"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
