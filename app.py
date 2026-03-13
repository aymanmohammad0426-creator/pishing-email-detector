from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        email = request.form["email"]

        email_vector = vectorizer.transform([email])

        result = model.predict(email_vector)

        if result[0] == 1:
            prediction = "Phishing Email ⚠️"
        else:
            prediction = "Safe Email ✅"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)