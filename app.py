from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

####################################################

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods=['get', 'post'])
def prediction():
    user_review = request.form.get("test_string")

    model = joblib.load("best_models/naive_bayes.pkl")

    # Predict the sentiment
    prediction = model.predict([user_review])
    
    # Interpret the prediction
    result = "Positive" if prediction[0] == 1 else "Negative"

    return render_template("output.html", prediction=result)

######################################################

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")