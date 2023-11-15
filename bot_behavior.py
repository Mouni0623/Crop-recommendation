from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import json
import requests



app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def f():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


@app.route("/output", methods=["GET", "POST"])
def output():
    if request.method == 'POST':
        var1 = request.form["N"]
        var2 = request.form["P"]
        var3 = request.form["K"]
        var4 = request.form["TEMPERATURE"]
        var5 = request.form["HUMIDITY"]
        var6 = request.form["PH"]
        var7 = request.form["RAINFALL"]
        var8 = request.form["LABEL"]
       

        # Convert the input data into a numpy array
        predict_data = np.array([var1, var2, var3, var4, var5, var6, var7, var8]).reshape(1, -1)

        # Use the loaded model to make predictions
        predict = model.predict(predict_data)

        if (predict == 1):
            return render_template('output.html', predict="Bot")
        else:
            return render_template('output.html', predict="Not Bot")
    return render_template("output.html")

if __name__ == "__main__":
    app.run(debug=False)
