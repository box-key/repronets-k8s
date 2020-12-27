from flask import Flask, render_template, request

import os
import requests


app = Flask(__name__)


LANGUAGES = ["japanese", "korean", "chinese"]
MODEL_SERVER_ROUTE = 'http://127.0.0.1:1133/'


@app.route('/', methods=("GET", "POST"))
def index():
    if request.method == "POST":
        language = request.form.get("language")
        input_text = request.form["input_text"]
        # send request to model server
        input_data = {
            "lan": language,
            "input_text": input_text,
            "max_pred_len": len(input_text) + 5
        }
        res = requests.post(MODEL_SERVER_ROUTE, data=input_text)
        result = {}
        if res.status_code != 200:
            result["prediction"] = "Bad input, try again"
            return render_template('index.html', languages=LANGUAGES, res=result)
        # format output
        result["prediction"] = "?"
        return render_template('index.html', languages=LANGUAGES, res=result)
    return render_template('index.html', languages=LANGUAGES, res=None)
