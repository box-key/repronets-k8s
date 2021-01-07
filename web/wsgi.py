from flask import Flask, render_template, request

import os
import requests


app = Flask(__name__)


LANGUAGES = [
    "japanese",
    "korean",
    "chinese",
    "hebrew",
    "arabic",
    "russian"
]
MODELS = ["phonetisaurus", "transformer"]
DEFAULT_LAN = "japanese"
BEAM_SIZE = [1, 2, 3, 4, 5]
DEFAULT_BSIZE = 1
PS_ROUTE = 'http://0.0.0.0:5001/predict'
TR_ROUTE = 'http://0.0.0.0:5002/predict'


def get_output(resp=None, language=None, bsize=None):
    # if inputs are None, returns default page
    if resp is None and language is None and bsize is None:
        language = DEFAULT_LAN
        bsize = DEFAULT_BSIZE
        resp = None
    return render_template('index.html',
                            languages=LANGUAGES,
                            beam_size=BEAM_SIZE,
                            selected_lan=language,
                            selected_bsize=bsize,
                            resp=resp)



def get_ps_output(language, beam_size, input_text):
    payload = {
        "language": language,
        "beam_size": beam_size,
        "input": input_text
    }
    resp = requests.get(PS_ROUTE, params=payload)
    return resp.json()


def get_ts_output(language, beam_size, input_text):
    payload = {
        "language": language,
        "beam_size": beam_size,
        "input": input_text
    }
    resp = requests.post(PS_ROUTE, data=payload)
    return resp.json()


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        language = request.form.get("language")
        beam_size = int(request.form.get("beam_size"))
        input_text = request.form["input_text"]
        if len(input_text) == 0:
            return get_output()
        # get phonetisaurus output
        ps_resp = get_ps_output(language, beam_size, input_text)
        if ps_resp['status'] == 400:
            return get_output(resp={"error": ps_resp['message']})
        # get transformer output
        ts_resp = get_ts_output(language, beam_size, input_text)
        if ts_resp['status'] == 400:
            return get_output(resp={"error": ts_resp['message']})
        predictions = []
        ranks = ["No.{}".format(i + 1) for i in range(beam_size)]
        for rank in ranks:
            prediction = {}
            prediction["ps"] = ps_resp["data"][rank]
            prediction["ts"] = ts_resp["data"][rank]
            prediction["rank"] = rank
            predictions.append(prediction)
        resp = {
            "model_names": MODELS,
            "predictions": predictions
        }
        return get_output(resp=resp,
                          language=language,
                          bsize=beam_size)
    return get_output()
