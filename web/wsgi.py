from flask import Flask, render_template, request
from flask_table import Table, Col

import os
import requests
import json
import logging


app = Flask(__name__)


LANGUAGE_MAPPER = {
    "Arabic": "ara",
    "Chinese": "chi",
    "Hebrew": "heb",
    "Japanese": "jpn",
    "Korean": "kor",
    "Russian": "rus"
}
LAN_LOOKUP = set([lan for lan in LANGUAGE_MAPPER.values()])
INPUT_LANGUAGES = [lan for lan in LANGUAGE_MAPPER.keys()]
MODELS = ["Phonetisaurus", "Transformer"]
MODEL_LOOKUP = set([x.lower() for x in MODELS])
BEAM_SIZE = [1, 2, 3, 4, 5]
PS_ROUTE = 'http://0.0.0.0:5001/predict'
TS_ROUTE = 'http://0.0.0.0:5002/predict'


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


class OutputTable(Table):
    rank = Col("Rank")
    ps_prediction = Col(MODELS[0])
    ts_prediction = Col(MODELS[1])
    border = True


def get_table_items(ps_resp, ts_resp, beam_size):
    items = []
    ranks = ["No.{}".format(i + 1) for i in range(beam_size)]
    for rank in ranks:
        item = {}
        # get ps output
        try:
            ps_token = ps_resp["data"][rank]["tokens"]
            ps_prob = ps_resp["data"][rank]["prob"]
        except KeyError:
            ps_token = '-'
            ps_prob = 0.0
        item["ps_prediction"] = "{} ({})".format(ps_token, ps_prob)
        # get ts output
        try:
            ts_token = ts_resp["data"][rank]["tokens"]
            ts_prob = ts_resp["data"][rank]["prob"]
        except KeyError:
            ts_token = '-'
            ts_prob = 0.0
        item["ts_prediction"] = "{} ({})".format(ts_token, ts_prob)
        item["rank"] = rank
        items.append(item)
    return items


def get_output(resp=None, language=None, bsize=None):
    # if inputs are None, returns default page
    if resp is None and language is None and bsize is None:
        language = INPUT_LANGUAGES[0]
        bsize = BEAM_SIZE[0]
        resp = None
    return render_template('index.html',
                            languages=INPUT_LANGUAGES,
                            beam_size=BEAM_SIZE,
                            selected_lan=language,
                            selected_bsize=bsize,
                            resp=resp)



def get_ps_output(language, beam_size, input_text):
    payload = {
        "language": language,
        "beam": beam_size,
        "input": input_text
    }
    resp = requests.get(PS_ROUTE, params=payload)
    logger.debug(resp.content)
    return resp.json()


def get_ts_output(language, beam_size, input_text):
    payload = {
        "language": language,
        "beam": beam_size,
        "input": input_text
    }
    resp = requests.get(TS_ROUTE, params=payload)
    logger.debug(resp.content)
    return resp.json()


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        selected_language = request.form.get("language", "")
        language = LANGUAGE_MAPPER[selected_language]
        beam_size = int(request.form.get("beam_size", 1))
        input_text = request.form.get("input_text", "")
        logger.debug("inputs = '{}'".format(request.form))
        # get phonetisaurus output
        ps_resp = get_ps_output(language, beam_size, input_text)
        if ps_resp['status'] == 400:
            return get_output(resp={"error": ps_resp['message']})
        # get transformer output
        ts_resp = get_ts_output(language, beam_size, input_text)
        if ts_resp['status'] == 400:
            return get_output(resp={"error": ts_resp['message']})
        items = get_table_items(ps_resp, ts_resp, beam_size)
        resp = {
            "model_names": MODELS,
            "table": OutputTable(items)
        }
        return get_output(resp=resp,
                          language=selected_language,
                          bsize=beam_size)
    return get_output()


@app.route("/predict", methods=["GET"])
def predict():
    language = request.args.get("language", "")
    input_text = request.args.get("input", "")
    model_type = request.args.get("model", "")
    # 400 if receive non integer value
    try:
        beam_size = int(request.args.get("beam", 0))
    except ValueError:
        resp = {
            "status": 400,
            "message": ("ILLEGAL INPUT: 'beam_size' must be from 1 to 5, instead "
                        "received '{}'".format(request.args.get("beam_size")))
        }
        return json.dumps(resp)
    # 400 if receive non integer value
    if (beam_size <= 0) or (beam_size > 5):
        resp = {
            "status": 400,
            "message": ("VALUE ERROR: 'beam_size' must be from 1 to 5, instead "
                        "received '{}'".format(beam_size))
        }
        return json.dumps(resp)
    # chekc input and langauge
    if not isinstance(language, str) or not isinstance(input_text, str):
        resp = {
            "status": 400,
            "message": "ILLEGAL INPUT: 'langauge' and 'input' must be string."
        }
        return json.dumps(resp)
    if language not in LAN_LOOKUP:
        resp = {
            "status": 400,
            "message": ("ILLEGAL INPUT: Currently '{}' language are available, "
                        "instead received '{}'".format(LAN_LOOKUP, language))
        }
        return json.dumps(resp)
    if len(input_text) > 36:
        resp = {
            "status": 400,
            "message": ("ILLEGAL INPUT: 'input' word must consist of less than "
                        "37 characters")
        }
        return json.dumps(resp)
    # get inference
    if model_type == "phonetisaurus":
        output = get_ps_output(language, beam_size, input_text)
    elif model_type == "transformer":
        output = get_ts_output(language, beam_size, input_text)
    else:
        # 400 if model_type doesn't match
        resp = {
            "status": 400,
            "message": ("Currently '{}' models are available, instead received "
                        "'{}'".format([x.lower() for x in MODELS], model_type))
        }
        return json.dumps(resp)
    # return output from the model
    return json.dumps(output, ensure_ascii=False)
