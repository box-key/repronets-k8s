from flask import Flask, render_template, request
from flask_table import Table, Col

import os
import requests
import json
import logging


app = Flask(__name__)


LANGUAGES = [
    "japanese",
    "korean",
    "chinese",
    "hebrew",
    "arabic",
    "russian"
]
MODELS = [" Phonetisaurus ", " Transformer "]
DEFAULT_LAN = "japanese"
BEAM_SIZE = [1, 2, 3, 4, 5]
DEFAULT_BSIZE = 1
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
        item["ps_prediction"] = ps_resp["data"][rank]
        item["ts_prediction"] = ts_resp["data"][rank]
        item["rank"] = rank
        items.append(item)
    return items


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
    resp = requests.post(TS_ROUTE, data=json.dumps(payload))
    logger.debug(resp.content)
    return resp.json()


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        language = request.form.get("language", "")
        beam_size = int(request.form.get("beam_size", 0))
        input_text = request.form.get("input_text", "")
        logger.debug("inputs = '{}'".format(request.form))
        # get phonetisaurus output
        ps_resp = get_ps_output(language, beam_size, input_text)
        if ps_resp['status'] == 400:
            return get_output(resp={"error": ps_resp['message']})
        logger.debug("output from phonetisaurus = '{}'".format(ps_resp))
        # get transformer output
        ts_resp = get_ts_output(language, beam_size, input_text)
        if ts_resp['status'] == 400:
            return get_output(resp={"error": ts_resp['message']})
        logger.debug("output from transformer = '{}'".format(ts_resp))
        items = get_table_items(ps_resp, ts_resp, beam_size)
        resp = {
            "model_names": MODELS,
            "table": OutputTable(items)
        }
        return get_output(resp=resp,
                          language=language,
                          bsize=beam_size)
    return get_output()
