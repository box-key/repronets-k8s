from flask import Flask, render_template, request, jsonify

import os
import requests
import subprocess
import re


app = Flask(__name__)


def get_output(language, input_text, beam_size):
    cmd = "phonetisaurus predict --nbest {} --model model_store/{}_ps_1/model.fst {}"
    res = subprocess.check_output([cmd.format(beam_size, language, input_text)],
                                  shell=True)
    output = []
    for pred in res.decode('utf-8').split('\n')[:-1]:
        output.append(''.join(pred.split()[1:]))
    return format_output(output)


def format_output(output):
    formatted_output = {}
    for i, pred in enumerate(output):
        key = "No.{}".format(i + 1)
        formatted_output[key] = pred
    return formatted_output


@app.route('/predict', methods=("GET"))
def index():
    language = request.args.get("language", "")
    beam_size = int(request.args.get("beam_size", 0))
    input_text = request.args.get("input", "")
    if len(input_text) == 0:
        resp = {"status": 400, "message": "input is empty"}
        return jsonify(resp)
    elif beam_size <= 0:
        resp = {
            "status": 400,
            "message": ("Beam size must be grater than 0, instead "
                        "received = '{}'".format(beam_size))
        }
        return jsonify(resp)
    # lower text and remove white space
    input_text = input_text.lower().replace(" ", "")
    # call phonetisaurus to get prediction
    if language == "japanese":
        output = get_output("katakana", input_text, beam_size)
    elif language == "korean":
        output = get_output("korean", input_text, beam_size)
    elif language == "chinese":
        output = get_output("chinese", input_text, beam_size)
    elif language == "arabic":
        output = get_output("arabic", input_text, beam_size)
    elif language == "hebrew":
        output = get_output("hebrew", input_text, beam_size)
    elif language == "russian":
        output = get_output("russian", input_text, beam_size)
    else:
        resp = {
            "status": 400,
            "message": "input language = '{}' doesn't exist".format(language)
        }
        return jsonify(resp)
    # format output
    result = {
        "data": format_output(output),
        "status": 200,
        "message": "Successfully make predictions"
    }
    return jsonify(resp)
