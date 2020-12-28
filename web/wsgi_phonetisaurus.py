from flask import Flask, render_template, request

import os
import requests
import subprocess
import re


app = Flask(__name__)


LANGUAGES = [
    "japanese",
    "korean",
    "chinese",
    "hebrew",
    "arabic",
    "russian"
]
BEAM_SIZE = [1, 2, 3, 4, 5]


def get_output(language, input_text, beam_size):
    cmd = "phonetisaurus predict --nbest {} --model model_store/{}_ps_1/model.fst {}"
    res = subprocess.check_output([cmd.format(beam_size, language, input_text)],
                                  shell=True)
    output = []
    for pred in res.decode('utf-8').split('\n'):
        output.append(''.join(pred.split()[1:]))
    unique_predicitons = set(output)
    return format_output(unique_predicitons)


def format_output(unique_predicitons):
    if len(unique_predicitons) == 0:
        formatted_output = ""
    elif len(unique_predicitons) == 1:
        formatted_output = str(unique_predicitons)
    elif len(unique_predicitons) == 2:
        formatted_output = ' or '.join(unique_predicitons)
    else:
        formatted_output = ', '.join(unique_predicitons)
    return formatted_output


@app.route('/', methods=("GET", "POST"))
def index():
    if request.method == "POST":
        language = request.form.get("language")
        beam_size = request.form.get("beam_size")
        input_text = request.form["input_text"]
        # lower text and remove white space
        input_text = input_text.lower().replace(" ", "")
        result = {}
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
            result["prediction"] = "Bad input, try again"
            return render_template('index.html',
                                   languages=LANGUAGES,
                                   beam_size=BEAM_SIZE,
                                   res=result)
        # format output
        result["prediction"] = "It's {}".format(output)
        return render_template('index.html',
                               languages=LANGUAGES,
                               beam_size=BEAM_SIZE,
                               res=result)
    return render_template('index.html',
                           languages=LANGUAGES,
                           beam_size=BEAM_SIZE,
                           res=None)
