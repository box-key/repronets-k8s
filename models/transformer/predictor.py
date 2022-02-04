from flask import Flask, request
from flask_restful import Resource, Api

from pprint import pformat
import logging
import math
import os


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class TransformerNETransliterator(Resource):
    def __init__(self, **kwargs):
        self.net_model = kwargs["net_model"]

    def format_output(self, output):
        formatted_output = {}
        probs = []
        for i, pred in enumerate(output, start=1):
            key = "No.{}".format(i)
            tokens = "".join(pred["tokens"])
            # weight is the log probability
            probs.append(math.exp(pred["score"]))
            formatted_output[key] = {"tokens": tokens}
        # convert weights to probability distribution
        denom = sum(probs)
        for prob, key in zip(probs, formatted_output.keys()):
            if denom > 0:
                formatted_output[key]["prob"] = round(prob / denom, 4)
            else:
                formatted_output[key]["prob"] = 0
        return formatted_output

    def get(self):
        beam_size = int(request.args.get("beam", 0))
        input_text = request.args.get("input", "")
        logger.debug("query params = '{}'".format(request.args))
        if len(input_text) == 0:
            resp = {"status": 400, "message": "input is empty"}
            return resp
        elif beam_size <= 0:
            resp = {
                "status": 400,
                "message": (
                    "Beam size must be grater than 0, instead "
                    "received = '{}'".format(beam_size)
                ),
            }
            return resp
        if len(input_text) > 36:
            resp = {
                "status": 400,
                "message": (
                    "ILLEGAL INPUT: 'input' word must consist of "
                    "less than 37 characters"
                ),
            }
            return resp
        # lower text and remove white space
        input_text = input_text.lower().replace(" ", "")
        # get prediction
        prediction = self.net_model.translate_batch(
            [list(input_text)],
            beam_size=beam_size,
            num_hypotheses=beam_size,
            return_scores=True,
        )
        resp = {
            "data": self.format_output(prediction[0]),
            "status": 200,
            "message": "Successfully made predictions",
        }
        return resp


def create_app():
    import ctranslate2
    import pathlib

    # get language name
    language = os.getenv("LANGUAGE_NAME")
    # packing model
    path = pathlib.Path(__file__).absolute().parents[2] / "model_store"
    logger.info(f"model directory path = '{str(path)}'")
    logger.info(f"{language=}")
    net_model = ctranslate2.Translator(str(path / language / "transformer" / "ctranslate2_released"))
    logger.info(f"packed {language} model")
    # init flask objects
    app = Flask(__name__)
    api = Api(app)
    api.app.config["RESTFUL_JSON"] = {"ensure_ascii": False}
    api.add_resource(
        TransformerNETransliterator,
        "/predict",
        resource_class_kwargs={"net_model": net_model},
    )
    logger.info(f"ready for service!")
    return app
