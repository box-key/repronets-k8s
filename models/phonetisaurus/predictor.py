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


NUM_BEAM = 100


class PhonetisaurusNETransliterator(Resource):
    def __init__(self, **kwargs):
        self.net_model = kwargs["net_model"]

    def format_output(self, output):
        formatted_output = {}
        for i, pred in enumerate(output, start=1):
            key = "No.{}".format(i)
            tokens = "".join([self.net_model.FindOsym(c) for c in pred.Uniques])
            seq_prob = math.exp(-pred.PathWeight)
            formatted_output[key] = {"prob": round(seq_prob, 10), "tokens": tokens}
        return formatted_output

    def get(self):
        beam_size = int(request.args.get("beam", 0))
        input_text = request.args.get("input", "")
        logger.info("input params = '{}'".format(request.args))
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
        prediction = self.net_model.Phoneticize(
            word=input_text,
            nbest=beam_size,
            beam=NUM_BEAM,
            write_fsts=False,
            accumulate=False,
            threshold=99,
            pmass=99,
        )
        # format output
        resp = {
            "data": self.format_output(prediction),
            "status": 200,
            "message": "Successfully made predictions",
        }
        return resp


def create_app():
    import phonetisaurus
    import pathlib

    # get language name
    language = os.getenv("LANGUAGE_NAME")
    # packing model
    path = pathlib.Path(__file__).absolute().parents[2] / "model_store"
    logger.info("model directory path = '{}'".format(path))
    logger.info(f"{language=}")
    net_model = phonetisaurus.Phonetisaurus(str(path / language / "phonetisaurus" / "model.fst"))
    # init flask objects
    app = Flask(__name__)
    api = Api(app)
    api.app.config["RESTFUL_JSON"] = {"ensure_ascii": False}
    api.add_resource(
        PhonetisaurusNETransliterator,
        "/predict",
        resource_class_kwargs={"net_model": net_model},
    )
    logger.info(f"ready for service!")
    return app
