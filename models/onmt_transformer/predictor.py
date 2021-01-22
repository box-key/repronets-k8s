from flask import Flask, request
from flask_restful import Resource, Api

import ctranslate2

import logging
import pathlib


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


class TransformerNETransliterator(Resource):

    def __init__(self, **kwargs):
        self.net_models = kwargs['net_models']

    def format_output(self, output):
        formatted_output = {}
        for i, pred in enumerate(output, start=1):
            key = "No.{}".format(i)
            formatted_output[key] = ''.join(pred['tokens'])
        return formatted_output

    def get(self):
        language = request.form.get("language", "")
        beam_size = int(request.form.get("beam", 0))
        input_text = request.form.get("input", "")
        if len(input_text) == 0:
            resp = {"status": 400, "message": "input is empty"}
            return resp
        elif beam_size <= 0:
            resp = {
                "status": 400,
                "message": ("Beam size must be grater than 0, instead "
                            "received = '{}'".format(beam_size))
            }
            return resp
        # lower text and remove white space
        input_text = input_text.lower().replace(" ", "")
        # get model depending on language
        if language not in self.net_models:
            resp = {
                "status": 400,
                "message": "language = '{}' is not supported".format(language)
            }
            return resp
        else:
            translator = self.net_models[language]
        # get prediction
        prediction = translator.translate_batch([list(input_text)],
                                                beam_size=beam_size,
                                                num_hypotheses=beam_size,
                                                return_scores=False)
        resp = {
            "data": self.format_output(prediction[0]),
            "status": 200,
            "message": "Successfully made predictions"
        }
        return resp


def get_app():
    path = pathlib.Path(__file__).absolute().parents[2] / 'model_store'
    logger.info("model directory path = '{}'".format(path))
    # start packing
    net_models = {}
    # pack arabic model
    model_ara = ctranslate2.Translator(str(path / 'arabic' / 'ctranslate2_released'))
    net_models["ara"] = model_ara
    logger.info("packed arabic model")
    # pack chiense model
    model_chi = ctranslate2.Translator(str(path / 'chinese' / 'ctranslate2_released'))
    net_models["chi"] = model_chi
    logger.info("packed chinese model")
    # pack hebrew model
    model_heb = ctranslate2.Translator(str(path / 'hebrew' / 'ctranslate2_released'))
    net_models["heb"] = model_heb
    logger.info("packed hebrew model")
    # pack japanese model
    model_jpn = ctranslate2.Translator(str(path / 'katakana' / 'ctranslate2_released'))
    net_models["jpn"] = model_jpn
    logger.info("packed japanese model")
    # pack korean model
    model_kor = ctranslate2.Translator(str(path / 'korean' / 'ctranslate2_released'))
    net_models["kor"] = model_kor
    logger.info("packed korean model")
    # pack russian model
    model_rus = ctranslate2.Translator(str(path / 'russian' / 'ctranslate2_released'))
    net_models["rus"] = model_rus
    logger.info("packed russian model")
    # init flask objects
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(TransformerNETransliterator,
                     '/predict',
                     resource_class_kwargs={"net_models": net_models})
    return app
