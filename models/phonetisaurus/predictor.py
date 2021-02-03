from flask import Flask, request
from flask_restful import Resource, Api

import logging
import math


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


class PhonetisaurusNETransliterator(Resource):

    def __init__(self, **kwargs):
        self.net_models = kwargs['net_models']

    def format_output(output):
        formatted_output = {}
        for i, pred in enumerate(output, start=1):
            key = "No.{}".format(i)
            tokens = ''.join([c for c in pred.Uniques])
            seq_prob = math.exp(-pred.PathWeights)
            formatted_output[key] = {
                "prob": seq_prob,
                "tokens": tokens
            }
        return formatted_output

    def get(self):
        language = request.args.get("language", "")
        beam_size = int(request.args.get("beam", 0))
        input_text = request.args.get("input", "")
        if len(input_text) == 0:
            resp = {"status": 400, "message": "input is empty"}
            return resp
        if len(input_text) > 36:
            resp = {
                "status": 400,
                "message": ("ILLEGAL INPUT: 'input' word must consist of less than "
                            "37 characters")
            }
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
        # call phonetisaurus to get prediction
        # get model depending on language
        if language not in self.net_models:
            resp = {
                "status": 400,
                "message": "language = '{}' is not supported".format(language)
            }
            return resp
        else:
            translator = self.net_models[language]
        prediction = translator.Phoneticize(word=input_text,
                                            nbest=beam_size,
                                            beam=beam_size*10,
                                            threshold=99,
                                            write_fsts=False,
                                            accumulate=False,
                                            pmass=99)
        # format output
        resp = {
            "data": self.format_output(prediction),
            "status": 200,
            "message": "Successfully made predictions"
        }
        return resp


def create_app():
    import phonetisaurus
    import pathlib
    path = pathlib.Path(__file__).absolute().parents[2] / 'model_store'
    logger.info("model directory path = '{}'".format(path))
    # start packing
    net_models = {}
    # pack arabic model
    model_ara = phonetisaurus.Phonetisaurus(
        str(path / 'arabic_ps_1' / 'model.fst')
    )
    net_models["ara"] = model_ara
    logger.info("packed arabic model")
    # pack chiense model
    model_chi = phonetisaurus.Phonetisaurus(
        str(path / 'chinese_ps_1' / 'model.fst')
    )
    net_models["chi"] = model_chi
    logger.info("packed chinese model")
    # pack hebrew model
    model_heb = phonetisaurus.Phonetisaurus(
        str(path / 'hebrew_ps_1' / 'model.fst')
    )
    net_models["heb"] = model_heb
    logger.info("packed hebrew model")
    # pack japanese model
    model_jpn = phonetisaurus.Phonetisaurus(
        str(path / 'katakana_ps_1' / 'model.fst')
    )
    net_models["jpn"] = model_jpn
    logger.info("packed japanese model")
    # pack korean model
    model_kor = phonetisaurus.Phonetisaurus(
        str(path / 'korean_ps_1' / 'model.fst')
    )
    net_models["kor"] = model_kor
    logger.info("packed korean model")
    # pack russian model
    model_rus = phonetisaurus.Phonetisaurus(
        str(path / 'russian_ps_1' / 'model.fst')
    )
    net_models["rus"] = model_rus
    logger.info("packed russian model")
    # init flask objects
    app = Flask(__name__)
    api = Api(app)
    api.app.config['RESTFUL_JSON'] = {'ensure_ascii': False}
    api.add_resource(PhonetisaurusNETransliterator,
                     '/predict',
                     resource_class_kwargs={"net_models": net_models})
    return app
