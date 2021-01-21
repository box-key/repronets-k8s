import bentoml as bento
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput, JsonOutput

import ctranslate2

import logging
import pathlib


@bento.env(requirements_txt_file='requirements.txt')
@bento.artifacts([
    PickleArtifact('model_ara'),
    PickleArtifact('model_chi'),
    PickleArtifact('model_heb'),
    PickleArtifact('model_jpn'),
    PickleArtifact('model_kor'),
    PickleArtifact('model_rus')
])
class BentoNETransliterator(bento.BentoService):

    def format_output(self, output):
        formatted_output = {}
        for i, pred in enumerate(output, start=1):
            key = "No.{}".format(i)
            formatted_output[key] = ''.join(pred['tokens'])
        return formatted_output

    @bento.api(input=JsonInput(), output=JsonOutput(), batch=False)
    def predict(self, json_obj):
        language = json_obj.get("language", "")
        beam_size = int(json_obj.get("beam", 0))
        input_text = json_obj.get("input", "")
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
        if language == "ara":
            translator = self.artifacts.model_ara
        elif language == "chi":
            translator = self.artifacts.model_chi
        elif language == "heb":
            translator = self.artifacts.model_heb
        elif language == "jpn":
            translator = self.artifacts.model_jpn
        elif language == "kor":
            translator = self.artifacts.model_kor
        elif language == "rus":
            translator = self.artifacts.model_rus
        else:
            resp = {
                "status": 400,
                "message": "language = '{}' is not supported".format(language)
            }
            return resp
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


def pack_model():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)
    path = pathlib.Path(__file__).absolute().parents[2] / 'model_store'
    logger.debug("model directory path = '{}'".format(path))
    # start packing
    bento_net = BentoNETransliterator()
    # pack arabic model
    model = ctranslate2.Translator(str(path / 'arabic' / 'ctranslate2_released'))
    bento_net.pack("model_ara", model)
    logger.info("packed arabic model")
    # pack chiense model
    model = ctranslate2.Translator(str(path / 'chinese' / 'ctranslate2_released'))
    bento_net.pack("model_chi", model)
    logger.info("packed chinese model")
    # pack hebrew model
    model = ctranslate2.Translator(str(path / 'hebrew' / 'ctranslate2_released'))
    bento_net.pack("model_heb", model)
    logger.info("packed hebrew model")
    # pack japanese model
    model = ctranslate2.Translator(str(path / 'katakana' / 'ctranslate2_released'))
    bento_net.pack("model_jpn", model)
    logger.info("packed japanese model")
    # pack korean model
    model = ctranslate2.Translator(str(path / 'korean' / 'ctranslate2_released'))
    bento_net.pack("model_kor", model)
    logger.info("packed korean model")
    # pack russian model
    model = ctranslate2.Translator(str(path / 'russian' / 'ctranslate2_released'))
    bento_net.pack("model_rus", model)
    logger.info("packed russian model")
    bento_net.save()


if __name__ == "__main__":
    pack_model()
