import bentoml as bento
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput, JsonOutput

from modules.seq2seq import Seq2Seq
from inference.factory import NETModelFactory
from inference.transliterator import NETransliterator

import torchtext
import torch

import logging
import pathlib


@bento.env(requirements_txt_file='requirements.txt')
@bento.artifacts([
    PytorchModelArtifact('model_ar'),
    PytorchModelArtifact('model_ch'),
    PytorchModelArtifact('model_he'),
    PytorchModelArtifact('model_ja'),
    PytorchModelArtifact('model_ko'),
    PytorchModelArtifact('model_ru'),
    PickleArtifact('ar_fields'),
    PickleArtifact('ch_fields'),
    PickleArtifact('he_fields'),
    PickleArtifact('ja_fields'),
    PickleArtifact('ko_fields'),
    PickleArtifact('ru_fields'),
    PickleArtifact('predictor'),
])
class BentoNETransliterator(bento.BentoService):

    device = torch.device('cpu')

    def format_output(self, output_list, beam_size):
        formatted_output = {"No.1": ''.join(output_list)}
        for i in range(beam_size - 1):
            key = "No.{}".format(i+2)
            formatted_output[key] = 'N/A'
        return formatted_output

    @bento.api(input=JsonInput(), output=JsonOutput(), batch=False)
    def predict(self, json_obj):
        language = json_obj.get("language", "")
        beam_size = int(json_obj.get("beam_size", 0))
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
        if language == "arabic":
            model = self.artifacts.model_ar
            trg_field = self.artifacts.ar_fields["target"]
            src_field = self.artifacts.ar_fields["source"]
        elif language == "chinese":
            model = self.artifacts.model_ch
            trg_field = self.artifacts.ch_fields["target"]
            src_field = self.artifacts.ch_fields["source"]
        elif language == "hebrew":
            model = self.artifacts.model_he
            trg_field = self.artifacts.he_fields["target"]
            src_field = self.artifacts.he_fields["source"]
        elif language == "japanese":
            model = self.artifacts.model_ja
            trg_field = self.artifacts.ja_fields["target"]
            src_field = self.artifacts.ja_fields["source"]
        elif language == "korean":
            model = self.artifacts.model_ko
            trg_field = self.artifacts.ko_fields["target"]
            src_field = self.artifacts.ko_fields["source"]
        elif language == "russian":
            model = self.artifacts.model_ru
            trg_field = self.artifacts.ru_fields["target"]
            src_field = self.artifacts.ru_fields["source"]
        else:
            resp = {
                "status": 400,
                "message": "input language = '{}' doesn't exist".format(language)
            }
            return resp
        # get prediction
        prediction = self.artifacts.predictor(
            named_entity=input_text,
            max_pred_len=len(input_text) + 2,
            model=model,
            src_field=src_field,
            trg_field=trg_field,
            device=self.device
        )
        resp = {
            "data": self.format_output(prediction, beam_size),
            "status": 200,
            "message": "Successfully make predictions"
        }
        return resp


def get_articrafts(dir_path, model_path):
    device = torch.device('cpu')
    factory = NETModelFactory(dir_path / 'config.yml',
                              dir_path / 'src_field.dill',
                              dir_path / 'trg_field.dill',
                              Seq2Seq,
                              device)
    model, src, trg = factory.produce(dir_path / model_path)
    return model, src, trg


def pack_model():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)
    path = pathlib.Path(__file__).absolute().parents[2] / 'model_store'
    logger.debug("model directory path = '{}'".format(path))
    device = torch.device('cpu')
    # start packing
    bento_net = BentoNETransliterator()
    # pack arabic model
    model, src, trg = get_articrafts(path / 'arabic', '50.pt')
    bento_net.pack("model_ar", model)
    bento_net.pack("ar_fields", {"source": src, "target": trg})
    logger.info("packed arabic model")
    # pack chiense model
    model, src, trg = get_articrafts(path / 'chinese', '50.pt')
    bento_net.pack("model_ch", model)
    bento_net.pack("ch_fields", {"source": src, "target": trg})
    logger.info("packed chinese model")
    # pack hebrew model
    model, src, trg = get_articrafts(path / 'hebrew', '50.pt')
    bento_net.pack("model_he", model)
    bento_net.pack("he_fields", {"source": src, "target": trg})
    logger.info("packed hebrew model")
    # pack japanese model
    model, src, trg = get_articrafts(path / 'katakana', '30.pt')
    bento_net.pack("model_ja", model)
    bento_net.pack("ja_fields", {"source": src, "target": trg})
    logger.info("packed japanese model")
    # pack korean model
    model, src, trg = get_articrafts(path / 'korean', '50.pt')
    bento_net.pack("model_ko", model)
    bento_net.pack("ko_fields", {"source": src, "target": trg})
    logger.info("packed korean model")
    # pack russian model
    model, src, trg = get_articrafts(path / 'russian', '50.pt')
    bento_net.pack("model_ru", model)
    bento_net.pack("ru_fields", {"source": src, "target": trg})
    logger.info("packed russian model")
    # pack predictor
    bento_net.pack("predictor", NETransliterator())
    logger.info("packed predictor")
    bento_net.save()


if __name__ == "__main__":
    pack_model()
