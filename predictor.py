import bentoml as bento
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput, JsonOutput

from models.inference.factory import NETModelFactory
from models.inference.transliterator import NETransliterator
from models.data.vectorizer import NETVectorizer
from models.data.lookup import SymLookup
from models.modules.net_transformer import NETTransformer

import torch
import logging


@bento.env(requirements_txt_file='requirements.txt')
@bento.artifacts([
    PickleArtifact('predictor_ch'),
    PickleArtifact('predictor_ja'),
    PickleArtifact('predictor_ko')
])
class BentoNETransliterator(bento.BentoService):

    @bento.api(input=JsonInput(), output=JsonOutput(), batch=False)
    def predict(self, json_obj):
        # get model depending on language
        if json_obj["data"]["lan"] == "japanese":
            predictor = self.artifacts.predictor_ja
        elif json_obj["data"]["lan"] == "korean":
            predictor = self.artifacts.predictor_ko
        elif json_obj["data"]["lan"] == "chinese":
            predictor = self.artifacts.predictor_ch
        else:
            return {
                "status": 400,
                "msg": ("Bad request, 'lan' must be either "
                        "'japanese', 'korean', or 'chinese'.")
            }
        # get prediction
        beam_size = 1
        prediction = predictor.predict(
            src_text=json_obj["data"]["input_text"],
            beam_size=beam_size,
            max_pred_len=json_obj["data"]["max_pred_len"],
        )
        return {
            "status": 200,
            "msg": "Sucefully, predicted",
            "data": prediction
        }


def get_predictor(config_path, vocab_path, model_path):
    device = torch.device('cpu')
    factory = NETModelFactory(config_path,
                              vocab_path,
                              NETTransformer,
                              NETVectorizer,
                              SymLookup.load,
                              device)
    model, vectorizer = factory.produce(model_path)
    predictor = NETransliterator(vectorizer, device, model)
    return predictor


def pack_model():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    # start packing
    bento_net = BentoNETransliterator()
    # japanese lan model
    model_path_ja = "model_store/katakana/3.pt"
    config_path_ja = "model_store/katakana/config.yml"
    vocab_path_ja = "model_store/katakana/vocab.json"
    predictor_ja = get_predictor(config_path=config_path_ja,
                                 vocab_path=vocab_path_ja,
                                 model_path=model_path_ja)
    bento_net.pack("predictor_ja", predictor_ja)
    logger.info("packed japaense lan model")
    # korean lan model
    model_path_ko = "model_store/korean/3.pt"
    config_path_ko = "model_store/korean/config.yml"
    vocab_path_ko = "model_store/korean/vocab.json"
    predictor_ko = get_predictor(config_path=config_path_ko,
                                 vocab_path=vocab_path_ko,
                                 model_path=model_path_ko)
    bento_net.pack("predictor_ko", predictor_ko)
    logger.info("packed korean lan model")
    # chinese lan model
    model_path_ch = "model_store/chinese/3.pt"
    config_path_ch = "model_store/chinese/config.yml"
    vocab_path_ch = "model_store/chinese/vocab.json"
    predictor_ch = get_predictor(config_path=config_path_ch,
                                 vocab_path=vocab_path_ch,
                                 model_path=model_path_ch)
    bento_net.pack("predictor_ch", predictor_ch)
    logger.info("packed chinese lan model")
    bento_net.save()


if __name__ == "__main__":
    pack_model()
