import json
import logging


logger = logging.getLogger(__name__)


class SymLookup:

    pad_token = '<<pad>>'
    eos_token = '<<eos>>'
    sos_token = '<<sos>>'

    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    def __len__(self):
        return len(self.stoi)

    def __contains__(self, symbol):
        if symbol in self.stoi:
            return True
        else:
            return False

    def save(self, output_path):
        with open(output_path, "w") as json_file:
            data = {"stoi": self.stoi, "itos": self.itos}
            json.dump(data, json_file)

    @classmethod
    def load(cls, vocab_path):
        with open(vocab_path) as json_file:
            data = json.load(json_file)
        return cls(stoi=data['stoi'], itos=data['itos'])

    @classmethod
    def build(cls, data_path):
        logger.debug("Building vocabulary")
        vocab = {}
        # read data file
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                items = line.strip().split()
                vocab.update({char:1 for char in list(items[0])})
                vocab.update({phoneme:1 for phoneme in items[1:]})
            vocab_list = [cls.pad_token, cls.eos_token, cls.sos_token]
            for key in sorted(vocab.keys()):
                vocab_list.append(key)
        # build itos and stoi
        itos = {}
        itos.update(enumerate(vocab_list))
        stoi = dict((v, k) for k, v in itos.items())
        logger.debug("Lookup size = {}".format(len(stoi)))
        return cls(stoi=stoi, itos=itos)
