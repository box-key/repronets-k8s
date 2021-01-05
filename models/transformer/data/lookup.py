import json
import logging


class SymLookup:

    pad_token = '<pad>'
    eos_token = '<eos>'
    sos_token = '<sos>'

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
        src_vocab, trg_vocab = {}, {}
        # read data file
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                items = line.strip().split()
                src_vocab.update({char:1 for char in list(items[0])})
                trg_vocab.update({phoneme:1 for phoneme in items[1:]})
            src_vocab_list = [cls.pad_token, cls.eos_token, cls.sos_token]
            trg_vocab_list = [cls.pad_token, cls.eos_token, cls.sos_token]
            for key in src_vocab.keys():
                src_vocab_list.append(key)
            for key in trg_vocab.keys():
                trg_vocab_list.append(key)
        # build itos and stoi for src
        src_itos = {}
        src_itos.update(enumerate(src_vocab_list))
        src_stoi = dict((v, k) for k, v in src_itos.items())
        # build itos and stoi for trg
        trg_itos = {}
        trg_itos.update(enumerate(trg_vocab_list))
        trg_stoi = dict((v, k) for k, v in trg_itos.items())
        return cls(src_stoi, src_itos), cls(trg_stoi, trg_itos)
