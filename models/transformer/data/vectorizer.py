import torch


class NETVectorizer:

    def __init__(self, lookup):
        self.lookup = lookup
        self.special_tokens = set([lookup.eos_token, lookup.sos_token])

    def vectorize(self, symbol_list):
        arr = [self.lookup.stoi[s] for s in symbol_list if s in self.lookup]
        return torch.tensor(arr, dtype=torch.long)

    def textualize(self, id_list, no_special_tokens):
        symbols = [self.lookup.itos[str(idx)] for idx in id_list]
        if no_special_tokens:
            symbols = [x for x in symbols if x not in self.special_tokens]
        return ''.join(symbols)
