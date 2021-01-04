from torch.utils.data import Dataset

from .lookup import SymLookup


class NETDataset(Dataset):

    def __init__(self, net_pairs):
        self.net_pairs = net_pairs

    def __len__(self):
        return len(self.net_pairs)

    def __getitem__(self, idx):
        return self.net_pairs[idx]

    @classmethod
    def load(cls, data_path, src_vectorizer, trg_vectorizer, trg_separator):
        net_pairs = []
        sos_list = [SymLookup.sos_token]
        eos_list = [SymLookup.eos_token]
        with open(data_path, mode="r", encoding='utf-8') as f:
            for line in f:
                if line:
                    items = line.split()
                    source = list(items[0].strip().lower())
                    target = " ".join(items[1:]).strip().split(trg_separator)
                    # print(source, target)
                    source_ints = src_vectorizer.vectorize(
                        sos_list + source + eos_list
                    )
                    target_ints = trg_vectorizer.vectorize(
                        sos_list + target + eos_list
                    )
                    net_pairs.append(
                        {"source": source_ints, "target": target_ints}
                    )
                    # print(source_ints, target_ints)
        return cls(net_pairs=net_pairs)
