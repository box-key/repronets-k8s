from torch.utils.data import Dataset


class NETDataset(Dataset):

    def __init__(self, net_pairs, vectorizer):
        self.net_pairs = net_pairs
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.net_pairs)

    def __getitem__(self, idx):
        return self.net_pairs[idx]

    @classmethod
    def load(cls, data_path, vectorizer, trg_separator):
        net_pairs = []
        sos_list = [vectorizer.lookup.sos_token]
        eos_list = [vectorizer.lookup.eos_token]
        with open(data_path, mode="r", encoding='utf-8') as f:
            for line in f:
                if line:
                    items = line.split()
                    source = list(items[0].strip().lower())
                    target = " ".join(items[1:]).strip().split(trg_separator)
                    print(source, target)
                    source_ints = vectorizer.vectorize(
                        sos_list + source + eos_list
                    )
                    target_ints = vectorizer.vectorize(
                        sos_list + target + eos_list
                    )
                    net_pairs.append(
                        {"source": source_ints, "target": target_ints}
                    )
                    print(source_ints, target_ints)
        return cls(net_pairs=net_pairs, vectorizer=vectorizer)
