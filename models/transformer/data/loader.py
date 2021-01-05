from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


class NETDataLoader:

    def __init__(self, pad_value):
        raise RuntimeError(
            "This class might have a bug. Use Torchtext's BucketLoader instead."
        )
        self.pad_value = pad_value

    def __call__(self, batch):
        output = {}
        source, target = [], []
        # decompose batch into sources and targets
        for x in batch:
            source.append(x["source"])
            target.append(x["target"])
        output['src_padded'] = pad_sequence(source, padding_value=self.pad_value)
        ## src_padded = [src_len, batch_size]
        output['trg_padded'] = pad_sequence(target, padding_value=self.pad_value)
        ## trg_padded = [trg_len, batch_size]
        return output

    @classmethod
    def generate(cls, dataset, batch_size, sampler, pad_value, **kwargs):
        net_collate = cls(pad_value)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=net_collate,
            **kwargs
        )
        return data_loader
