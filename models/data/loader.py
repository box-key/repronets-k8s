from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


class NETDataLoader:

    def __init__(self, pad_value):
        self.pad_value = pad_value

    def __call__(self, batch):
        source, target = [], []
        # decompose batch into sources and targets
        for x in batch:
            source.append(x["source"])
            target.append(x["target"])
        src_padded, src_len, src_pad_mask = self._pad(source)
        trg_padded, trg_len, trg_pad_mask = self._pad(target)
        memory_key_padding_mask = src_pad_mask.clone()
        trg_mask = self._gen_nopeek_mask(trg_padded.shape[0])
        return (src_padded,
                src_pad_mask,
                trg_padded,
                trg_pad_mask,
                memory_key_padding_mask,
                trg_mask)

    def _gen_nopeek_mask(self, seq_len):
        """Source: https://andrewpeng.dev/transformer-pytorch/"""
        mask = torch.triu(torch.ones(seq_len, seq_len)) == 1
        mask = mask.float().masked_fill(mask == 0, 1e-8).masked_fill(mask == 1, float(0.0))
        return mask

    def _pad(self, tensors):
        lengths = torch.tensor([t.shape[0] for t in tensors])
        ## lengths = [batch_size]
        max_len = lengths.max().item()
        masks = []
        for seq_len in lengths:
            mask = [False for _ in range(seq_len)] + [True for _ in range(max_len - seq_len)]
            masks.append(mask)
        masks = torch.tensor(masks, dtype=torch.bool)
        ## masks = [batch_size]
        padded = pad_sequence(tensors, padding_value=self.pad_value)
        ## padded = [seq_len, batch_size]
        return padded, lengths, masks

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
