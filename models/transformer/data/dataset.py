from .lookup import SymLookup

from torch.utils.data import Dataset
import torchtext

import logging


logger = logging.getLogger(__name__)


class TorchtextNETDataset(torchtext.data.Dataset):

  def __init__(self, lines, src_field, trg_field):
    fields = [('source', src_field), ('target', trg_field)]
    examples = []
    printed = False
    for line in lines:
        items = line.split()
        source = list(items[0].strip())
        target = items[1:]
        if not printed:
            logger.debug('source: {}\ntarget: {}'.format(source, target))
            printed = True
        examples.append(data.Example.fromlist([source, target], fields))
    self.sort_key = lambda x: len(x.source)
    super(NETDataset, self).__init__(examples, fields)

  @classmethod
  def loads(cls, train_path, val_path, src_field, trg_field):
    with open(train_path, mode="r", encoding='utf-8') as f:
        train_lines = [x for x in f]
    with open(val_path, mode="r", encoding='utf-8') as f:
        val_lines = [x for x in f]
    train_data = cls(train_lines, src_field, trg_field)
    val_data = cls(val_lines, src_field, trg_field)
    return train_data, val_data


class NETDataset(Dataset):

    def __init__(self, net_pairs):
        raise RuntimeError(
            'This class might have a bug. Use TorchtextNETDataset instead.'
        )
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
