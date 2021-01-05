from torchtext import data
import logging


logger = logging.getLogger(__name__)


class TorchtextNETDataset(data.Dataset):

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
    super(TorchtextNETDataset, self).__init__(examples, fields)

  @classmethod
  def loads(cls, train_path, val_path, src_field, trg_field):
    with open(train_path, mode="r", encoding='utf-8') as f:
        train_lines = [x for x in f]
    with open(val_path, mode="r", encoding='utf-8') as f:
        val_lines = [x for x in f]
    train_data = cls(train_lines, src_field, trg_field)
    val_data = cls(val_lines, src_field, trg_field)
    return train_data, val_data
