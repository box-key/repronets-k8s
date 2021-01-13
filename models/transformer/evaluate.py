from inference.transliterator import NETransliterator
from inference.factory import NETModelFactory

from modules.seq2seq import Seq2Seq

import torch

from pathlib import Path
from collections import Counter
from tqdm import tqdm
import click
import logging


def compute_accuracy(results, beam_size):
    init_counter = {}
    for i in range(beam_size):
        init_counter[i + 1] = 0
    match_counter = Counter(init_counter)
    for result in results:
        for bi, pred in enumerate(result['predictions']):
            if pred == result['truth']:
                match_counter[bi + 1] += 1
                break
    total_matches = 0
    accuracies = {}
    for bi in range(beam_size):
        total_matches += match_counter[bi + 1]
        n_best = "{}best".format(bi + 1)
        accuracies[n_best] = total_matches / len(results)
    return accuracies


@click.command()
@click.option("-b", "--beam", type=click.INT, default=3)
@click.option(
    "-t", "--device-type",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    required=True
)
@click.option(
    "-n", "--num-print",
    type=click.INT, default=10,
    help="The number of predictions to print per model."
)
@click.option(
    "-f", "--file-path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
    help="Path to test file."
)
@click.option(
    "-d", "--model-dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=True),
    required=False,
    help=("Path to model directory. It tests all files with '.pt' extension. "
          "The directory must have 'config.yml', 'src_field.dill' and "
          "'trg_field.dill'.")
)
@click.option(
    "-m", "--model-path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=False,
    help=("Path to model file, whose extension must be '.pt'. Its parent "
          "directory must have 'config.yml', 'src_field.dill' and "
          "'trg_field.dill'.")
)
def evaluate(**kwargs):
    # config logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    # get path to necessary fiels
    if 'model_dir' in kwargs:
        dir_path = Path(kwargs['model_dir'])
        config_path = dir_path / 'config.yml'
        src_field_path = dir_path / 'src_field.dill'
        trg_field_path = dir_path / 'trg_field.dill'
        models = [x for x in dir_path.glob('**/*.pt')]
        logger.info("Test {} models in dir='{}'".format(len(models), dir_path))
    elif 'model_path' in kwargs:
        model_path = Path(kwargs["model_path"])
        dir_path = model_path.resolve().parent
        config_path = dir_path / 'config.yml'
        src_field_path = dir_path / 'src_field.dill'
        trg_field_path = dir_path / 'trg_field.dill'
        models = [model_path]
        logger.info("Test model='{}'".format(model_path))
    else:
        raise RuntimeError("You must specify either 'model_dir' or 'model_path'")
    # set model factory
    device = torch.device(kwargs["device_type"])
    factory = NETModelFactory(config_path=config_path,
                              src_field_path=src_field_path,
                              trg_field_path=trg_field_path,
                              model_class=Seq2Seq,
                              device=device)
    # prepare dataset
    examples = []
    source_words = set()
    with open(kwargs['file_path'], mode="r", encoding="utf-8") as f:
        for line in f:
            # extract source and target tokens
            items = line.split()
            source = list(items[0].strip())
            target = items[1:]
            # store an example
            source_word = ''.join(source)
            if source_word not in source_words:
                examples.append({"source": source, "target": target})
                source_words.add(source_word)
    logger.info("Loaded {} examples from {}".format(len(examples), kwargs['file_path']))
    # start evaluating models
    transliterator = NETransliterator()
    best_accuracy = 0
    best_model = {}
    for model_path in models:
        logger.info("Testing model='{}'".format(model_path))
        model, src_field, trg_field = factory.produce(model_path, eval=True)
        pad_idx = trg_field.vocab.stoi[trg_field.pad_token]
        sos_idx = trg_field.vocab.stoi[trg_field.init_token]
        eos_idx = trg_field.vocab.stoi[trg_field.eos_token]
        special_tokens_idxs = set([pad_idx, sos_idx, eos_idx])
        results = []
        for i, example in tqdm(enumerate(examples), initial=kwargs['num_print']):
            predictions = transliterator(named_entity=example["source"],
                                         max_pred_len=32,
                                         beam_size=kwargs["beam"],
                                         model=model,
                                         src_field=src_field,
                                         trg_field=trg_field,
                                         device=device,
                                         pad_idx=pad_idx,
                                         sos_idx=sos_idx,
                                         eos_idx=eos_idx,
                                         special_tokens=special_tokens_idxs,
                                         tokenize_input=False)
            # print(predictions[0], type(predictions[0]), example['target'])
            if i < kwargs["num_print"]:
                logger.info("Source input: {}".format(''.join(example["source"])))
                for b, p in enumerate(predictions):
                    logger.info("Beam {}: {}".format(b, p))
            results.append({
                "source": example["source"],
                "predictions": predictions,
                "truth": ''.join(example["target"])
            })
        accuracies = compute_accuracy(results, kwargs["beam"])
        logger.info("Accuracy:\n{}".format(accuracies))
        bottom_beam = "{}best".format(kwargs["beam"])
        if best_accuracy < accuracies[bottom_beam]:
            best_model= {"accuracy": accuracies, "model": model_path}
    logger.info("Result:\n{}".format(best_model))


if __name__ == "__main__":
    evaluate()
