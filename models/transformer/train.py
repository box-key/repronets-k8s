from data.dataset import NETDataset
from data.loader import NETDataLoader
from data.lookup import SymLookup
from data.vectorizer import NETVectorizer

from modules.net_transformer import NETTransformer
from modules.test import Encoder, Decoder, Seq2Seq

from modules.trainer import NETTrainer

from torch.utils.data import RandomSampler
import torch.nn as nn
import torch

from pathlib import Path
import click
import os
import logging
import yaml


def get_loader(dataset, batch_size, num_workers, pad_value):
    sampler = RandomSampler(dataset)
    iter = NETDataLoader.generate(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  pad_value=pad_value,
                                  num_workers=num_workers)
    return sampler, iter


def save_config(output_path, model_name, model_params, train_params):
    config = {
        "model_name": model_name,
        "model_params": model_params,
        "train_params": train_params
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f)


@click.command()
@click.option("--batch-size", type=click.INT, default=64)
@click.option("--learning-rate", type=click.FLOAT, default=0.005)
@click.option("--dropout", type=click.FLOAT, default=0.2)
@click.option("--n-heads", type=click.INT, default=8)
@click.option("--num-workers", type=click.INT, default=4)
@click.option("--max-seq-len", type=click.INT, default=64)
@click.option("--hid-dim", type=click.INT, default=256)
@click.option("--n-layers", type=click.INT, default=8)
@click.option("--epochs", type=click.INT, default=10)
@click.option("--checkpoint", type=click.INT, default=4)
@click.option("--clip", type=click.FLOAT, default=1.0)
@click.option(
    "--train-path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True
)
@click.option(
    "--val-path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=False),
    required=True
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    required=True
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning"], case_sensitive=False),
    required=True
)
def train(**kwargs):
    # print(kwargs.keys())
    # config logger
    if kwargs["log_level"] == "debug":
        log_level = logging.DEBUG
    elif kwargs["log_level"] == "info":
        log_level = logging.INFO
    elif kwargs["log_level"] == "warning":
        log_level = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=log_level
    )
    logger = logging.getLogger(__name__)
    # convert input to pathlib object
    kwargs["model_dir"] = Path(kwargs["model_dir"])
    kwargs["train_path"] = Path(kwargs["train_path"])
    kwargs["val_path"] = Path(kwargs["val_path"])
    # create model_dir if it doesn't exist
    if not kwargs["model_dir"].exists():
        kwargs["model_dir"].mkdir()
    else:
        click.confirm(
            "Path '{}' already exists. Continue?".format(kwargs["model_dir"]),
            abort=True
        )
    if kwargs["train_path"].exists() and kwargs["val_path"].exists():
        src_lookup, trg_lookup = SymLookup.build(kwargs["train_path"])
        logger.info("Build src vocab with {} elements".format(len(src_lookup)))
        logger.info("Build trg vocab with {} elements".format(len(trg_lookup)))
        src_vectorizer = NETVectorizer(src_lookup)
        trg_vectorizer = NETVectorizer(trg_lookup)
        PAD_IDX = src_lookup.stoi[src_lookup.pad_token]
        logger.debug("Pad value = '{}'".format(PAD_IDX))
        # create train loader
        train_set = NETDataset.load(kwargs["train_path"],
                                    src_vectorizer,
                                    trg_vectorizer,
                                    kwargs["trg_separator"])
        train_sampler, train_iter = get_loader(train_set,
                                               kwargs["batch_size"],
                                               1,
                                               PAD_IDX)
                                               # kwargs["num_workers"])
        logger.info("Trainig set size = {}".format(len(train_set)))
        # create val loader
        val_set = NETDataset.load(kwargs["val_path"],
                                  src_vectorizer,
                                  trg_vectorizer,
                                  kwargs["trg_separator"])
        val_sampler, val_iter = get_loader(val_set,
                                           kwargs["batch_size"],
                                           1,
                                           PAD_IDX)
                                           # kwargs["num_workers"])
        logger.info("Validation set size = {}".format(len(val_set)))
        # init model
        device = torch.device(kwargs["device"])
        model_params = {
            "input_dim": len(src_lookup),
            "output_dim": len(trg_lookup),
            "hid_dim": kwargs['hid_dim'],
            "n_layers": kwargs['n_layers'],
            "n_heads": kwargs['n_heads'],
            "pf_dim": kwargs['max_seq_len'],
            "dropout": kwargs['dropout'],
            "max_seq_len": kwargs['max_seq_len'],
            "src_pad_idx": PAD_IDX,
            "trg_pad_idx": PAD_IDX,
        }
        model = Seq2Seq(**model_params, device=device).to(device)
        # init trainer
        trainer = NETTrainer(
            model=model,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=torch.optim.Adam,
            learning_rate=kwargs["learning_rate"],
            device=device,
            train_sampler=train_sampler,
            loss_fn=nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        )
        # save dependency files before training
        src_lookup.save(kwargs["model_dir"] / "src_vocab.json")
        trg_lookup.save(kwargs["model_dir"] / "trg_vocab.json")
        train_params = {
            "epochs": kwargs["epochs"],
            "lr": kwargs["learning_rate"],
            "clip": kwargs["clip"],
            "batch_size": kwargs["batch_size"],
            "train_path": str(kwargs["train_path"].absolute()),
            "val_path": str(kwargs["val_path"].absolute()),
            "device_type": kwargs["device"],
            "checkpoint": kwargs["checkpoint"],
            "optim": "Adam",
            "loss": "CrossEntropyLoss"
        }
        save_config(kwargs["model_dir"] / "config.yml",
                    "net-transofrmer",
                    model_params,
                    train_params)
        logger.debug("Saved config and vocabulary at '{}'".format(kwargs["model_dir"]))
        best_loss = trainer.epoch(n_epochs=kwargs["epochs"],
                                  clip=kwargs["clip"],
                                  model_dir=kwargs["model_dir"],
                                  checkpoint=kwargs["checkpoint"])
        logger.info("Best_loss = {:.3f}".format(best_loss))
    else:
        raise RuntimeError("`train-path` or `val-path` doesn't exist")


if __name__ == "__main__":
    train()
