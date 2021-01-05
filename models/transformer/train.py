from modules.seq2seq import Seq2Seq
from modules.trainer import NETTrainer

from data.dataset import TorchtextNETDataset

from torchtext import data

import torch.nn as nn
import torch

from pathlib import Path
import click
import os
import logging
import yaml
import dill


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
@click.option("--trg-separator", type=click.STRING, default=" ", required=True)
@click.option("--learning-rate", type=click.FLOAT, default=0.0005)
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
        # load dataset
        src_field = data.Field(eos_token='<eos>', sequential=True)
        trg_field = data.Field(eos_token='<eos>', sequential=True)
        train_data, val_data = TorchtextNETDataset.loads(kwargs["train_path"],
                                                         kwargs["val_path"],
                                                         src_field,
                                                         trg_field)
        src_field.build_vocab(train_data, val_data)
        trg_field.build_vocab(train_data, val_data)
        logger.info("Source vocab size = {}".format(len(src_field.vocab)))
        logger.info("Target vocab size = {}".format(len(trg_field.vocab)))
        logger.info("Trainig set size = {}".format(len(train_data)))
        logger.info("Validation set size = {}".format(len(val_data)))
        # make iterator
        device = torch.device(kwargs['device'])
        train_iter = data.BucketIterator(train_data,
                                         batch_size=kwargs["batch_size"],
                                         repeat=False,
                                         train=True,
                                         device=device)
        val_iter = data.BucketIterator(val_data,
                                       batch_size=kwargs["batch_size"],
                                       train=False,
                                       device=device)
        # init model
        model_params = {
            "input_dim": len(src_field.vocab),
            "output_dim": len(trg_field.vocab),
            "hid_dim": kwargs['hid_dim'],
            "n_layers": kwargs['n_layers'],
            "n_heads": kwargs['n_heads'],
            "pf_dim": kwargs['max_seq_len'],
            "dropout": kwargs['dropout'],
            "max_seq_len": kwargs['max_seq_len'],
            "src_pad_idx": src_field.vocab.stoi[src_field.pad_token],
            "trg_pad_idx": trg_field.vocab.stoi[trg_field.pad_token]
        }
        model = Seq2Seq(**model_params, device=device).to(device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=model_params["trg_pad_idx"])
        # init trainer
        trainer = NETTrainer(
            model=model,
            train_iter=train_iter,
            val_iter=val_iter,
            optimizer=torch.optim.Adam,
            learning_rate=kwargs["learning_rate"],
            device=device,
            loss_fn=loss_fn
        )
        # save dependency files before training
        with open(kwargs["model_dir"] / "src_field.dill", "wb") as dill_f:
            dill.dump(src_field, dill_f)
        with open(kwargs["model_dir"] / "trg_field.dill", "wb") as dill_f:
            dill.dump(trg_field, dill_f)
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
        logger.info(
            "Saved config and field objects at '{}'".format(kwargs["model_dir"])
        )
        best_loss = trainer.epoch(n_epochs=kwargs["epochs"],
                                  clip=kwargs["clip"],
                                  model_dir=kwargs["model_dir"],
                                  checkpoint=kwargs["checkpoint"])
        logger.info("Best_loss = {:.3f}".format(best_loss))
    else:
        raise RuntimeError("`train-path` or `val-path` doesn't exist")


if __name__ == "__main__":
    train()
