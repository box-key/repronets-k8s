import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import os
import logging

from . import utils


logger = logging.getLogger(__name__)


def _default_init_weights(model):
    for name, param in model.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.01)


class NETTrainer:

    def __init__(
        self,
        model,
        train_iter,
        val_iter,
        optimizer,
        learning_rate,
        device,
        loss_fn,
        weight_initializer=None
    ):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.loss_fn = loss_fn
        if weight_initializer is None:
            self.weight_initializer = _default_init_weights
        else:
            self.weight_initializer = weight_initializer
        self.init_model()
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.device = device

    def init_model(self):
        logger.debug("Initializing weights")
        self.model.apply(self.weight_initializer)

    def train(self, clip):
        self.model.train()
        epoch_loss = 0
        for batch in tqdm(self.train_iter, total=len(self.train_iter), unit="batch"):
            src = batch.source.permute(1, 0)
            ## src = [batch_size, src_len]
            trg = batch.target.permute(1, 0)
            ## trg = [batch_size, src_len]
            # reset gradients
            self.optimizer.zero_grad()
            # compute output
            output, _ = self.model(src, trg[:, :-1])
            ## output = [batch_size, trg_len, output_dim]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            ## output = [batch_size * trg_len - 1, output_dim]
            ## trg = [batch_size * trg_len - 1]
            loss = self.loss_fn(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            epoch_loss += loss.item()
            logger.debug("batch loss = {}".format(loss.item()))
        return epoch_loss / len(self.train_iter)

    @torch.no_grad()
    def evaluate(self, iterator):
        self.model.eval()
        epoch_loss = 0
        for batch in tqdm(iterator, total=len(iterator), unit="batch"):
            src = batch.source.permute(1, 0)
            ## src = [batch_size, src_len]
            trg = batch.target.permute(1, 0)
            ## trg = [batch_size, src_len]
            output, _ = self.model(src, trg[:, :-1])
            ## output = [batch_size, trg_len, output_dim]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            ## output = [batch_size * trg_len - 1, output_dim]
            ## trg = [batch_size * trg_len - 1]
            loss = self.loss_fn(output, trg)
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def epoch(self, n_epochs, clip, model_dir, checkpoint):
        best_loss = float("inf")
        for epoch in range(1, n_epochs+1):
            # must update epoch if sampler is DistributedSampler
            train_loss = self.train(clip)
            val_loss = self.evaluate(self.val_iter)
            if epoch % checkpoint == 0:
                model_name = os.path.join(model_dir, str(epoch) + '.pt')
                torch.save(self.model.state_dict(), model_name)
            if val_loss < best_loss:
                best_loss = val_loss
            logger.info("{} | Epoch: {} | Train Loss: {:.3f}".format(
                self.device, epoch, train_loss
            ))
            logger.info("{} | Epoch: {} | Validation Loss: {:.3f}".format(
                self.device, epoch, val_loss
            ))
        return best_loss
