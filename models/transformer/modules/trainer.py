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
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)

class NETTrainer:

    def __init__(
        self,
        model,
        train_iter,
        val_iter,
        optimizer,
        learning_rate,
        device,
        train_sampler,
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
        self.train_sampler = train_sampler

    def init_model(self):
        self.model.apply(self.weight_initializer)

    def train(self, clip):
        self.model.train()
        epoch_loss = 0
        for batch in tqdm(self.train_iter, total=len(self.train_iter), unit="batch"):
            # reset gradients
            self.optimizer.zero_grad()
            loss = self.get_loss(batch)
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
            loss = self.get_loss(batch)
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def get_loss(self, batch):
        src_padded = batch['src_padded'].permute(1, 0)
        ## src_padded = [batch_size, src_len]
        trg_padded = batch['trg_padded'].permute(1, 0)
        ## trg_padded = [batch_size, trg_len]
        output, _ = self.model(src=src_padded, trg=trg_padded[:, :-1])
        ## output = [batch_size, trg_len - 1, vocab_size]
        output_dim = output.shape[2]
        output = output.contiguous().view(-1, output_dim)
        ## output = [batch size * trg len - 1, output dim]
        trg_padded = trg_padded[:, 1:].contiguous().view(-1)
        # trg = [batch size * trg len - 1]
        # utils.print_shape(trg_padded=trg_padded, output=output)
        return self.loss_fn(output, trg_padded)

    def epoch(self, n_epochs, clip, model_dir, checkpoint):
        best_loss = float("inf")
        for epoch in range(1, n_epochs+1):
            # must update epoch if sampler is DistributedSampler
            if isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
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
