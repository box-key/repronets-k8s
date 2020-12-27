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
        src_padded = batch[0].to(self.device)
        ## src_padded = [src_len, batch_size]
        src_pad_mask = batch[1].to(self.device)
        ## src_pad_mask = [batch_size, src_len]
        trg_padded = batch[2].to(self.device)
        ## trg_padded = [trg_len, batch_size]
        trg_pad_mask = batch[3].to(self.device)
        ## trg_pad_mask = [batch_size, trg_len]
        memory_key_padding_mask = batch[4].to(self.device)
        ## memory_key_padding_mask = [trg_len, src_len]
        trg_mask = batch[5].to(self.device)
        ## trg_mask = [trg_len, trg_len]
        output = self.model(
            src_seq=src_padded,
            trg_seq=trg_padded,
            src_key_padding_mask=src_pad_mask,
            trg_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            trg_mask=trg_mask
        )
        ## output = [batch_size, trg_len, vocab_size]
        output = output.permute(1, 0, 2)
        # utils.print_shape(output=output, trg_out=trg_padded[1:, :])
        ## output = [trg_len, batch_size, vocab_size]
        trg_out = trg_padded[1:, :]
        output = output[1:, :, :]
        # utils.print_shape(output=output, trg_out=trg_out)
        # trg_out = trg_padded
        ## trg_out = [trg_len, batch_size]
        # print(output[0, :])
        # print("con:\n{}".format(output.contiguous()))
        output = output.contiguous().view(-1, output.shape[2])
        trg_out = trg_out.contiguous().view(-1)
        ## output = [batch_size * trg_len, output_dim]
        ## trg_out = [batch_size * trg_len]
        # utils.print_shape(trg_out=trg_out, output=output)
        # print("conout:\n{}".format(output))
        # print(output
        return self.loss_fn(output, trg_out)

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
