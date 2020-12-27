import torch
import torch.nn as nn

import math

from . import utils

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NETTransformer(nn.Module):

    def __init__(
        self,
        d_model,
        vocab_size,
        max_seq_len,
        n_head,
        n_enc_layers,
        n_dec_layers,
        trans_dropout,
        pos_dropout
    ):
        super().__init__()
        self.emb_src = nn.Embedding(vocab_size, d_model)
        self.emb_trg = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_len)
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=n_head,
                                          num_encoder_layers=n_enc_layers,
                                          num_decoder_layers=n_dec_layers,
                                          dropout=trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(
        self,
        src_seq,
        trg_seq,
        src_key_padding_mask,
        trg_key_padding_mask,
        memory_key_padding_mask,
        trg_mask
    ):
        ## src_seq = [src_len, batch_size]
        ## trg_seq = [trg_len, batch_size]
        ## src_key_padding_mask = [batch_size, src_len]
        ## trg_key_padding_mask = [batch_size, trg_len]
        ## memory_key_padding_mask = [batch_size, src_len]
        ## trg_mask = [trg_len, trg_len]
        # utils.print_shape(trg_seq=trg_seq, src_seq=src_seq)
        # print("seq:\n{}".format(src_seq[:,0]))
        # print("emb:\n{}".format(self.emb_src(src_seq)[:,0,:]))
        src_emb = self.pos_enc(self.emb_src(src_seq) * math.sqrt(self.d_model))
        ## src_emb = [src_len, batch_size, d_model]
        trg_emb = self.pos_enc(self.emb_trg(trg_seq) * math.sqrt(self.d_model))
        ## trg_emb = [trg_len, batch_size, d_model]
        # print("pos:\n{}".format(src_emb[:,0,:]))
        # utils.print_shape(trg_emb=trg_emb, src_emb=src_emb)
        output = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # print("tout:\n{}".format(output[:,0,:]))
        ## output = [trg_len, batch_size, d_model]
        output = output.permute(1, 0, 2)
        ## output = [batch_size, trg_len, d_model]
        output = self.fc(output)
        ## output = [batch_size, trg_len, vocab_size]
        # print("fcout:\n{}".format(output[0,:,:]))
        return output
