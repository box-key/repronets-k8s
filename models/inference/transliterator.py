import torch


class NETransliterator:

    def __init__(self, vectorizer, device, model):
        self.model = model
        self.device = device
        self.vectorizer = vectorizer
        self.eos_token = vectorizer.lookup.eos_token
        self.sos_token = vectorizer.lookup.sos_token

    @torch.no_grad()
    def predict(self, src_text, beam_size, max_pred_len):
        # vectorize source token
        src_tokens = list(self.sos_token + src_text + self.eos_token)
        src_seq = self.vectorizer.vectorize(src_tokens)
        ## src_seq = [src_len]
        src_seq = src_seq.unsqueeze(1)
        ## src_seq = [src_len, 1]
        # prepare vars
        prediction = [self.vectorizer.lookup.stoi[self.sos_token]]
        trg_seq = torch.tensor(prediction)
        ## trg_seq = [1]
        trg_seq = trg_seq.unsqueeze(1)
        ## trg_seq = [1, 1]
        eos_idx = self.vectorizer.lookup.stoi[self.eos_token]
        # start prediction
        while (prediction[-1] != eos_idx) and (len(prediction) < max_pred_len):
            # output is the cond prob dist of the next word in vocab space
            # print(src_seq.shape, trg_seq.shape)
            # trg_mask = self.gen_nopeek_mask(trg_seq.shape[0])
            output = self.model(
                src_seq=src_seq,
                trg_seq=trg_seq,
                src_key_padding_mask=None,
                trg_key_padding_mask=None,
                memory_key_padding_mask=None,
                trg_mask=None
            )
            print(output.shape)
            ## output = [1, curr_pred_len, vocab_size]
            cond_prob_dist = output.squeeze(0)[-1, :].squeeze(0)
            # print(cond_prob_dist.shape)
            ## cond_prob_dist = [vocab_size]
            values, indices = torch.topk(cond_prob_dist, beam_size)
            ## values, indices = [beam_size]
            print(indices, values)
            prediction.append(indices[0].item())
            trg_seq = torch.tensor(prediction).unsqueeze(1)
        # convert intergers into str
        prediction = [str(x) for x in prediction][1:]
        return self.vectorizer.textualize(prediction, no_special_tokens=True)

    def gen_nopeek_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len)) == 1
        mask = mask.float().masked_fill(mask == 0, 1e-8).masked_fill(mask == 1, float(0.0))
        return mask
