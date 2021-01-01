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
        src_tokens = [self.sos_token] + list(src_text) + [self.eos_token]
        src_seq = self.vectorizer.vectorize(src_tokens)
        ## src_seq = [src_len]
        src_seq = src_seq.unsqueeze(1).repeat(1, beam_size)
        ## src_seq = [src_len, beam_size]
        # prepare vars
        predictions = []
        for _ in range(beam_size):
            predictions.append([self.vectorizer.lookup.stoi[self.sos_token]])
        # beam search takes one argument at first
        trg_seq = torch.tensor(predictions[0]).unsqueeze(1)
        # print(trg_seq)
        ## trg_seq = [1, 1]
        # print(src_seq[:,0].shape, trg_seq.shape)
        output = self.model(
            src_seq=src_seq[:, 0].unsqueeze(1), ## src_seq = [src_len, 1]
            trg_seq=trg_seq,
            src_key_padding_mask=None,
            trg_key_padding_mask=None,
            memory_key_padding_mask=None,
            trg_mask=None
        )
        ## output = [1, 1, vocab_size]
        # print(output.shape)
        output = output.squeeze(0).squeeze(0)
        ## output = [vocab_size]
        # print(output.shape)
        values, indices = torch.topk(output, beam_size)
        ## values, indices = [beam_size]
        print(indices.shape)
        for hyp, idx in zip(predictions, indices.tolist()):
            hyp.append(idx)
        eos_idx = self.vectorizer.lookup.stoi[self.eos_token]
        trg_seq = torch.tensor(predictions).permute(1, 0)
        ## [2, beam_size]
        print(predictions)
        # start prediction
        print(self.vectorizer.textualize([str(x) for x in src_seq[:,0].tolist()], False))
        while not self.stop_prediction(predictions, max_pred_len, eos_idx):
            # output is the cond prob dist of the next word in vocab space
            print(src_seq.shape, trg_seq.shape)
            # trg_mask = self.gen_nopeek_mask(trg_seq.shape[0])
            output = self.model(
                src_seq=src_seq,
                trg_seq=trg_seq,
                src_key_padding_mask=None,
                trg_key_padding_mask=None,
                memory_key_padding_mask=None,
                trg_mask=None
            )
            # print(output.shape)
            ## output = [beam_size, curr_pred_len, vocab_size]
            cond_prob_dist = output[:, -1, :].squeeze(1)
            # print(cond_prob_dist.shape)
            ## cond_prob_dist = [beam_size, vocab_size]
            vocab_size = cond_prob_dist.shape[1]
            cond_prob_dist_concat = cond_prob_dist.contiguous().view(-1)
            ## cond_prob_dist_concat = [beam_size * vocab_size]
            values, indices = torch.topk(cond_prob_dist_concat, beam_size)
            ## values, indices = [beam_size]
            # print(values, indices)
            for hyp, idx in zip(predictions, indices.tolist()):
                # recover index
                vocab_idx = idx % vocab_size
                print(vocab_idx)
                hyp.append(vocab_idx)
            print(predictions)
            trg_seq = torch.tensor(predictions).permute(1, 0)
            ## trg_seq = [curr_trg_len, beam_size]
        # get predicitons
        results = []
        for rank, hyp in enumerate(predictions):
            _hyp = [str(x) for x in hyp][1:]
            pred = self.vectorizer.textualize(_hyp, no_special_tokens=True)
            results.append({"hyp_rank": rank + 1, "prediciton": pred})
        return results

    def stop_prediction(self, predictions, max_pred_len, eos_idx):
        if len(predictions[0]) > max_pred_len:
            return True
        elif all([hyp[-1] == eos_idx for hyp in predictions]):
            return True
        else:
            return False

    def gen_nopeek_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len)) == 1
        mask = mask.float().masked_fill(mask == 0, 1e-8).masked_fill(mask == 1, float(0.0))
        return mask
