import torch


class Beam:
    """Ordered beam of candidate outputs.
    Code borrowed from: https://github.com/MaximumEntropy/Seq2Seq-PyTorch/
    """

    def __init__(self, size, pad, bos, eos, device):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = pad
        self.bos = bos
        self.eos = eos
        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(size).fill_(self.pad).unsqueeze(1).to(device)]
        self.nextYs[0][0] = self.bos
        self.device = device

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return torch.cat(self.nextYs, dim=1)

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)
        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]
        flat_beam_lk = beam_lk.view(-1)
        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0,
                                                     True, True)
        self.scores = bestScores
        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId // num_words # torch no longer supports '/' operator
        self.prevKs.append(prev_k)
        next_y = bestScoresId - prev_k * num_words
        self.nextYs.append(next_y.unsqueeze(1))
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]


class NETransliterator:

    @torch.no_grad()
    def __call__(
        self,
        named_entity,
        max_pred_len,
        beam_size,
        model,
        src_field,
        trg_field,
        device,
        pad_idx,
        sos_idx,
        eos_idx,
        special_tokens,
        tokenize_input=True
    ):
        # set beam object
        beam = Beam(beam_size, pad_idx, sos_idx, eos_idx, device)
        # process source
        if tokenize_input:
            input_tokens = list(self.clean_input(named_entity))
        else:
            input_tokens = named_entity
        tokens = [src_field.init_token] + input_tokens + [src_field.eos_token]
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device)
        ## src_tensor = [1, src_len + 2]; two for sos and eos tokens
        src_mask = model.make_src_mask(src_tensor)
        enc_src = model.encoder(src_tensor, src_mask)
        # expand tensor for beam size
        src_mask = src_mask.repeat(beam_size, 1, 1, 1)
        ## src_mask = [beam_size, 1, 1, src_len + 2]
        enc_src = enc_src.repeat(beam_size, 1, 1)
        ## enc_src = [beam_size, src_len + 2, hid_dim]
        for i in range(max_pred_len):
            trg_tensor = beam.get_current_state()
            trg_mask = model.make_trg_mask(trg_tensor)
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            ## output = [beam_size, curr_trg_len, output_dim]
            if beam.advance(output[:, -1, :]):
                break
        predictions = []
        for hyp_idx in range(beam_size):
            hyp = beam.get_hyp(hyp_idx)
            pred = [trg_field.vocab.itos[i.item()] for i in hyp if i.item() not in special_tokens]
            predictions.append("".join(pred))
        return predictions

    def clean_input(self, word):
        # lower text
        word = word.lower()
        # remove white spaces
        word = word.replace(' ', '')
        return word

    @torch.no_grad()
    def greedy_search(
        self,
        named_entity,
        max_pred_len,
        model,
        src_field,
        trg_field,
        device
    ):
        if tokenize_input:
            input_tokens = list(self.clean_input(named_entity))
        else:
            input_tokens = named_entity
        tokens = [src_field.init_token] + input_tokens + [src_field.eos_token]
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)
        enc_src = model.encoder(src_tensor, src_mask)
        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
        for i in range(max_pred_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break
        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        return ''.join(trg_tokens[1:-1])
