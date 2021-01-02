import torch


class NETransliterator:

    @torch.no_grad()
    def __call__(self, src_text, max_pred_len, model, vectorizer, device):
        sos_token = vectorizer.lookup.sos_token
        eos_token = vectorizer.lookup.eos_token
        sos_idx = vectorizer.lookup.stoi[sos_token]
        eos_idx = vectorizer.lookup.stoi[eos_token]
        src_tokens = [sos_token] + list(src_text.lower()) + [eos_token]
        print(src_tokens)
        src_tensor = vectorizer.vectorize(src_tokens).unsqueeze(0)
        print(src_tensor)
        ## src_tensor = [1, src_len]
        src_mask = model.make_src_mask(src_tensor)
        ## src_mask = [1, 1, 1, src_len]
        enc_src = model.encoder(src_tensor, src_mask)
        ## enc_src = [1, src_len, hid_dim]
        predictions = [sos_idx]
        # start generation
        # print(src_tensor.shape, src_mask.shape, enc_src.shape)
        while len(predictions) < max_pred_len:
            trg_tensor = torch.tensor(predictions, dtype=torch.long).unsqueeze(0).to(device)
            ## trg_tensor = [1, curr_trg_len]
            trg_mask = model.make_trg_mask(trg_tensor)
            ## trg_mask = [1, 1, curr_trg_len, curr_trg_len]
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            ## output = [1, curr_trg_len, output dim]
            ## attention = [1, n_heads, curr_trg_len, src_len]
            next_pred_token = output.argmax(2)[:, -1].item()
            # print(trg_tensor.shape, trg_mask.shape, output.shape)
            # print(next_pred_token)
            predictions.append(next_pred_token)
            if next_pred_token == eos_idx:
                break
        print(predictions)
        predictions = vectorizer.textualize(predictions, no_special_tokens=True)
        return predictions, attention
