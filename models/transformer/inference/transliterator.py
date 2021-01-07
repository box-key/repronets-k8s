import torch


class NETransliterator:

    @torch.no_grad()
    def __call__(
        self,
        named_entity,
        max_pred_len,
        model,
        src_field,
        trg_field,
        device
    ):
        tokens = [src_field.init_token] + list(named_entity.lower()) + [src_field.eos_token]
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
        return trg_tokens[1:-1]
