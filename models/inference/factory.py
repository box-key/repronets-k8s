import torch

import json
import yaml


class NETModelFactory:

    def __init__(
        self,
        config_path,
        vocab_path,
        model_class,
        vectorizer_class,
        lookup_loader,
        device
    ):
        self.model_class = model_class
        self.config = self._load_config(config_path)
        self.vectorizer = vectorizer_class(lookup_loader(vocab_path))
        self.device = device

    def _load_config(self, config_path):
        with open(config_path) as yml_file:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)
        return config

    def produce(self, model_path, eval=True):
        model = self.model_class(**self.config["model_params"])
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        if eval:
            model.eval()
        return model, self.vectorizer
