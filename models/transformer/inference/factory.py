import torch

import json
import yaml
import dill


class NETModelFactory:

    def __init__(
        self,
        config_path,
        src_field_path,
        trg_field_path,
        model_class,
        device
    ):
        self.model_class = model_class
        self.config = self._load_config(config_path)
        self.src_field = self._load_field(src_field_path)
        self.trg_field = self._load_field(trg_field_path)
        self.device = device

    def _load_config(self, config_path):
        with open(config_path) as yml_file:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)
        return config

    def _load_field(self, field_path):
        with open(field_path, 'rb') as dill_f:
            field = dill.load(dill_f)
        return field

    def produce(self, model_path, eval=True):
        model = self.model_class(**self.config["model_params"],
                                 device=self.device)
        if self.config['train_params']['device_type'] != 'cpu':
            model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        else:
            model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        if eval:
            model.eval()
        return model, self.src_field, self.trg_field
