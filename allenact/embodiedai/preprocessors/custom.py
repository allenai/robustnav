from typing import List, Callable, Optional, Any, cast, Dict

import os
import gym
import torch
import inspect
import importlib
import numpy as np
from torch import nn as nn
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.main import find_sub_modules

def load_model(model_name: str, encoder_base: str):
    assert os.path.exists(
        encoder_base
    ), "The path '{}' does not seem to exist (your current working directory is '{}').".format(
        encoder_base, os.getcwd()
    )
    rel_base_dir = os.path.relpath(  # Normalizing string representation of path
        os.path.abspath(encoder_base), os.getcwd()
    )
    rel_base_dot_path = rel_base_dir.replace("/", ".")
    if rel_base_dot_path == ".":
        rel_base_dot_path = ""

    exp_dot_path = model_name
    if exp_dot_path[-3:] == ".py":
        exp_dot_path = exp_dot_path[:-3]
    exp_dot_path = exp_dot_path.replace("/", ".")

    module_path = (
        f"{rel_base_dot_path}.{exp_dot_path}"
        if len(rel_base_dot_path) != 0
        else exp_dot_path
    )

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
            raise e
        all_sub_modules = set(find_sub_modules(os.getcwd()))
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            "Could not import state encoder model '{}', are you sure this is the right path?"
            " Possibly relevant files include {}.".format(
                module_path, relevant_submodules
            ),
        ) from e

    models = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], nn.Module) and m[0] == model_name
    ]
    
    assert (
        len(models) == 1
    ), "There should only be one model with name {} in {}".format(model_name, module_path)

    return models[0]

class CustomPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model."""

    def __init__(
        self,
        model_name: str,
        encoder_base: str,
        latent_size: int,
        input_uuids: List[str],
        output_uuid: str,
        input_height: int,
        input_width: int,
        ckpt_path: str = None,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any
    ):
        def f(x, k):
            assert k in x, "{} must be set in CustomPreprocessor".format(k)
            return x[k]

        def optf(x, k, default):
            return x[k] if k in x else default

        self.input_height = input_height
        self.input_width = input_width
        self.model = load_model(model_name, encoder_base)(content_latent_size = latent_size)
        self.model_name = model_name
        self.shape = (1, latent_size)

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            
            for p in self.model.parameters():
                p.requires_grad = False

        low = -np.inf
        high = np.inf

        assert (
            len(input_uuids) == 1
        ), "custom preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=self.shape)

        super().__init__(**prepare_locals_for_super(locals()))

    def to(self, device: torch.device) -> str:
        self.model = self.model.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        # print(obs)
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        if 'AE' in self.model_name:
            return self.model(x.to(self.device))[1]
        elif 'VAE' in self.model_name or 'LUSR' in self.model_name or 'DDVAE' in self.model_name:
            return self.model(x.to(self.device), return_latent=True)[3]
        elif 'DARLA' in self.model_name:
            return self.model(x.to(self.device))[3]
        else:
            raise ValueError(f'None of the known model names in {self.model_name}')
