from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from math import sqrt
import math
import pickle
from loguru import logger
import torch.nn.functional as F

from protector import Protector

import matplotlib.pyplot as plt
# from exp_utils.viz_utils import *


class POEM(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, protector, steps=1, episodic=False, e0=math.log(1000)*0.40, adapt=True, vanilla_loss=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.protector = protector
        self.steps = steps
        self.losses_before = []
        self.losses_after = []

        self.e0 = e0

        assert steps > 0, "poem requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.orig_lr = self.optimizer.param_groups[0]['lr']
        self.delayed_start = 100
        self.curr_n_samples = 0
        self.adapt = adapt
        self.vanilla_loss = vanilla_loss

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        with torch.no_grad():
            outputs = self.model(x)
            ents = softmax_entropy(outputs).view(-1, 1)

            # OUR ADDITION
            protected_ents, protection_info = self.protector.protect(ents)


        self.curr_n_samples += x.shape[0]
        if self.curr_n_samples < self.delayed_start:
            return outputs

        if self.adapt:
            for _ in range(self.steps):
                # print(f"step {_}")
                outputs, loss = forward_and_adapt(x, self.model, self.optimizer, protected_ents, e_margin=self.e0, vanilla_loss=self.vanilla_loss)
            if self.episodic:
                outputs = self.model(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)




@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, protected_ents, e_margin: float = math.log(1000)*0.40, vanilla_loss = False):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """

    # forward
    outputs = model(x)

    # adapt
    ents = softmax_entropy(outputs)
    protected_ents = protected_ents.view(-1)


    if vanilla_loss:
        filter_ents = torch.where(ents < e_margin)
        ents = ents[filter_ents]
        protected_ents = protected_ents[filter_ents]

        # see equation (9) in the paper
        coeff = 1 / (torch.exp(ents.clone().detach() - e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        ents = ents.mul(coeff) # reweight entropy losses for diff. samples
        protected_ents = protected_ents.mul(coeff)

    # l-match equation (2) and (9)
    loss = 0.5 * torch.nn.functional.mse_loss(ents.view(-1), protected_ents.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs, loss




# def load_protector_from_path(path, **kwargs):
#     with open(path, 'rb') as file:
#         # Call load method to deserialze
#         protector = pickle.load(file)
#
#     protector.reset()
#
#     gamma = kwargs.get('gamma', 1 / (4 * sqrt(3)))
#     eps_clip_val = kwargs.get('eps_clip_val', 0.95)
#
#     logger.info(f"loading protector with {gamma=:.4f} and {eps_clip_val=}")
#
#     if gamma:
#         protector.set_gamma(gamma)
#     if eps_clip_val:
#         protector.set_eps_clip_val(eps_clip_val)
#
#     return protector


# def collect_params(model):
#     """Collect the affine scale + shift parameters from batch norms.
#
#     Walk the model's modules and collect all batch normalization parameters.
#     Return the parameters and their names.
#
#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
#     return params, names


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
