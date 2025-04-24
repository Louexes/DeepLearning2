# https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import numpy as np
import torch
import torch.jit
import torch.nn as nn


class Tent_ext(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, protector, steps=1, episodic=False, slope_threshold=0.01):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        # TODO: Make sure the model setup works correctly with the protector
        self.protector = protector
        # Threshold value for adaptation based on martingale slope
        self.slope_threshold = slope_threshold

    # Forward pass structured like POEM
    def forward(self, x):
        if self.episodic:
            self.reset()

        # Decision to adapt based on POEM protector and Martingale slope
        # 1. Retrieve protejected entropies
        with torch.no_grad():
            outputs = self.model(x)
            ents = softmax_entropy(outputs).view(-1, 1)
            # # 1.
            protected_ents, protection_info = self.protector.protect(ents)

        # 2. Compute slope of martingale process
        slope = self.martingale_slope(protected_ents)
        #print(f"Slope: {slope:.4f}")	
        # TODO: May need to insert logic for delayed start? See POEM class.

        # 3. Decide to adapt or not based on slope
        if np.abs(slope) >= self.slope_threshold:
            #print(f"Adapting on slope: {slope:.4f}")
            for _ in range(self.steps):
                outputs = forward_and_adapt(x, self.model, self.optimizer)
            if self.episodic:
                outputs = self.model(x)

        return outputs

    # This mostly based on run_martingale from utilities
    def martingale_slope(self, entropies):
        """Compute the slope of the martingale process."""
        self.protector.reset()
        logs = list()
        for z in entropies:
            # Move tensor to CPU before using with numpy
            z_cpu = z.cpu() if isinstance(z, torch.Tensor) else z
            u = self.protector.cdf(z_cpu)
            self.protector.protect_u(u)
            # TODO: Log Sj or Sj?
            logs.append(self.protector.martingales[-1] + 1e-8)
            # logs.append(np.log(self.protector.martingales[-1] + 1e-8))

        # TODO: Not sure if this will work, but it's a start
        # Doing the mean here so it can just deal with batches
        slope = np.mean((np.gradient(logs, 1)))
        return slope

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:  # weight is scale, bias is shift
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
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: check which require grad"
    assert not has_all_params, "tent should not update all params: check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
