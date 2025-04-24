from math import sqrt

import numpy as np
import torch
from loguru import logger

from cdf import CDF

# from exp_utils.utils import get_ents_acc


class Protector:
    def __init__(self, cdf, device, **kwargs):
        self.cdf = cdf
        self.device = device

        self.C = 1.0
        self.D = kwargs.get("eps_clip_val", 1.80)
        self.gamma = kwargs.get("gamma", 2 / sqrt(3))

        self.martingales = []
        self.gradients = []
        self.epsilons = [0]

        self.info = {
            "u_before": [],
            "u_after": [],
            "z_before": [],
            "z_after": [],
            "last_batch_u_before": [],
            "last_batch_u_after": [],
        }

    @property
    def last_eps(self):
        return self.epsilons[-1]

    def set_eps_clip_val(self, eps_clip_val):
        logger.info(f"setting epsilon clip val to {eps_clip_val}")
        self.D = eps_clip_val

    def set_gamma(self, gamma):
        logger.info(f"setting gamma val to {gamma}")
        self.gamma = gamma

    def export_info(self):
        return {
            "u_before": np.array(self.info["u_before"]).reshape(-1, 1),
            "u_after": np.array(self.info["u_after"]).reshape(-1, 1),
            "z_before": np.array(self.info["z_before"]).reshape(-1, 1),
            "z_after": np.array(self.info["z_after"]).reshape(-1, 1),
            "martingales": np.array(self.martingales).reshape(-1, 1),
            "epsilons": np.array(self.epsilons).reshape(-1, 1),
        }

    def reset(self):
        self.C = 1.0
        self.martingales = []
        self.gradients = []
        self.epsilons = [0]
        self.info = {
            "u_before": [],
            "u_after": [],
            "z_before": [],
            "z_after": [],
            "last_batch_u_before": [],
            "last_batch_u_after": [],
        }

    def protect(self, z):
        self.info["last_batch_u_before"] = []
        self.info["last_batch_u_after"] = []

        zz = z.clone().cpu().detach().numpy()
        for j in range(zz.shape[0]):
            self.info["z_before"].append(float(zz[j]))
            u = self.cdf(zz[j])

            u_protected = self.protect_u(u)

            z_fixed = self.cdf.inverse(u_protected)
            zz[j] = z_fixed

            self.info["u_before"].append(float(u))
            self.info["u_after"].append(float(u_protected))
            self.info["z_after"].append(float(z_fixed))
            self.info["last_batch_u_before"].append(float(u))
            self.info["last_batch_u_after"].append(float(u_protected))

        zz = torch.tensor(zz).to(self.device)
        return zz, self.info

    def sfogd(self, u_t):
        if len(self.epsilons) == 0:
            return 0

        if len(self.info["u_before"]) == 0:
            return 0

        eps_t = self.last_eps
        v_t = u_t - 0.5

        E_tau = self.D * np.sign(u_t - 0.50)
        ind = 0 if E_tau * eps_t > 0 and np.abs(eps_t) > self.D else 1

        grad_t = (v_t / (1 + eps_t * v_t)) * ind
        self.gradients.append(grad_t)

        if grad_t != 0:
            grad_arr = np.array(self.gradients)
            c = self.gamma * ((grad_t) / (np.sqrt((grad_arr**2).sum())))
            eps_new = eps_t + c
        else:
            eps_new = eps_t

        # print(f"Grad = {grad_t:.6f}, New eps = {eps_new:.6f}") # added by Louis for debugging

        return eps_new

    def protect_u(self, u_t):
        C = self.C

        eps_t = self.last_eps
        b = 1 + eps_t * (u_t - 0.5)

        u_protected = 0.5 * eps_t * (u_t**2) + (1 - 0.5 * eps_t) * u_t

        self.info["u_before"].append(float(u_t))  # move this up!

        eps_new = self.sfogd(u_t)

        self.C = min(float(C * b), 1e200)
        self.martingales.append(self.C)
        self.epsilons.append(float(eps_new))

        # print(f"u_t = {u_t:.4f}, eps = {eps_t:.4f}, b = {b:.4f}, S = {self.C:.4f}")  # added by Louis for debugging

        return u_protected


def get_protector_from_ents(ents, args):
    logger.info("creating protector from ents")
    cdf = CDF(ents)
    gamma = args.gamma
    eps_clip_val = args.eps_clip
    protector = Protector(cdf, args.device)

    protector.set_gamma(gamma)
    protector.set_eps_clip_val(eps_clip_val)
    return protector
