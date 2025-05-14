import numpy as np
import torch
from torch.utils.data import DataLoader

class WeightedCDF:
    def __init__(self, entropies, pseudo_labels, p_s, p_t):
        weights = np.array([p_t[y] / p_s[y] if p_s[y] > 0 else 0.0 for y in pseudo_labels])
        weights = np.clip(weights, 0.1, 10.0)
        print(f"min weight: {weights.min():.4f}, max weight: {weights.max():.4f}")
        sorted_idx = np.argsort(entropies)
        self.sorted_ents = np.array(entropies)[sorted_idx]
        self.sorted_weights = weights[sorted_idx]
        self.cum_weights = np.cumsum(self.sorted_weights)
        self.cum_weights /= self.cum_weights[-1]

    def __call__(self, z):
        idx = np.searchsorted(self.sorted_ents, z, side="right")
        return self.cum_weights[min(idx, len(self.cum_weights) - 1)]

    def inverse(self, u):
        idx = np.searchsorted(self.cum_weights, u, side="right")
        return self.sorted_ents[min(idx, len(self.sorted_ents) - 1)]

def estimate_label_distribution(model, loader, device, num_classes=10):
    counts = torch.zeros(num_classes)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            for cls in preds.cpu():
                counts[cls] += 1
    p_y = counts / counts.sum()
    return p_y


def estimate_pseudo_label_distribution(model, loader, device, num_classes=10):
    counts = torch.zeros(num_classes)
    pseudo_labels = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            pseudo_labels.extend(preds.cpu().tolist())
            for cls in preds.cpu():
                counts[cls] += 1

    p_y = counts / counts.sum()
    return p_y, pseudo_labels


