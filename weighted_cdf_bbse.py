import numpy as np
import torch


class BBSEWeightedCDF:
    def __init__(self, entropies, pseudo_labels, weights):
        # Map class weights to sample weights
        weights = np.array([weights[label].item() for label in pseudo_labels])
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


def estimate_confusion_matrix(model, loader, device, num_classes, eps=1e-6):
    """
    Estimate P(ŷ = i | y = j) on a labeled source set.
    Returns:
      C   : [num_classes × num_classes] torch.Tensor, with rows summing to 1
      p_s : [num_classes] torch.Tensor, the empirical source label marginals
    """
    # confusion_counts[j, i] = # samples with true=j, pred=i
    # indexing for consistency with formulation of the conditional probability above
    confusion_counts = torch.zeros(num_classes, num_classes, device=device)
    true_counts = torch.zeros(num_classes, device=device)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            for t, p in zip(y, preds):
                # true labels on rows, predictions on columns
                confusion_counts[t, p] += 1
                true_counts[t] += 1

    # empirical training set marginals p_s(y)
    p_s = (true_counts) / (true_counts.sum())

    # avoid zero‐division, add small eps
    true_counts = true_counts + eps
    confusion_matrix = confusion_counts / true_counts.unsqueeze(1)  # each row j sums to ≈1

    # # empirical training set marginals p_s(y)
    # p_s = (true_counts - eps) / (true_counts.sum() - eps * num_classes)
    return confusion_matrix, p_s


def estimate_target_distribution_from_preds(model, loader, device, num_classes=10):
    """
    Fundamentally the same function as estimate_label_distribution.
    """
    counts = torch.zeros(num_classes, device=device)
    total = 0

    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            for p in preds:
                counts[p] += 1
            total += preds.size(0)

    return counts / total


def estimate_shift_weights(model, src_loader, tgt_loader, device, num_classes=10):
    """
    Full pipeline to get w(y) = q(y)/p(y):
      1. Estimate confusion matrix C = P(ŷ|y) on labeled training set.
      2. Estimate p_t_pred = P(ŷ) on unlabeled target.
      3. Invert C^T to recover p_t_true (≈ true target marginals).
      4. Compute w = p_t_true / p_s_true.
    """
    # 1. source confusion + source marginals
    C, p_s = estimate_confusion_matrix(model, src_loader, device, num_classes)

    # 2. target predicted marginals
    p_t_pred = estimate_target_distribution_from_preds(model, tgt_loader, device, num_classes)

    # # 3. recover p_t_true by solving (C^T) · p_t_true = p_t_pred
    # # add "ridge" to the diagonal so all eigenvalues are non-zero; thus making matrix invertible
    ridge = 1e-6 * torch.eye(num_classes, device=device)
    # C_T = C.t() + ridge
    # p_t_true = torch.linalg.solve(C_T, p_t_pred)

    # Temporary fix for MPS
    # Move matrices to CPU for the solve operation
    C_cpu = C.to("cpu")
    p_t_pred_cpu = p_t_pred.to("cpu")
    ridge_cpu = ridge.to("cpu")
    # Perform operation on CPU
    C_T_cpu = C_cpu.t() + ridge_cpu
    p_t_true_cpu = torch.linalg.solve(C_T_cpu, p_t_pred_cpu)
    # Move result back to original device
    p_t_true = p_t_true_cpu.to(device)

    # enforce valid probability vector
    p_t_true = torch.clamp(p_t_true, min=0.0)
    p_t_true = p_t_true / p_t_true.sum()

    # 4. weights w(y) = q(y)/p(y)
    w = p_t_true / p_s
    # clipping extreme weights similar to Louis' approach.
    w = torch.clamp(w, min=0.1, max=10.0)

    return w, p_s, p_t_true, p_t_pred
