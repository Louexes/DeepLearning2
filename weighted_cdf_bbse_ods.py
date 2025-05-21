import numpy as np
import torch


class BBSEODSWeightedCDF:
    def __init__(
        self, entropies, p_test, p_source, source_labels, confusion_matrix, device, ods_alpha=0.05, num_classes=10
    ):
        ## BBSE / ODS stuff;
        self.device = device
        self.ods_alpha = ods_alpha  # ODS update parameter
        self.num_classes = num_classes
        self.C = confusion_matrix  # Confusion matrix
        self.p_source = p_source  # source distribution
        self.p_test = p_test  # initial test distribution
        self.p_test_true = self.correct_p_test(self.p_test)  # Corrected test distribution

        # Calculate weights and map to source labels
        self.source_labels = source_labels
        weights = self.p_test_true / self.p_source  # Once p_test_true is updated, we can update the weights.
        weights = torch.clamp(weights, min=0.1, max=10.0)  # clipping extreme weights similar to Louis' approach.

        # Claude Fix: Use .cpu() before converting to numpy array
        weights_cpu = weights.cpu() if isinstance(weights, torch.Tensor) else weights
        self.weights = np.array([weights_cpu[label].item() for label in source_labels])

        # Sort entropies and weights
        self.sorted_idx = np.argsort(entropies)
        self.sorted_ents = np.array(entropies)[self.sorted_idx]
        self.sorted_weights = self.weights[self.sorted_idx]

        # Compute cumulative weights for CDF
        self.cum_weights = np.cumsum(self.sorted_weights)
        self.cum_weights /= self.cum_weights[-1]

    def correct_p_test(self, p_tilde):
        """
        Corrected target distribution using the confusion matrix.
        From p_tilde (predicted label distribution) to p (true label distribution)
        """
        # Move tensors to CPU for numpy operations if needed
        C_cpu = self.C.cpu() if isinstance(self.C, torch.Tensor) else self.C
        p_tilde_cpu = p_tilde.cpu() if isinstance(p_tilde, torch.Tensor) else p_tilde

        # Add small ridge to ensure invertibility
        ridge = 1e-6 * torch.eye(self.num_classes, device="cpu")
        C_T = C_cpu.t() + ridge

        # Solve for p_true: C^T · p_true = p_tilde
        p_true = torch.linalg.solve(C_T, p_tilde_cpu)

        # Ensure valid probability distribution
        p_true = torch.clamp(p_true, min=0.0)
        p_true = p_true / p_true.sum()

        # Move back to original device if needed
        if isinstance(p_tilde, torch.Tensor) and hasattr(p_tilde, "device") and p_tilde.device.type != "cpu":
            p_true = p_true.to(p_tilde.device)

        return p_true

    def __call__(self, z):
        idx = np.searchsorted(self.sorted_ents, z, side="right")
        return self.cum_weights[min(idx, len(self.cum_weights) - 1)]

    def inverse(self, u):
        idx = np.searchsorted(self.cum_weights, u, side="right")
        return self.sorted_ents[min(idx, len(self.sorted_ents) - 1)]

    def update_weights(self):
        # If you want weights updates without BBSE, rename to p_test_true to p_test
        weights = self.p_test_true / self.p_source
        weights = torch.clamp(weights, min=0.1, max=10.0)

        # CPU conversion before numpy operations
        weights_cpu = weights.cpu() if isinstance(weights, torch.Tensor) else weights
        self.weights = np.array([weights_cpu[label].item() for label in self.source_labels])

        self.sorted_weights = self.weights[self.sorted_idx]
        self.cum_weights = np.cumsum(self.sorted_weights)
        self.cum_weights /= self.cum_weights[-1]

    def batch_ods_update(self, pseudo_labels):
        """Update distribution with a batch of pseudo-labels."""
        for label in pseudo_labels:
            self.ods_update(label)

    def ods_update(self, pseudo_label):
        """Update test distribution with new pseudo-label using EMA."""
        # Create one-hot encoding for the pseudo-label
        one_hot = torch.zeros(self.num_classes, device=self.device)
        one_hot[pseudo_label] = 1.0

        # EMA update with alpha:
        self.p_test = (1 - self.ods_alpha) * self.p_test + self.ods_alpha * one_hot

        # Corrected update
        self.p_test_true = self.correct_p_test(self.p_test)

        # Update weights based on the new distribution
        self.update_weights()



def estimate_confusion_matrix(model, loader, device, num_classes=10, eps=1e-6):
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
