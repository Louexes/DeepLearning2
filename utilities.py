import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from temperature_scaling import ModelWithTemperature
from tent import Tent, collect_params, configure_model
from utils.cli_utils import softmax_ent


class BasicDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y = x, y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]


def load_cifar10c(
    n_examples: int, severity: int, corruption: str, data_dir: str = "./data/CIFAR-10-C"
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_dir = os.path.abspath(data_dir)
    x = np.load(os.path.join(base_dir, f"{corruption}.npy"))
    y = np.load(os.path.join(base_dir, "labels.npy"))
    assert 1 <= severity <= 5

    n_total = 10000
    x = x[(severity - 1) * n_total : severity * n_total][:n_examples]
    y = y[:n_examples]

    x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32) / 255.0
    return torch.tensor(x), torch.tensor(y)


def get_model(method: str, device: str):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    model = ModelWithTemperature(model, 1.8).to(device)

    if method == "tent":
        model = configure_model(model)
        params, _ = collect_params(model)
        optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
        model = Tent(model, optimizer)
    return model.eval()


def evaluate(model, dataloader, device):
    """Evaluate entropy values, accuracy, logits and labels."""
    entropies, correct, total = [], 0, 0
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            entropies.extend(softmax_ent(logits).tolist())
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            logits_list.append(logits.cpu())
            labels_list.append(y.cpu())
    accuracy = correct / total
    return np.array(entropies), accuracy, logits_list, labels_list


def run_martingale(entropy_streams: Dict[str, np.ndarray], protector) -> Dict[str, Dict[str, list]]:
    results = {}
    for name, z_seq in entropy_streams.items():
        protector.reset()
        logs, eps = [], []
        for z in z_seq:
            u = protector.cdf(z)
            protector.protect_u(u)
            # logs.append(protector.martingales[-1] + 1e-8)
            logs.append(np.log(protector.martingales[-1] + 1e-8))
            eps.append(protector.epsilons[-1])
        results[name] = {"log_sj": logs, "eps": eps}
    return results


def compute_accuracy_over_time_from_logits(logits_list, labels_list):
    accs = []
    for logits, labels in zip(logits_list, labels_list):
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        accs.append(acc)
    return accs


def compute_detection_delays(results_dict, threshold=np.log(100)):
    delays = {}
    for name, data in results_dict.items():
        log_sj = np.array(data["log_sj"])
        above_thresh = np.where(log_sj > threshold)[0]
        if len(above_thresh) > 0:
            delays[name] = int(above_thresh[0])
        else:
            delays[name] = len(log_sj)
    return delays


def collect_method_comparison_results(method_names, raw_logs):
    """
    method_names: list of str, e.g., ['no_adapt', 'tent', 'poem']
    raw_logs: dict of method name -> dict with keys like 'log_sj', 'eps', 'ents', 'accs1'
    """
    results = {}
    for method in method_names:
        method_data = raw_logs.get(method, {})
        results[method] = {
            "log_sj": method_data.get("log_sj", []),
            "eps": method_data.get("eps", []),
            "ents": method_data.get("ents", []),
            "accs": method_data.get("accs1", []),
        }
    return results


def load_clean_then_corrupt_sequence(
    corruption: str,
    severity: int,
    n_examples: int = 1000,
    data_dir: str = "./data",
    transform=None,
    batch_size: int = 64,
) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    """
    Create a DataLoader with clean CIFAR-10 test set followed by corrupted CIFAR-10-C samples.
    Returns the DataLoader, and a boolean mask (is_clean), and labels.
    """
    # Load clean test set
    clean_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    clean_x = torch.stack([img for img, _ in clean_ds])[:n_examples]
    clean_y = torch.tensor([label for _, label in clean_ds])[:n_examples]

    # Load corrupted samples
    corrupt_x, corrupt_y = load_cifar10c(
        n_examples, severity, corruption, data_dir=os.path.join(data_dir, "CIFAR-10-C")
    )

    # Combine
    all_x = torch.cat([clean_x, corrupt_x], dim=0)
    all_y = torch.cat([clean_y, corrupt_y], dim=0)
    is_clean = np.array([True] * len(clean_x) + [False] * len(corrupt_x))

    if transform:
        dataset = BasicDataset(all_x, all_y, transform=transform)
    else:
        dataset = BasicDataset(all_x, all_y)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, is_clean, all_y.numpy()


def compute_detection_delays_from_threshold(
    log_sj_dict: Dict[str, List[float]], threshold: float = np.log(100), start_index: int = 4000
) -> Dict[str, int]:
    """
    Compute number of samples after corruption onset (start_index) until log(Sj) crosses threshold.

    Parameters:
        log_sj_dict (Dict[str, List[float]]): Dictionary of log(Sj) values for each corruption/severity.
        threshold (float): Detection threshold (default: log(100)).
        start_index (int): Index at which corruption begins.

    Returns:
        Dict[str, int]: Detection delays (number of samples after start_index) for each key.
                        If not detected, returns total remaining samples.
    """
    detection_delays = {}
    for key, log_sj in log_sj_dict.items():
        log_sj = np.array(log_sj)
        post_shift = log_sj[start_index:]
        above_thresh = np.where(post_shift > threshold)[0]
        if len(above_thresh) > 0:
            delay = above_thresh[0]
        else:
            delay = len(post_shift)  # never crosses threshold
        detection_delays[key] = delay
    return detection_delays


def compute_accuracy_drops(accuracies_dict, split_index=62):  # 4000/64 ≈ 62 batches
    """
    Compute accuracy drops between clean and corrupted data.

    Args:
        accuracies_dict: Dictionary containing batch-wise accuracies
        split_index: Batch index where corruption starts (4000/batch_size)

    Returns:
        Dictionary of accuracy drops for each corruption type
    """
    accuracy_drops = {}
    for corruption, accs in accuracies_dict.items():
        accs = np.array(accs)
        # Ensure we have enough batches
        if len(accs) > split_index:
            acc_before = np.mean(accs[:split_index])
            acc_after = np.mean(accs[split_index:])
            drop = acc_before - acc_after
            accuracy_drops[corruption] = drop

    return accuracy_drops


def compute_entropy_spikes(entropy_streams: Dict[str, np.ndarray]):
    """
    Compute entropy spikes for each corruption type.

    Returns:
        Dictionary of entropy spikes for each corruption type
    """
    spikes = {}
    for key, ents in entropy_streams.items():
        ents = np.array(ents)
        spike = np.max(ents) - np.min(ents)
        spikes[key] = spike

    return spikes


def compute_detection_confidence_slope(
    log_sj_dict: Dict[str, List[float]], eps_dict: Dict[str, List[float]], start_index: int = 4000
) -> Dict[str, float]:
    """
    Compute the slope of the confidence curve for each corruption type.

    Parameters:
        log_sj_dict (Dict[str, List[float]]): Dictionary of log(Sj) values for each corruption/severity.
        eps_dict (Dict[str, List[float]]): Dictionary of epsilon values for each corruption/severity.
        start_index (int): Index at which corruption begins.

    Returns:
        Dict[str, float]: Slopes for each key.
    """
    slopes = {}
    for key in log_sj_dict.keys():
        log_sj = np.array(log_sj_dict[key])
        eps = np.array(eps_dict[key])
        post_shift_log_sj = log_sj[start_index:]
        post_shift_eps = eps[start_index:]

        if len(post_shift_log_sj) > 1 and len(post_shift_eps) > 1:
            slope = np.polyfit(post_shift_log_sj, post_shift_eps, 1)[0]
            slopes[key] = slope
        else:
            slopes[key] = None  # Not enough data to compute slope

    return slopes
