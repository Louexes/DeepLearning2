import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pickle


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


def load_cifar10_clean(
    n_examples: int,
    data_dir: str = "./data/cifar-10-batches-py"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads clean CIFAR-10 test data (from 'test_batch') and returns tensors of shape (N, 3, 32, 32).
    Equivalent to how load_cifar10c slices corrupted data.
    """
    # Path to test_batch
    test_batch_path = os.path.join(data_dir, "test_batch")

    # Load with pickle
    with open(test_batch_path, 'rb') as f:
        entry = pickle.load(f, encoding='bytes')
        x = entry[b'data']
        y = entry[b'labels']

    # Reshape and normalize
    x = x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(y)

    # Slice to n_examples
    x = x[:n_examples]
    y = y[:n_examples]

    return torch.tensor(x), torch.tensor(y)



def load_cifar10_label_shift(
    keep_classes: Sequence[int] = (0, 1, 2),
    n_examples: int = 8000,
    shift_point: int = 4000,
    data_dir: str = "./data/cifar-10-batches-py"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads CIFAR-10 clean test data and introduces label shift after a given index.

    Parameters:
        keep_classes: Classes to retain after shift_point.
        n_examples: Total number of examples to return.
        shift_point: Index at which label shift begins.
        data_dir: Path to unzipped 'cifar-10-batches-py' directory.

    Returns:
        Tuple of (x, y) as torch tensors with shape [N, 3, 32, 32] and [N]
    """
    # Load raw test batch
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        entry = pickle.load(f, encoding="bytes")
        x_all = entry[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_all = np.array(entry[b"labels"])

    # Balanced clean portion
    x_clean = x_all[:shift_point]
    y_clean = y_all[:shift_point]

    # Label-shifted portion: only keep selected classes
    mask = np.isin(y_all, keep_classes)
    #x_shift = x_all[mask][:(n_examples - shift_point)]
    #y_shift = y_all[mask][:(n_examples - shift_point)]

    n_per_class = (n_examples - shift_point) // len(keep_classes)
    x_shift, y_shift = sample_balanced_subset(x_all[mask], y_all[mask], keep_classes, n_per_class)

    # Combine clean + shifted portions
    x = np.concatenate([x_clean, x_shift])
    y = np.concatenate([y_clean, y_shift])

    return torch.tensor(x), torch.tensor(y)


def sample_balanced_subset(x, y, classes, n_per_class):
    indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        indices.extend(cls_indices[:n_per_class])
    return x[indices], y[indices]



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



def load_clean_then_label_shift_sequence(
    keep_classes: list,
    n_examples: int,
    shift_point: int = 4000,
    data_dir: str = "./data/cifar-10-batches-py",
    transform=None,
    batch_size: int = 64
):
    """
    Loads a dataset where the first part is clean and the second part is label-shifted.

    Parameters:
        keep_classes: List of class labels to retain after shift point.
        n_examples: Total number of examples.
        shift_point: Index at which label shift occurs.
        data_dir: Path to unzipped cifar-10-batches-py directory.
        transform: Normalization or transform function.
        batch_size: Batch size for the DataLoader.

    Returns:
        loader, is_clean_mask, true_labels
    """
    with open(os.path.join(data_dir, "test_batch"), "rb") as f:
        entry = pickle.load(f, encoding="bytes")
        x_all = entry[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_all = np.array(entry[b"labels"])

    x_clean = x_all[:shift_point]
    y_clean = y_all[:shift_point]

    mask = np.isin(y_all, keep_classes)
    x_shift = x_all[mask][:(n_examples - shift_point)]
    y_shift = y_all[mask][:(n_examples - shift_point)]

    x_combined = np.concatenate([x_clean, x_shift])
    y_combined = np.concatenate([y_clean, y_shift])

    x_tensor = torch.tensor(x_combined)
    y_tensor = torch.tensor(y_combined)

    dataset = BasicDataset(x_tensor, y_tensor, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    is_clean = np.array([1] * shift_point + [0] * (n_examples - shift_point))

    return loader, is_clean, y_combined



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


def apply_label_noise_to_dataset(dataset, noise_rate=0.3, num_classes=10):
    new_dataset = list(dataset)
    n = len(new_dataset)
    num_noisy = int(noise_rate * n)
    noisy_indices = np.random.choice(n, num_noisy, replace=False)

    for idx in noisy_indices:
        img, true_label = new_dataset[idx]
        noisy_label = random.choice([i for i in range(num_classes) if i != true_label])
        new_dataset[idx] = (img, noisy_label)

    return new_dataset


def simulate_prior_shift(dataset, class_priors, total_size=None):
    """
    dataset: list of (img, label) tuples
    class_priors: dict like {0: 0.4, 1: 0.1, ..., 9: 0.05}
    total_size: total number of samples in output (defaults to len(dataset))
    """
    by_class = defaultdict(list)
    for x in dataset:
        by_class[x[1]].append(x)

    if total_size is None:
        total_size = len(dataset)

    shifted_dataset = []
    for cls, prior in class_priors.items():
        n_samples = int(total_size * prior)
        class_samples = by_class[cls]
        if len(class_samples) < n_samples:
            # Oversample with replacement
            sampled = random.choices(class_samples, k=n_samples)
        else:
            # Undersample without replacement
            sampled = random.sample(class_samples, k=n_samples)
        shifted_dataset.extend(sampled)

    return shifted_dataset


def simulate_prior_shift_with_label_noise(dataset, class_priors, noise_rate=0.3, num_classes=10, total_size=None):
    """
    Applies class prior shift followed by label noise.

    Args:
        dataset: list of (img, label) tuples
        class_priors: dict like {0: 0.4, 1: 0.1, ..., 9: 0.05}
        noise_rate: fraction of labels to flip
        num_classes: number of possible classes
        total_size: number of samples in output (default: len(dataset))

    Returns:
        shifted_dataset: list of (img, possibly noisy_label) tuples
    """
    # Step 1: Apply class prior shift
    by_class = defaultdict(list)
    for x in dataset:
        by_class[x[1]].append(x)

    if total_size is None:
        total_size = len(dataset)

    shifted_dataset = []
    for cls, prior in class_priors.items():
        n_samples = int(total_size * prior)
        class_samples = by_class[cls]
        if len(class_samples) < n_samples:
            sampled = random.choices(class_samples, k=n_samples)
        else:
            sampled = random.sample(class_samples, k=n_samples)
        shifted_dataset.extend(sampled)

    # Step 2: Apply label noise
    n = len(shifted_dataset)
    num_noisy = int(noise_rate * n)
    noisy_indices = np.random.choice(n, num_noisy, replace=False)

    for i in noisy_indices:
        img, true_label = shifted_dataset[i]
        noisy_label = random.choice([c for c in range(num_classes) if c != true_label])
        shifted_dataset[i] = (img, noisy_label)

    return shifted_dataset


def load_dynamic_sequence(
    segments: list,
    segment_size: int,
    corruption_name: str,
    data_dir: str = "./data",
    transform=None,
    batch_size: int = 64,
) -> Tuple[DataLoader, np.ndarray, np.ndarray, list]:
    base_clean_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    base_clean_list = list(base_clean_ds)

    full_x, full_y, is_clean_mask, segment_labels = [], [], [], []

    for seg in segments:
        if seg == "clean":
            segment = random.sample(base_clean_list, segment_size)
            imgs = torch.stack([x[0] for x in segment])
            labels = torch.tensor([x[1] for x in segment])

        else:
            severity = int(seg[1])  # e.g., from "s2" → 2
            corrupt_x, corrupt_y = load_cifar10c(
                segment_size, severity, corruption_name, data_dir=os.path.join(data_dir, "CIFAR-10-C")
            )
            imgs = corrupt_x
            labels = corrupt_y

        full_x.append(imgs)
        full_y.append(labels)
        is_clean_mask.extend([seg == "clean"] * segment_size)
        segment_labels.extend([seg] * segment_size)

    all_x = torch.cat(full_x, dim=0)
    all_y = torch.cat(full_y, dim=0)

    if transform:
        dataset = BasicDataset(all_x, all_y, transform=transform)
    else:
        dataset = BasicDataset(all_x, all_y)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, np.array(is_clean_mask), all_y.numpy(), np.array(segment_labels)
