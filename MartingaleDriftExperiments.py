import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import protector as protect
from temperature_scaling import ModelWithTemperature
from utils.cli_utils import softmax_ent
from tent import Tent, configure_model, collect_params
from typing import Sequence, Tuple, Dict, Optional
import argparse

from utilities import *
from plotting import *

CORRUPTIONS = (
    "shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise",
    "defocus_blur", "brightness", "fog", "zoom_blur", "frost", "glass_blur",
    "impulse_noise", "contrast", "jpeg_compression", "elastic_transform"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--method', type=str, default='none', choices=['none', 'tent'])
    parser.add_argument('--corruption', type=str, default='gaussian_noise')
    parser.add_argument('--all_corruptions', action='store_true')
    parser.add_argument('--n_examples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = args.device if torch.backends.mps.is_available() or args.device == 'mps' else 'cpu'
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))

    print("🚀 Loading model...")
    model = get_model(args.method, device)

    print("📦 Loading clean CIFAR-10 as source entropy")
    clean_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(), transform]))
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False)
    source_ents, _ = evaluate(model, clean_loader, device)

    protector = protect.get_protector_from_ents(source_ents, argparse.Namespace(gamma=1 / (8 * np.sqrt(3)), eps_clip=1.8, device=device))

    corruptions = CORRUPTIONS if args.all_corruptions else [args.corruption]
    entropy_streams, accs = {}, {}

    for corruption in corruptions:
        for severity in range(1, 6):
            print(f"🔎 {corruption} severity {severity}")
            x, y = load_cifar10c(args.n_examples, severity, corruption)
            dataset = BasicDataset(x, y, transform=transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            ents, acc = evaluate(model, loader, device)
            key = f"{corruption}_s{severity}"
            entropy_streams[key] = ents
            accs[key] = acc

    results = run_martingale(entropy_streams, protector)
    #plot_results(results, metric='log_sj', ylabel='log(Sj)', title='Martingale Wealth over Time')
    #plot_results(results, metric='eps', ylabel='Epsilon', title='Epsilon Adaptation over Time')

    print("\n📊 Accuracy summary:")
    for key in accs:
        print(f"{key}: {accs[key]*100:.2f}%")

if __name__ == '__main__':
    main()
