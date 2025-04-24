import argparse
import math
import os
import random
import time
import uuid
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torchvision
import torchvision.transforms as transforms
from pycm import *
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import poem
import protector as protect
import sar
import tent
import tent_ext
from temperature_scaling import ModelWithTemperature, _ECELoss

# from utils.utils import get_logger
from utils.cli_utils import *

CORRUPTIONS = (
    "shot_noise",
    "motion_blur",
    "snow",
    "pixelate",
    "gaussian_noise",
    "defocus_blur",
    "brightness",
    "fog",
    "zoom_blur",
    "frost",
    "glass_blur",
    "impulse_noise",
    "contrast",
    "jpeg_compression",
    "elastic_transform",
)


class BenchmarkDataset(Enum):
    cifar_10 = "cifar10"
    cifar_100 = "cifar100"
    imagenet = "imagenet"


def my_load_cifar10c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = "./data",
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruptions_cifar(BenchmarkDataset.cifar_10, n_examples, severity, data_dir, corruptions, shuffle)


def load_corruptions_cifar(
    dataset: BenchmarkDataset,
    n_examples: int,
    severity: int,
    data_dir: str,
    corruptions: Sequence[str] = CORRUPTIONS,
    shuffle: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir

    # Louis: here's their automatic Corruptions do
    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / "labels.npy"
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + ".npy")
        if not corruption_file_path.is_file():
            raise DownloadError(f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar : severity * n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test


class BasicDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.x[idx]), self.y[idx]
        return self.x[idx], self.y[idx]


def get_args():
    parser = argparse.ArgumentParser(description="SAR exps")

    # path
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--data", default="/datasets/ImageNet", help="path to dataset")
    parser.add_argument(
        "--data_corruption", default="/datasets/ImageNet/ImageNet-C", help="path to corruption dataset"
    )
    parser.add_argument(
        "--v2_path",
        default="/datasets/ImageNet2/imagenetv2-matched-frequency-format-val/",
        help="path to corruption dataset",
    )
    parser.add_argument("--output", default="./exps", help="the output directory of this experiment")
    parser.add_argument("--source_domain", default="Real World", help="name of source domain")
    parser.add_argument("--target_domain", default="Real World", help="name of source domain")

    parser.add_argument("--seed", default=2021, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--debug", default=False, type=bool, help="debug or not.")

    # dataloader
    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument(
        "--test_batch_size", default=4, type=int, help="mini-batch size for testing, before default value is 4"
    )
    parser.add_argument("--if_shuffle", default=True, type=bool, help="if shuffle the test set.")

    # corruption settings
    parser.add_argument("--level", default=5, type=int, help="corruption level of test(val) set.")
    parser.add_argument("--corruption", default="gaussian_noise", type=str, help="corruption type of test(val) set.")

    # eata settings
    parser.add_argument(
        "--fisher_size", default=2000, type=int, help="number of samples to compute fisher information matrix."
    )
    parser.add_argument(
        "--fisher_alpha",
        type=float,
        default=2000.0,
        help="the trade-off between entropy and regularization loss, in Eqn. (8)",
    )
    parser.add_argument(
        "--e_margin",
        type=float,
        default=math.log(1000) * 0.40,
        help="entropy margin E_0 in Eqn. (3) for filtering reliable samples",
    )
    parser.add_argument(
        "--d_margin", type=float, default=0.05, help="\epsilon in Eqn. (5) for filtering redundant samples"
    )

    # Exp Settings
    parser.add_argument("--method", default="eata", type=str, help="no_adapt, tent, eata, sar, cotta, poem")
    parser.add_argument(
        "--model", default="resnet50_gn_timm", type=str, help="resnet50_gn_timm or resnet50_bn_torch or vitbase_timm"
    )
    parser.add_argument(
        "--exp_type",
        default="normal",
        type=str,
        help="normal, continual, bs1, in_dist, natural_shift, severity_shift, eps_cdf, martingale",
    )
    parser.add_argument("--cont_size", default=7500, type=int, help="each corruption size for continual type")
    parser.add_argument("--severity_list", nargs="+", type=int, default=[5, 4, 3, 2, 1, 2, 3, 4, 5])
    parser.add_argument("--temp", type=float, default=1, help="temperature for the model to be calibrated")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exp_comment", type=str, default="")

    # SAR parameters
    parser.add_argument(
        "--sar_margin_e0",
        default=math.log(1000) * 0.40,
        type=float,
        help="the threshold for reliable minimization in SAR, Eqn. (2)",
    )
    parser.add_argument(
        "--imbalance_ratio",
        default=500000,
        type=float,
        help="imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order). See Section 4.3 for details;",
    )

    # PEM parameters
    parser.add_argument("--gamma", type=float, help="protector's gamma", default=1 / (8 * sqrt(3)))
    parser.add_argument("--eps_clip", type=float, help="clipping value for epsilon during protection", default=1.80)
    parser.add_argument("--lr_factor", type=float, default=1, help="multiplies the learning rate for poem")
    parser.add_argument(
        "--vanilla_loss", action="store_false", dest="vanilla_loss", help="Use vanilla match loss (not l match ++)."
    )

    return parser.parse_args()


def run(data_loader, model, args):
    ents = []
    accs1 = []
    accs5 = []
    logits_list = []
    labels_list = []

    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(data_loader), [batch_time, top1, top5], prefix="Test: ")

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, dl in enumerate(data_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            # compute output
            output = model(images).detach()

            # _, targets = output.max(1)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            accs1.extend(acc1.tolist())
            accs5.extend(acc5.tolist())
            logits_list.append(output)
            labels_list.append(target)

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            ents.extend(softmax_ent(output).tolist())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # break
            if i % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break

    with torch.no_grad():
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        ece_criterion = _ECELoss().cuda()
        ece = ece_criterion(logits, labels.view(-1)).item()

        base_model = get_model(args)

        model_delta = get_models_delta(model, base_model.to(args.device))

    info = {
        "top1": top1.avg.item(),
        "top5": top5.avg.item(),
        "accs1": accs1,
        "accs5": accs5,
        "ents": ents,
        "ece": ece,
        "model_delta": model_delta,
    }

    return info


def get_model(args):
    # build model for adaptation
    bs = args.test_batch_size
    # net = load_model('Addepalli2021Towards_WRN34', "./ckpt", 'cifar10', ThreatModel.corruptions).cuda()

    if args.dataset == "cifar10":
        net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True).cuda()
    elif args.dataset == "cifar100":
        net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True).cuda()

    elif args.dataset == "office_home":
        # Load the model architecture (make sure this matches the architecture you used for training)
        if args.model == "resnet50_gn_timm":
            net = timm.create_model("resnet50_gn", pretrained=False, num_classes=65).cuda()

        elif args.model == "vitbase_timm":
            net = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=65).cuda()

        else:
            raise "Not implemented"

        # Load the saved state dictionary
        ret = net.load_state_dict(
            torch.load(
                f"models/office_home_pretrained/{args.source_domain.split()[0]}_{args.model}_{args.seed}_best.pth"
            )
        )
        print(ret)

    else:
        raise "Not valid dataset"

    # For cifar10 temp is 1.8
    # For cifar100 temp is 10000000
    net = ModelWithTemperature(net, args.temp)
    net.eval()
    return net


def run_comparison(test_loader, holdout_loader, holdout_dataset, args):
    # ----------------------------------------------------------------------------------
    # Tent Extension with gradient based adaptation
    # ----------------------------------------------------------------------------------
    start = time.time()
    net = get_model(args)
    net = tent_ext.configure_model(net)
    params, param_names = tent_ext.collect_params(net)
    # logger.info(param_names)
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)

    #  stuff needed by the protector
    info_on_holdout = run(holdout_loader, net, args)
    info_on_holdout["method"] = "holdout"
    info_on_holdout.update(**vars(args))
    protector = protect.get_protector_from_ents(info_on_holdout["ents"], args)
    # Modify slope threshold here
    tented_ext_model = tent_ext.Tent_ext(net, optimizer, protector, slope_threshold=0.02)

    tent_ext_info = run(test_loader, tented_ext_model, args)
    end = time.time()
    tent_ext_info["runtime"] = end - start

    # ----------------------------------------------------------------------------------
    # End extension
    # ----------------------------------------------------------------------------------

    # No Adapt
    start = time.time()
    net = get_model(args)
    info_no_adapt = run(test_loader, net, args)
    end = time.time()
    info_no_adapt["runtime"] = end - start

    # Tent
    start = time.time()
    net = get_model(args)
    net = tent.configure_model(net)
    params, param_names = tent.collect_params(net)
    # logger.info(param_names)
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    tented_model = tent.Tent(net, optimizer)

    tent_info = run(test_loader, tented_model, args)
    end = time.time()
    tent_info["runtime"] = end - start

    # POEM
    start = time.time()
    net = get_model(args)

    info_on_holdout = run(holdout_loader, net, args)
    info_on_holdout["method"] = "holdout"
    info_on_holdout.update(**vars(args))
    protector = protect.get_protector_from_ents(info_on_holdout["ents"], args)
    # resetting the model
    net = sar.configure_model(net)
    params, param_names = sar.collect_params(net)

    if args.exp_type == "martingale":
        args.adapt = i % 2 == 0

    protector = protect.get_protector_from_ents(info_on_holdout["ents"], args)
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    adapt_model = poem.POEM(net, optimizer, protector, e0=args.sar_margin_e0, adapt=True, vanilla_loss=False)
    poem_info = run(test_loader, adapt_model, args)
    end = time.time()
    poem_info["runtime"] = end - start

    window_size = args.cont_size // args.bs
    data = [
        {
            "method": "tent_ext",
            "corruption_acc": np.array(tent_ext_info["accs1"]).reshape(-1, window_size).mean(axis=1),
            "top1": tent_ext_info["top1"],
            "runtime": tent_ext_info["runtime"],
        },
        {
            "method": "no_adapt",
            "corruption_acc": np.array(info_no_adapt["accs1"]).reshape(-1, window_size).mean(axis=1),
            "top1": info_no_adapt["top1"],
            "runtime": info_no_adapt["runtime"],
        },
        {
            "method": "tent",
            "corruption_acc": np.array(tent_info["accs1"]).reshape(-1, window_size).mean(axis=1),
            "top1": tent_info["top1"],
            "runtime": tent_info["runtime"],
        },
        {
            "method": "poem",
            "corruption_acc": np.array(poem_info["accs1"]).reshape(-1, window_size).mean(axis=1),
            "top1": poem_info["top1"],
            "runtime": poem_info["runtime"],
        },
    ]

    args.timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    args.exp_name = args.timestamp + "seed{}".format(args.seed) + "_" + str(uuid.uuid4())[:6]
    output_path = Path(args.output) / args.dataset / "cont_long" / args.exp_name
    output_path.parent.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(data)
    df["model"] = args.model
    df["lr"] = args.lr
    df["dataset"] = args.dataset
    df["source_domain"] = args.source_domain
    df["target_domain"] = args.target_domain
    df["seed"] = args.seed
    df["temp"] = args.temp
    df["bs"] = args.bs
    df["cont_size"] = args.cont_size
    df.to_csv(f"{output_path}.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.bs = args.test_batch_size
    args.print_freq = 4000 // 20 // args.bs
    n_examples = 10000
    severity = args.level

    datasets = []

    holdout_indices, test_indices = torch.tensor_split(
        torch.randperm(n_examples, generator=torch.Generator().manual_seed(args.seed)), [int(n_examples * 0.25)]
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]
    )
    # CONTINUAL
    if args.dataset in ["cifar10", "cifar100"]:
        for corruption in tqdm(CORRUPTIONS):
            # Load data for current corruption
            if args.dataset == "cifar10":
                x, y = my_load_cifar10c(n_examples, severity, "./data/CIFAR-10-C", False, [corruption])
            elif args.dataset == "cifar100":
                x, y = my_load_cifar10c(n_examples, severity, "./data/CIFAR-100-C", False, [corruption])
            else:
                raise "not valid dataset"

            # Split into test set
            x_test, y_test = x[test_indices, ...], y[test_indices, ...]

            # Randomly sample 1000 examples
            # Permute all examples
            sample_indices = random.sample(range(len(x_test)), args.cont_size)

            x_sample = x_test[sample_indices]
            y_sample = y_test[sample_indices]

            # Create BasicDataset and append to list
            datasets.append(BasicDataset(x_sample, y_sample))

        # Combine all datasets into one
        combined_dataset = ConcatDataset(datasets)
        print(len(combined_dataset))

        if args.dataset == "cifar10":
            original_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        elif args.dataset == "cifar100":
            original_ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        else:
            raise "not valid dataset"

        holdout_dataset = Subset(original_ds, holdout_indices.tolist())
        cifar10_test = Subset(original_ds, test_indices.tolist())

        holdout_loader = DataLoader(holdout_dataset, num_workers=8, batch_size=args.bs, shuffle=False)
        test_loader = DataLoader(combined_dataset, num_workers=8, batch_size=args.bs, shuffle=False)
    elif args.dataset == "office_home":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        augment_transform = transforms.Compose(
            [
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        all_domains = ["Art", "Clipart", "Product", "Real World"]
        other_domains = [domain for domain in all_domains if domain != args.source_domain]

        full_dataset = ImageFolder(f"./data/office_home/{args.source_domain}", transform=transform)
        # Split the dataset
        train_dataset, val_dataset = random_split(
            full_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(args.seed)
        )
        holdout_dataset = val_dataset
        holdout_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)

        datasets = {}

        for target_domain in other_domains:
            original_dataset = ImageFolder(f"./data/office_home/{target_domain}", transform=transform)
            num_samples = len(original_dataset)

            # Generate a random permutation of indices
            indices = torch.randperm(num_samples).tolist()
            datasets[target_domain] = Subset(original_dataset, indices)

        test_loaders = {
            domain: DataLoader(datasets[domain], batch_size=args.test_batch_size, shuffle=False, num_workers=8)
            for domain in other_domains
        }

    else:
        raise "Not implemented"

    if args.dataset == "office_home":
        for domain, test_loader in test_loaders.items():
            print(domain)
            args.target_domain = domain
            run_comparison(test_loader, holdout_loader, holdout_dataset, args)

    elif args.dataset in ["cifar10", "cifar100"]:
        run_comparison(test_loader, holdout_loader, holdout_dataset, args)
