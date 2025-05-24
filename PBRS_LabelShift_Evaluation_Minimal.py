# PBRS_LabelShift_Evaluation_Minimal.py
# -------------------------------------
import os
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from protector  import PBRSBuffer
from utilities  import evaluate           # ← your helper


# ------------------------------------------------------------------ FPR
def evaluate_PBRS_FPR(
    model, load_cifar10_label_shift, BasicDataset, run_martingale, protector,
    transform, args, device, *,
    seeds=list(range(10)),
    buffer_capacity=512,
    confidence_threshold=0.5,
    num_classes_list=list(range(1, 11)),
    use_pbrs=True,
    log_path='fpr_results.csv',
    verbose=True,
):
    """False-positive rate under label-shift for the (optionally) PBRS-filtered
    entropy stream.  Returns a dict {subset_size: mean_FPR}."""
    label = "PBRS" if use_pbrs else "No PBRS"
    fprs  = {k: [] for k in num_classes_list}

    for seed in seeds:
        if verbose:
            print(f"\n🔁 Seed {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        for k in num_classes_list:
            thresh_hits = {}

            for subset in np.random.permutation(
                    list(combinations(range(10), k)))[:30]:

                # --------- build stream (2000 clean | 2000 label-shift) -----
                x, y = load_cifar10_label_shift(
                    keep_classes=subset, n_examples=4000, shift_point=2000)
                ds   = BasicDataset(x, y, transform=transform)
                dl   = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

                # --------- (optional) PBRS filtering -----------------------
                if use_pbrs:
                    buf  = PBRSBuffer(buffer_capacity, num_classes=k)
                    z    = np.full(len(ds), np.nan, dtype=np.float32)
                    idx  = 0

                    with torch.no_grad():
                        for xb, _ in dl:
                            pr  = torch.softmax(model(xb.to(device)), dim=1)
                            ent = -torch.sum(pr * torch.log(pr+1e-8), dim=1)
                            yhat = torch.argmax(pr, dim=1)
                            conf = torch.max(pr, dim=1).values

                            for e, c, yh in zip(ent.cpu(), conf.cpu(), yhat.cpu()):
                                if c > confidence_threshold and buf.accept(int(yh)):
                                    buf.add(idx, float(e), int(yh))
                                idx += 1

                    for j, e in buf.get_indexed_entropies():
                        z[j] = e
                else:
                    z, _, _, _ = evaluate(model, dl, device)

                # --------- martingale --------------------------------------
                key  = f"{label}_{k}cls_{'_'.join(map(str, subset))}"
                valid = z[~np.isnan(z)]
                if valid.size == 0:
                    triggered = False
                else:
                    logS = run_martingale({key: valid}, protector)[key]["log_sj"]
                    triggered = np.nanmax(logS) > np.log(100)

                thresh_hits[key] = triggered

            # aggregate over 30 random subsets
            fprs[k].append(np.mean(list(thresh_hits.values())))

            if verbose:
                hit = sum(thresh_hits.values())
                print(f"[DEBUG] k={k}: {hit}/30 triggered (FPR={fprs[k][-1]:.3f})")

    # ----------------------- pretty log + optional CSV ------------------
    print("\n📊 FPR summary")
    for k in num_classes_list:
        m, s = np.mean(fprs[k]), np.std(fprs[k])
        print(f"{k:2d} classes – {label}: {m:.3f} ± {s:.3f}")
    return fprs #{k: float(np.mean(v)) for k, v in fprs.items()}



# ------------------------------------------------------------------ TPR
def evaluate_PBRS_TPR(
    *, model, load_cifar10_corruption, BasicDataset, run_martingale, protector,
    transform, args, device,
    corruption_types, severities,
    n_examples=2000,
    seeds=range(3),
    buffer_capacity=512,
    confidence_threshold=0.5,
    num_classes=10,
    use_pbrs=True,
    log_path=None,
    verbose=True,
):
    """True-positive rate & delay under covariate-shift."""
    label   = "PBRS" if use_pbrs else "No PBRS"
    results = defaultdict(dict)
    shift_pt = n_examples                       # position where corruption starts

    for corr in corruption_types:
        for sev in severities:
            det_flags, delays = [], []

            for sd in seeds:
                np.random.seed(sd); torch.manual_seed(sd)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark     = False

                dl, _, _ = load_cifar10_corruption(
                    corruption=corr, severity=sev,
                    n_examples=n_examples, data_dir="./data",
                    transform=transform, batch_size=args.batch_size)

                # ---------- PBRS filter -----------------------------------
                if use_pbrs:
                    buf  = PBRSBuffer(buffer_capacity, num_classes)
                    z    = np.full(2*n_examples, np.nan, dtype=np.float32)
                    idx  = 0
                    with torch.no_grad():
                        for xb, _ in dl:
                            pr  = torch.softmax(model(xb.to(device)), 1)
                            ent = -torch.sum(pr * torch.log(pr+1e-8), 1)
                            yhat = torch.argmax(pr, 1)
                            conf = torch.max(pr, 1).values

                            for e, c, yh in zip(ent.cpu(), conf.cpu(), yhat.cpu()):
                                if idx == shift_pt:
                                    buf.reset()
                                if c > confidence_threshold and buf.accept(int(yh)):
                                    buf.add(idx, float(e), int(yh))
                                idx += 1
                    for j, e in buf.get_indexed_entropies():
                        z[j] = e
                else:
                    ent_list = []
                    with torch.no_grad():
                        for xb, _ in dl:
                            pr  = torch.softmax(model(xb.to(device)), 1)
                            ent_list.extend(
                                (-torch.sum(pr * torch.log(pr+1e-8), 1)).cpu().tolist())
                    z = np.asarray(ent_list, dtype=np.float32)

                # ---------- martingale ------------------------------------
                valid = z[~np.isnan(z)]
                if valid.size == 0:
                    triggered, delay = False, None
                else:
                    logS = run_martingale({"st": valid}, protector)["st"]["log_sj"]
                    if np.nanmax(logS) > np.log(100):
                        pos     = np.argmax(logS > np.log(100))
                        triggered = True
                        delay     = pos - shift_pt
                    else:
                        triggered, delay = False, None

                det_flags.append(int(triggered))
                delays.append(delay)

            # aggregate over seeds
            rate   = float(np.mean(det_flags))
            avg_d  = float(np.mean([d for d in delays if d is not None])) \
                     if any(delays) else None
            results[(corr, sev)] = dict(
                detection_rate=rate, avg_delay=avg_d,
                raw_detections=det_flags, raw_delays=delays)

            if verbose:
                print(f"[{label}] {corr} s{sev}: rate={rate:.2f} delay={avg_d}")

    return results

