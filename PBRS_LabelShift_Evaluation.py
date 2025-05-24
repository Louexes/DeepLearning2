# ----------------------------------------------------------------------
import numpy as np
import torch
import pandas as pd
from itertools import combinations
from collections import defaultdict
from utilities import *
from protector import PBRSBuffer

def evaluate_PBRS_FPR(model, load_cifar10_label_shift, BasicDataset,
                             run_martingale, protector, transform, args, device,
                             seeds=list(range(10)),
                            buffer_capacity=512,
                            confidence_threshold=0.5,
                             num_classes_list=[1,2,3,4,5,6,7,8,9,10],
                             use_pbrs=True, log_path='fpr_results.csv'):
    label = "PBRS" if use_pbrs else "No PBRS"
    fprs = {k: [] for k in num_classes_list}

    print(f"Use PBRS: {use_pbrs}")

    for seed in seeds:
        print(f"\n🔁 Running with seed: {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for num_classes in num_classes_list:
            candidate_class_sets = list(combinations(range(10), num_classes))
            np.random.shuffle(candidate_class_sets)
            candidate_class_sets = candidate_class_sets[:30]

            threshold_crossed = {}

            for subset in candidate_class_sets:
                n_examples = 4000
                shift_point = 2000
                x, y = load_cifar10_label_shift(keep_classes=subset, n_examples=n_examples, shift_point=shift_point)
                dataset = BasicDataset(x, y, transform=transform)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

                if use_pbrs:
                    buffer = PBRSBuffer(capacity=buffer_capacity, num_classes=num_classes)
                    confidence_threshold = confidence_threshold
                    step_idx = 0
                    entropy_stream = [np.nan] * len(dataset)

                    with torch.no_grad():
                        for x_batch, _ in loader:
                            x_batch = x_batch.to(device)
                            logits = model(x_batch)
                            probs = torch.softmax(logits, dim=1)
                            entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                            pseudo_labels = torch.argmax(probs, dim=1)
                            max_probs = torch.max(probs, dim=1).values

                            for entropy, y_hat, confidence in zip(entropies.cpu().tolist(),
                                                                  pseudo_labels.cpu().tolist(),
                                                                  max_probs.cpu().tolist()):
                                if confidence > confidence_threshold and buffer.accept(y_hat):
                                    buffer.add(step_idx, entropy, y_hat)
                                step_idx += 1

                    for idx, ent in buffer.get_indexed_entropies():
                        if 0 <= idx < len(entropy_stream):
                            entropy_stream[idx] = ent

                    ents = np.array(entropy_stream)
                else:
                    ents, _, _, _ = evaluate(model, loader, device)

                key = f"{label}_{num_classes}cls_{'_'.join(map(str, subset))}"

                valid_entries = [(i, e) for i, e in enumerate(ents) if not np.isnan(e)]
                if not valid_entries:
                    log_sj = [np.nan] * len(ents)
                    triggered = False
                else:
                    _, valid_ents = zip(*valid_entries)
                    valid_ents = np.array(valid_ents)
                    result = run_martingale({key: valid_ents}, protector)[key]
                    log_sj = result["log_sj"]
                    triggered = np.nanmax(log_sj) > np.log(100)
        
                threshold_crossed[key] = triggered

            #print(f"[DEBUG] Max log_sj = {np.nanmax(log_sj):.2f} for subset size {num_classes} with classes {subset}. Detector was triggered: {triggered}")
            print(f"[DEBUG] Threshold crossed in {sum(threshold_crossed.values())}/{len(threshold_crossed)} cases ({sum(threshold_crossed.values())/len(threshold_crossed)*100:.1f}%)")

            fpr = sum(threshold_crossed.values()) / len(threshold_crossed)
            fprs[num_classes].append(fpr)

    print(f"\n📊 FPR summary across seeds for method: {label}")
    for num_classes in num_classes_list:
        mean_fpr = np.mean(fprs[num_classes])
        std_fpr = np.std(fprs[num_classes])
        print(f"{num_classes} classes: {label} → FPR = {mean_fpr:.3f} ± {std_fpr:.3f}")

    print(f"\n📊 Logging FPR results to {log_path} for method: {label}")
    log_fpr_results(fprs, label=label, out_path=log_path)
    return {k: float(np.mean(v)) for k, v in fprs.items()}


def log_fpr_results(fpr_dict, label, out_path='fpr_results.csv'):
    rows = []
    for k, fpr_list in fpr_dict.items():
        rows.append({
            'Subset Size': k,
            'Method': label,
            'FPR Mean': np.mean(fpr_list),
            'FPR Std': np.std(fpr_list)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, mode='a', header=not pd.read_csv(out_path).empty if os.path.exists(out_path) else True, index=False)


def evaluate_PBRS_TPR(
        *,
        model,
        load_clean_then_corrupt_sequence,
        BasicDataset,
        run_martingale,
        protector,
        transform,
        args,
        device,
        corruption_types,
        severities,
        n_examples      = 2000,   #  <-- knob ❶  (clean part = n_examples, corrupt part = n_examples)
        seeds           = range(3),#  <-- knob ❷
        buffer_capacity = 512,     #  <-- PBRS knob
        confidence_threshold  = 0.5,     #  <-- PBRS knob
        num_classes     = 10,
        use_pbrs        = True,
        log_path        = None):   #  set to None to skip CSV logging
    """
    Fast TPR evaluation under covariate-shift streams.
    Returns: dict[(corruption, severity)] -> {detection_rate, avg_delay, …}
    """
    label   = "PBRS" if use_pbrs else "No PBRS"
    results = defaultdict(dict)
    shift_pt = n_examples                # sample-index where corruption begins

    for corr in corruption_types:
        for sev in severities:
            detections, delays = [], []

            for sd in seeds:
                # reproducibility -------------------------------------------------
                np.random.seed(sd);  torch.manual_seed(sd)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark     = False

                # stream loader ---------------------------------------------------
                loader, _, _ = load_clean_then_corrupt_sequence(
                    corruption = corr,
                    severity   = sev,
                    n_examples = n_examples,
                    data_dir   = "./data",
                    transform  = transform,
                    batch_size = args.batch_size,
                )

                # ------------------------------------------------ PBRS branch ---
                if use_pbrs:
                    buf   = PBRSBuffer(capacity=buffer_capacity, num_classes=num_classes)
                    ents  = [np.nan] * (2 * n_examples)           # placeholder
                    step  = 0

                    with torch.no_grad():
                        for xb, _ in loader:
                            xb  = xb.to(device)
                            pr  = torch.softmax(model(xb), dim=1)
                            ent = -torch.sum(pr * torch.log(pr + 1e-8), dim=1)
                            yhat = torch.argmax(pr, dim=1)
                            conf = torch.max(pr,  dim=1).values

                            for e, y, c in zip(ent.cpu(), yhat.cpu(), conf.cpu()):
                                if step == shift_pt:
                                    buf.reset()                   # discard clean part
                                if c > confidence_threshold and buf.accept(int(y)):
                                    buf.add(step, float(e), int(y))
                                step += 1

                    for idx, e in buf.get_indexed_entropies():
                        ents[idx] = e

                    ents = np.asarray(ents, dtype=np.float32)
                # ------------------------------------------------ baseline -----
                else:
                    ent_list = []
                    with torch.no_grad():
                        for xb, _ in loader:
                            xb  = xb.to(device)
                            pr  = torch.softmax(model(xb), dim=1)
                            ent_list.extend(
                                (-torch.sum(pr * torch.log(pr + 1e-8), dim=1)).cpu().tolist()
                            )
                    ents = np.array(ent_list)
                    #print(f"ents: {ents}")

                # ---------------- run martingale on valid entropy values -------------
                mask = ~np.isnan(ents)          # boolean mask of finite entries
                if not mask.any():              # nothing to test
                    triggered, delay = False, None
                    print("No valid entropies")
                else:
                    ents_valid   = ents[mask]                # 1-D float array
                    idx_valid    = np.nonzero(mask)[0]       # integer indices that survive

                    res   = run_martingale({"stream": ents_valid}, protector)["stream"]
                    logSj = np.array(res["log_sj"])

                    print(f"logSj: {np.nanmax(logSj)}")

                    triggered = np.nanmax(logSj) > np.log(100)
                    delay     = None
                    if triggered:
                        trig_pos      = np.argmax(logSj > np.log(100))
                        detect_sample = idx_valid[trig_pos]
                        delay         = detect_sample - shift_pt           # ≥ 0

                detections.append(int(triggered))
                delays.append(delay)

            # ---------------- aggregate over seeds -----------------------------
            det_rate = float(np.mean(detections))
            avg_del  = float(np.mean([d for d in delays if d is not None])) \
                       if any(d is not None for d in delays) else None

            print(f"[{label}] {corr} s{sev}: detected {sum(detections)}/{len(seeds)} "
                  f"| delay = {avg_del}")

            results[(corr, sev)] = dict(
                detection_rate = det_rate,
                avg_delay      = avg_del,
                raw_detections = detections,
                raw_delays     = delays,
            )

    # optional CSV --------------------------------------------------------------
    if log_path is not None:
        import pandas as pd, os
        rows = []
        for (c, s), v in results.items():
            rows.append(dict(Corruption=c, Severity=s, Method=label,
                             DetRate=v['detection_rate'], AvgDelay=v['avg_delay']))
        mode  = 'a' if os.path.exists(log_path) else 'w'
        pd.DataFrame(rows).to_csv(log_path, mode=mode, header=(mode=='w'), index=False)
        print(f"↳ logged to {log_path}")

    return results



def log_tpr_results(results, label, out_path='tpr_results.csv'):
    rows = []
    for (corruption, severity), stats in results.items():
        rows.append({
            'Corruption': corruption,
            'Severity': severity,
            'Method': label,
            'Detection Rate': stats['detection_rate'],
            'Average Delay': stats['avg_delay'],
        })
    pd.DataFrame(rows).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)

