# ----------------------------------------------------------------------
import numpy as np
import torch
import pandas as pd
from itertools import combinations
from collections import defaultdict
from utilities import *
from protector import PBRSBuffer


def compare_fpr_across_seeds(model, load_cifar10_label_shift_balanced, BasicDataset,
                             run_martingale, protector, transform, args, device,
                             seeds=list(range(10)),
                            buffer_capacity=512,
                            confidence_threshold=0.5,
                             num_classes_list=[1,2,3,4,5,6,7,8,9,10],
                             use_pbrs=True, log_path='fpr_results.csv'):
    label = "PBRS" if use_pbrs else "No PBRS"
    fprs = {k: [] for k in num_classes_list}

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
                #print(f"Evaluating label shift ({num_classes}-class): {subset}")
                x, y = load_cifar10_label_shift_balanced(keep_classes=subset, n_examples=8000, shift_point=4000)
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

            print(f"[DEBUG] Max log_sj = {np.nanmax(log_sj):.2f} for subset size {num_classes} with classes {subset}")

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


def evaluate_covariate_shift_detection(model, load_cifar10_corruption, BasicDataset,
                                       run_martingale, protector, transform, args, device,
                                       corruption_types=['gaussian_noise', 'brightness', 'fog'],
                                       severities=[1, 2, 3, 4, 5],
                                       seeds=range(5),
                                       buffer_capacity=512,
                                       confidence_threshold=0.8,
                                       num_classes=10,
                                       use_pbrs=True,
                                       log_path='tpr_results.csv'):
    
    label = "PBRS" if use_pbrs else "No PBRS"
    results = defaultdict(dict)

    for corruption in corruption_types:
        for severity in severities:
            detections = []
            delays = []

            for seed in seeds:
                #print(f"\nCorruption: {corruption}, Severity: {severity}, Seed: {seed}")
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                loader, is_clean, labels = load_clean_then_corrupt_sequence(
                    corruption=corruption,
                    severity=severity,
                    n_examples=4000,
                    data_dir="./data",
                    transform=transform,
                    batch_size=args.batch_size,
                )

                if use_pbrs:
                    step_idx = 0
                    total_steps = len(loader.dataset)
                    entropy_stream = [np.nan] * total_steps
                    buffer = PBRSBuffer(capacity=buffer_capacity, num_classes=num_classes)

                    with torch.no_grad():
                        for x_batch, _ in loader:
                            x_batch = x_batch.to(device)
                            logits = model(x_batch)
                            probs = torch.softmax(logits, dim=1)
                            entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                            pseudo_labels = torch.argmax(probs, dim=1)
                            max_probs = torch.max(probs, dim=1).values

                            for entropy, y_hat, conf in zip(entropies.cpu().tolist(),
                                                            pseudo_labels.cpu().tolist(),
                                                            max_probs.cpu().tolist()):
                                if step_idx == 4000:
                                    buffer.reset()
                                if conf > confidence_threshold and buffer.accept(y_hat):
                                    buffer.add(step_idx, entropy, y_hat)
                                step_idx += 1

                    accepted = buffer.get_indexed_entropies()
                    if accepted:
                        accepted_idxs = [idx for idx, _ in accepted]
                        #print(f"[DEBUG] Accepted idx range: min={min(accepted_idxs)}, max={max(accepted_idxs)}")
                        #print(f"[DEBUG] Number of accepted indices >= 4000: {sum(1 for idx in accepted_idxs if idx >= 4000)}")
                    else:
                        print("[DEBUG] No entropies accepted by PBRS")

                    for idx, ent in accepted:
                        if 0 <= idx < total_steps:
                            entropy_stream[idx] = ent

                    n_written = np.count_nonzero(~np.isnan(entropy_stream))
                    n_skipped = len([1 for idx, _ in accepted if not (0 <= idx < total_steps)])
                    #print(f"[DEBUG] Entropies written to stream: {n_written} / {total_steps}")
                    #print(f"[DEBUG] Skipped due to invalid idx: {n_skipped}")
                    ents = np.array(entropy_stream)
                else:
                    ents, _, _, _ = evaluate(model, loader, device)

                key = f"{corruption}_s{severity}_seed{seed}"
                #print(f"[DEBUG] Entropy stream summary for {key}:")
                #print(f"  - Total length: {len(ents)}")
                #print(f"  - Num NaNs: {np.isnan(ents).sum()}")
                #print(f"  - Num valid: {np.count_nonzero(~np.isnan(ents))}")
                #if np.count_nonzero(~np.isnan(ents)) > 0:
                    ##print(f"  - Min entropy: {np.nanmin(ents):.4f}")
                    #print(f"  - Max entropy: {np.nanmax(ents):.4f}")
                    #print(f"  - Mean entropy: {np.nanmean(ents):.4f}")

                # === Run martingale only on valid entropy values ===
                valid_entries = [(i, e) for i, e in enumerate(ents) if not np.isnan(e)]

                if not valid_entries:
                    #print(f"[DEBUG] All entropy values are NaN — skipping martingale for {key}")
                    log_sj = [np.nan] * len(ents)
                    triggered = False
                    delay = None
                else:
                    step_indices, valid_ents = zip(*valid_entries)
                    valid_ents = np.array(valid_ents)

                    result = run_martingale({key: valid_ents}, protector)[key]
                    log_sj_valid = result["log_sj"]
                    #print(f"[DEBUG] Max log_sj = {np.nanmax(log_sj_valid):.2f}")

                    triggered = np.nanmax(log_sj_valid) > np.log(100)
                    if triggered:
                        trigger_idx = next(i for i, val in enumerate(log_sj_valid) if val > np.log(100))
                        detection_step = step_indices[trigger_idx]
                        delay = detection_step - 4000
                    else:
                        delay = None

                detections.append(int(triggered))
                delays.append(delay)

            detection_rate = np.mean(detections)
            avg_delay = np.mean([d for d in delays if d is not None]) if any(delays) else None
            print(f"✅ Detected in {sum(detections)} / {len(seeds)} runs "
                  f"(rate = {detection_rate:.2f})  |  Avg delay: {avg_delay}")

            results[(corruption, severity)] = {
                'detection_rate': detection_rate,
                'avg_delay': avg_delay,
                'raw_detections': detections,
                'raw_delays': delays
            }

    print(f"\n📊 Logging TPR results to {log_path} for method: {label}")
    log_tpr_results(results, label=label, out_path=log_path)
    return results

def compare_fpr_across_seedsOLD(model, load_cifar10_label_shift_balanced, BasicDataset,
                             run_martingale, protector, transform, args, device,
                             seeds=list(range(10)), num_classes_list=[1,2,3,4,5,6,7,8,9],
                             use_pbrs=True, log_path='fpr_results.csv'):
    label = "PBRS" if use_pbrs else "No PBRS"
    
    fprs = {k: [] for k in num_classes_list}

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
                print(f"Evaluating label shift ({num_classes}-class): {subset}")
                x, y = load_cifar10_label_shift_balanced(keep_classes=subset, n_examples=8000, shift_point=4000)
                dataset = BasicDataset(x, y, transform=transform)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

                if use_pbrs:
                    buffer = PBRSBuffer(capacity=512, num_classes=num_classes)
                    confidence_threshold = 0.8
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
                                    buffer.add(None, entropy, y_hat)

                    ents = np.array(buffer.get_entropies())
                else:
                    ents, _, _, _ = evaluate(model, loader, device)

                key = f"{label}_{num_classes}cls_{'_'.join(map(str, subset))}"
                result = run_martingale({key: ents}, protector)[key]
                threshold_crossed[key] = np.max(result["log_sj"]) > np.log(100)

            fpr = sum(threshold_crossed.values()) / len(threshold_crossed)
            fprs[num_classes].append(fpr)

            max_log_sj = max(result["log_sj"])
            print(f"Maximum log S_j value: {max_log_sj}")

    # === Now compute final stats across all seeds ===
    print(f"\n📊 FPR summary across seeds for method: {label}")
    for num_classes in num_classes_list:
        mean_fpr = np.mean(fprs[num_classes])
        std_fpr = np.std(fprs[num_classes])
        print(f"{num_classes} classes: {label} → FPR = {mean_fpr:.3f} ± {std_fpr:.3f}")


    print(f"\n📊 Logging FPR results to {log_path} for method: {label}")
    log_fpr_results(fprs, label=label, out_path=log_path)



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

