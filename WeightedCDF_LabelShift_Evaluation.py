import numpy as np, torch, pandas as pd
from collections import defaultdict
from itertools import combinations
from torch.utils.data import DataLoader, Subset


from weighted_cdf          import WeightedCDF
from weighted_cdf_bbse     import BBSEWeightedCDF
from weighted_cdf_bbse_ods import (
        BBSEODSWeightedCDF,
        estimate_confusion_matrix,
        estimate_target_distribution_from_preds,
)

# -----------------------------------------------------------------------
#  lightweight martingale step (we use this to incrementally increase Martingale as opposed to running it all in one go)
# -----------------------------------------------------------------------
def mart_step(ent, protector):
    u = protector.cdf(ent)
    protector.protect_u(u)
    return np.log(protector.martingales[-1] + 1e-8)

def _no_update(self, *a, **kw):         
    pass


# -----------------------------------------------------------------------
#  FACTORY returning a ready-to-use Protector for any variant
# -----------------------------------------------------------------------
def build_protector(
        variant:str,
        model, dl_clean, dl_shift,
        device,
        gamma=1/(8*np.sqrt(3)), eps_clip=1.8):

    if variant == "plain":        # ---------------- WeightedCDF
        ents, yhat = [], []
        with torch.no_grad():
            for xb,_ in dl_clean:
                pr  = torch.softmax(model(xb.to(device)),1)
                ents.extend((-torch.sum(pr*torch.log(pr+1e-8),1)).cpu().tolist())
                yhat.extend(torch.argmax(pr,1).cpu().tolist())

        p_s = torch.bincount(torch.tensor(yhat), minlength=10).float()
        p_s /= p_s.sum()
        p_t = estimate_target_distribution_from_preds(model, dl_shift, device)
        cdf = WeightedCDF(ents, yhat, p_s.numpy(), p_t.cpu().numpy())

        # make it look like the ODS class (add dummy updater)
        cdf.batch_ods_update = _no_update.__get__(cdf)

    elif variant == "bbse":       # ---------------- static BBSE
        ents, y_true = [], []
        with torch.no_grad():
            for xb,yb in dl_clean:
                pr  = torch.softmax(model(xb.to(device)),1)
                ents.extend((-torch.sum(pr*torch.log(pr+1e-8),1)).cpu().tolist())
                y_true.extend(yb.cpu().tolist())

        C, p_s     = estimate_confusion_matrix(model, dl_clean, device)
        p_t_pred   = estimate_target_distribution_from_preds(model, dl_shift, device)
        # BBSE correction: solve for p_t_true
        ridge = 1e-6 * torch.eye(len(p_s), device=p_s.device)
        C_T = C.t() + ridge
        # --- Fix for MPS ---
        C_T_cpu = C_T.cpu()
        p_t_pred_cpu = p_t_pred.cpu()
        p_t_true_cpu = torch.linalg.solve(C_T_cpu, p_t_pred_cpu)
        p_t_true = p_t_true_cpu.to(C_T.device)
        w = p_t_true / p_s
        w = torch.clamp(w, min=0.1, max=10.0)
        cdf = BBSEWeightedCDF(ents, y_true, w)

        cdf.batch_ods_update = _no_update.__get__(cdf)

    elif variant == "bbseods":    # ---------------- online BBSE+ODS
        ents, y_true = [], []
        with torch.no_grad():
            for xb,yb in dl_clean:
                pr  = torch.softmax(model(xb.to(device)),1)
                ents.extend((-torch.sum(pr*torch.log(pr+1e-8),1)).cpu().tolist())
                y_true.extend(yb.cpu().tolist())
        C, p_s     = estimate_confusion_matrix(model, dl_clean, device)
        p_t_pred   = estimate_target_distribution_from_preds(model, dl_shift, device)

        cdf = BBSEODSWeightedCDF(
                entropies        = ents,
                p_test           = p_t_pred,
                p_source         = p_s,
                source_labels    = y_true,
                confusion_matrix = C,
                device           = device,
                ods_alpha        = 0.05,
                num_classes      = 10)

    else:
        raise ValueError(f"unknown variant {variant}")

    # ---------------- wrap into POEM Protector --------------------------
    from protector import Protector
    prot = Protector(cdf=cdf, device=device)
    prot.set_gamma(gamma)
    prot.set_eps_clip_val(eps_clip)
    return prot
# =======================================================================



# =======================================================================
#  GENERIC COV-SHIFT  (TPR)
# =======================================================================
def evaluate_weightedCDF_TPR_variant(
        variant:str,
        *,
        model, load_clean_then_corrupt_sequence, BasicDataset,
        transform, args, device,
        corruption_types, severities, seeds,
        split=2000, log_path=None, **_):

    label   = f"Weighted-{variant.upper()}"
    results = defaultdict(dict)

    for corr in corruption_types:
        for sev in severities:
            dets, delays = [], []
            for sd in seeds:
                np.random.seed(sd); torch.manual_seed(sd)

                loader,_,_ = load_clean_then_corrupt_sequence(
                    corruption=corr, severity=sev, n_examples=split,
                    data_dir="./data", transform=transform,
                    batch_size=args.batch_size)

                dl_c = DataLoader(Subset(loader.dataset, range(0, split)),
                                  batch_size=args.batch_size, shuffle=False)
                dl_s = DataLoader(Subset(loader.dataset, range(split, 2*split)),
                                  batch_size=args.batch_size, shuffle=False)

                prot = build_protector(variant, model, dl_c, dl_s, device)

                trig, det_pos, idx = False, None, 0
                for xb,_ in loader:
                    pr   = torch.softmax(model(xb.to(device)),1)
                    ents = (-torch.sum(pr*torch.log(pr+1e-8),1)).cpu().tolist()
                    yhat = torch.argmax(pr,1).cpu().tolist()

                    # ODS update (no-op for static variants)
                    prot.cdf.batch_ods_update(yhat)

                    for e in ents:
                        if not trig and mart_step(e, prot) > np.log(100):
                            trig, det_pos = True, idx
                        idx += 1

                dets.append(int(trig))
                delays.append(det_pos - split if trig else None)

            rate = float(np.mean(dets))
            avgd = (float(np.mean([d for d in delays if d is not None]))
                    if any(d is not None for d in delays) else None)
            print(f"[{label}] {corr}-s{sev}  TPR={rate:.2f}  delay={avgd}")
            results[(corr,sev)] = dict(detection_rate=rate,
                                       avg_delay=avgd,
                                       raw_detections=dets,
                                       raw_delays=delays)

    if log_path:
        rows=[dict(Corr=c,S=sv,M=label,TPR=v['detection_rate'],Delay=v['avg_delay'])
              for (c,sv),v in results.items()]
        pd.DataFrame(rows).to_csv(log_path,index=False)
    return results



# =======================================================================
#  GENERIC LABEL-SHIFT  (FPR)
# =======================================================================
def evaluate_weightedCDF_FPR_variant(
        variant:str,
        *,
        model, load_cifar10_label_shift, BasicDataset,
        transform, args, device,
        num_classes_list, seeds,
        split=2000, log_path=None, **_):

    label = f"Weighted-{variant.upper()}"
    fprs  = {k: [] for k in num_classes_list}

    print(f"[DEBUG] Starting FPR evaluation for {label}")
    print(f"[DEBUG] Testing {len(seeds)} seeds, {len(num_classes_list)} class sizes")

    for sd in seeds:
        print(f"\n[DEBUG] Seed {sd}")
        np.random.seed(sd); torch.manual_seed(sd)
        for k in num_classes_list:
            fires=[]
            for subset in combinations(range(10), k):
                x,y = load_cifar10_label_shift(subset, n_examples=2*split, shift_point=split)
                ds  = BasicDataset(x,y,transform=transform)


                if len(ds) < 2*split:  #When evaluating small subsets, we may need to reduce split to half as there are not enough examples. This does not effect the results.
                        split = len(ds) // 2       

                dl_c = DataLoader(Subset(ds, range(0, split)),
                                  batch_size=args.batch_size, shuffle=False)
                dl_s = DataLoader(Subset(ds, range(split, 2*split)),
                                  batch_size=args.batch_size, shuffle=False)
                dl_a = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

                prot = build_protector(variant, model, dl_c, dl_s, device)
                #print(f"[DEBUG] Protector gamma={prot.gamma:.6f}, epsilon={prot.last_eps:.6f}")

                triggered=False
                for batch_idx, (xb,_) in enumerate(dl_a):
                    pr=torch.softmax(model(xb.to(device)),1)
                    ents=(-torch.sum(pr*torch.log(pr+1e-8),1)).cpu().tolist()
                    yhat=torch.argmax(pr,1).cpu().tolist()

                    prot.cdf.batch_ods_update(yhat)

                    for e in ents:
                        logsj = mart_step(e, prot)
                        #print(f"[DEBUG] logSj={logsj:.3f}, epsilon={prot.last_eps:.3f}")
                        if logsj > np.log(100):
                            triggered=True
                            #print(f"[DEBUG] Triggered at batch {batch_idx}")
                            break                    
                    
                    if triggered: break
                fires.append(triggered)
            fprs[k].append(np.mean(fires))
            print(f"[DEBUG] Class size {k} FPR: {fprs[k][-1]:.3f}")

    #mean_fpr={k:float(np.mean(v)) for k,v in fprs.items()}
    #print(f"[DEBUG] Final results:")
    #print(f"[{label}] mean FPR:",mean_fpr)

    print(f"\nFPR summary across seeds for method: {label}")
    for num_classes in num_classes_list:
        mean_fpr = np.mean(fprs[num_classes])
        std_fpr = np.std(fprs[num_classes])
        print(f"{num_classes} classes: {label} → FPR = {mean_fpr:.3f} ± {std_fpr:.3f}")

    if log_path:
        print(f"[DEBUG] Saving results to {log_path}")
        pd.DataFrame([dict(Size=k,FPR=v) for k,v in mean_fpr.items()]) \
          .to_csv(log_path,index=False)
    return fprs #mean_fpr



# -----------------------------------------------------------------------
#  EXPORTED convenience wrappers 
# -----------------------------------------------------------------------
def evaluate_weightedCDF_TPR(**kw):
    return evaluate_weightedCDF_TPR_variant("plain", **kw)

def evaluate_weightedCDF_FPR(**kw):
    return evaluate_weightedCDF_FPR_variant("plain", **kw)

def evaluate_weightedCDFBBSE_TPR(**kw):
    return evaluate_weightedCDF_TPR_variant("bbse", **kw)

def evaluate_weightedCDFBBSE_FPR(**kw):
    return evaluate_weightedCDF_FPR_variant("bbse", **kw)

def evaluate_weightedCDFBBSEODS_TPR(**kw):
    return evaluate_weightedCDF_TPR_variant("bbseods", **kw)

def evaluate_weightedCDFBBSEODS_FPR(**kw):
    return evaluate_weightedCDF_FPR_variant("bbseods", **kw)