import os, argparse, numpy as np, torch, torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utilities          import *
from experiment_logger  import (
        setup_logging, log_experiment_start, log_progress,
        save_results, log_results)
from plotting           import plot_tpr_comparison
from protector          import get_protector_from_ents
from utils_entropy_cache import get_source_entropies

# ---------- PBRS imports -------------------------------------
from PBRS_LabelShift_Evaluation import *
#from PBRS_LabelShift_Evaluation_Minimal import *

# ---------- WeightedCDF imports -------------------------------------
from WeightedCDF_LabelShift_Evaluation import (
     evaluate_weightedCDF_FPR,          # static
     evaluate_weightedCDF_TPR,
     evaluate_weightedCDFBBSE_FPR,      # static + BBSE
     evaluate_weightedCDFBBSE_TPR,
     evaluate_weightedCDFBBSEODS_FPR,   # static + BBSE + online ODS
     evaluate_weightedCDFBBSEODS_TPR,
)

# -------------------------------------------------------------
def get_methods():
    return {
        "baseline": dict(
            fpr = evaluate_PBRS_FPR,
            tpr = evaluate_PBRS_TPR,
            needs_initial_protector = True),
        "pbrs": dict(
            fpr = evaluate_PBRS_FPR,
            tpr = evaluate_PBRS_TPR,
            needs_initial_protector = True),
        "w-cdf": dict(
            fpr = evaluate_weightedCDF_FPR,
            tpr = evaluate_weightedCDF_TPR,
            needs_initial_protector = False),
        "w-bbse": dict(
            fpr = evaluate_weightedCDFBBSE_FPR,
            tpr = evaluate_weightedCDFBBSE_TPR,
            needs_initial_protector = False),
        "w-bbseods": dict(
            fpr = evaluate_weightedCDFBBSEODS_FPR,
            tpr = evaluate_weightedCDFBBSEODS_TPR,
            needs_initial_protector = False),
    }

METHODS = get_methods()

def parse_args():
    p = argparse.ArgumentParser("Label-shift / cov-shift experiments")
    p.add_argument("--method", choices=METHODS.keys(), default="w-bbseods")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--buffer_capacity", type=int, default=512)   # PBRS only
    p.add_argument("--confidence_threshold", type=float, default=0.5)  # PBRS
    p.add_argument("--seeds",   type=int, nargs="+", default=[0])
    p.add_argument("--number_classes", type=int, nargs="+",
                   default=[1,2,3,4,5,6,7,8,9,10])
    p.add_argument("--corruptions", nargs="+", default=[
        "shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise",
        "defocus_blur", "brightness", "fog", "zoom_blur", "frost",
        "glass_blur", "impulse_noise", "contrast", "jpeg_compression",
        "elastic_transform"])
    p.add_argument("--severities",  type=int, nargs="+",
                   default=[1,2,3,4,5])
    return p.parse_args()


def main():
    args = parse_args()
    method_cfg = METHODS[args.method]

    # ------------------------------------------------- logging
    log_dir = setup_logging(args)
    log_experiment_start(args)

    # ------------------------------------------------- device & tfm
    device = torch.device(args.device)
    norm   = T.Normalize((0.4914,0.4822,0.4465),(0.2471,0.2435,0.2616))

    # ------------------------------------------------- model
    log_progress("Loading model …")
    model = get_model(device)       

    # ------------------------------------------------- clean entropies
    log_progress("Computing source entropies …")
    clean_ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True,
        transform=T.Compose([T.ToTensor(), norm]))
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False)
    source_ents = get_source_entropies(model, clean_loader, device)

    # We'll pass *protector* only when PBRS needs it up-front
    protector0 = None
    if method_cfg["needs_initial_protector"]:
        protector0 = get_protector_from_ents(
            source_ents,
            argparse.Namespace(gamma=1/(16*np.sqrt(3)), eps_clip=1.5, device=device))
        log_progress("Initial protector ready (PBRS).")

    # ================================================= FPR
    log_progress(f"▶ FPR eval ({args.method}) …")
    fpr_fn = method_cfg["fpr"]
    fpr_kw = dict(
        model      = model,
        load_cifar10_label_shift = load_cifar10_label_shift,
        BasicDataset = BasicDataset,
        run_martingale = run_martingale,
        transform  = norm,
        args       = args,
        device     = device,
        seeds      = args.seeds,
        num_classes_list = args.number_classes,
        protector = protector0)
    
    if args.method in ["pbrs", "baseline"]:
        fpr_kw.update(
            protector   = protector0,
            buffer_capacity     = args.buffer_capacity,    
            confidence_threshold    = args.confidence_threshold,
            use_pbrs = (args.method == "pbrs"))

    fpr_res = METHODS[args.method]["fpr"](**fpr_kw)

    # ================================================= TPR
    log_progress(f"▶ TPR eval ({args.method}) …")
    tpr_fn = method_cfg["tpr"]
    tpr_kw = dict(
        model      = model,
        load_clean_then_corrupt_sequence = load_clean_then_corrupt_sequence,
        BasicDataset = BasicDataset,
        run_martingale = run_martingale,
        transform  = norm,
        args       = args,
        device     = device,
        corruption_types = args.corruptions,
        severities = args.severities,
        seeds      = args.seeds)

    if args.method in ["pbrs", "baseline"]:
        tpr_kw.update(
            protector   = protector0,
            buffer_capacity     = args.buffer_capacity,
            confidence_threshold    = args.confidence_threshold,
            use_pbrs = (args.method == "pbrs"))

    tpr_res = METHODS[args.method]["tpr"](**tpr_kw)

    # ================================================= save / plot
    save_results({args.method: fpr_res}, log_dir, "fpr")
    save_results({args.method: tpr_res}, log_dir, "tpr")
    log_results({args.method: fpr_res},{args.method: tpr_res})

    plot_path = os.path.join(log_dir,f"{args.method}_tpr_plot.png")
    plot_tpr_comparison({args.method: tpr_res}, plot_path)
    log_progress(f"Finished - plot saved to {plot_path}")


if __name__ == "__main__":
    main()