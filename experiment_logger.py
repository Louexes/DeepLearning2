import os
import json
import logging
from datetime import datetime
import numpy as np
def setup_logging(args):
    """Set up logging configuration and create logs directory."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'experiment.log')),
            logging.StreamHandler()
        ]
    )
    
    # Save experiment configuration
    config = vars(args)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    return log_dir

def save_results(results, log_dir, result_type):
    """
    Persist FPR / TPR dictionaries to <log_dir>/<result_type>/<method>_results.json

    Parameters
    ----------
    results      : {method -> result-dict}
                   • FPR  → {subset_size : float}
                   • TPR  → {(corr,sever) | str key : {
                                   detection_rate  : float,
                                   avg_delay       : float | None,
                                   raw_detections  : [int],
                                   raw_delays      : [float | None] }}
    log_dir      : root folder created by `setup_logging`
    result_type  : "fpr" | "tpr"
    """
    out_dir = os.path.join(log_dir, result_type)
    os.makedirs(out_dir, exist_ok=True)

    for method, res in results.items():

        # ---------- FPR: keys are already simple (ints) -----------------------
        # ---------- FPR branch (robust) -------------------------------
        if result_type == "fpr":
            serializable = {}
            for k, v in res.items():
                # v can be either a list OR a single float
                if isinstance(v, (list, tuple)):
                    vals = list(map(float, v))
                else:                           # single number → promote to 1-element list
                    vals = [float(v)]

                serializable[str(k)] = {
                    "mean": float(np.mean(vals)),
                    "std" : float(np.std(vals)),
                    "all" : vals,
                }

        # ---------- TPR: keys may be tuples (corruption, severity) ------------
        else:  # "tpr"
            serializable = {}
            for k, v in res.items():

                # ensure JSON-friendly key
                key_str = (f"{k[0]}_s{k[1]}"        # ('fog',3) → "fog_s3"
                           if isinstance(k, tuple) else str(k))

                serializable[key_str] = {
                    "detection_rate": float(v["detection_rate"]),
                    "avg_delay"     : (None if v["avg_delay"] is None
                                       else float(v["avg_delay"])),
                    "raw_detections": [int(x)             for x in v["raw_detections"]],
                    "raw_delays"    : [(None if d is None else float(d))
                                       for d in v["raw_delays"]],
                }

        # ----------- write ----------------------------------------------------
        out_file = os.path.join(out_dir, f"{method}_results.json")
        with open(out_file, "w") as f:
            json.dump(serializable, f, indent=4)

        # (optional) console confirmation
        print(f"✓ saved {result_type.upper()} results → {out_file}")
# ------------------------------------------------------------------

def log_experiment_start(args):
    """Log the start of the experiment with configuration."""
    logging.info("Starting experiments with configuration:")
    logging.info(json.dumps(vars(args), indent=4))

def log_progress(message):
    """Log a progress message."""
    logging.info(message)

# ------------------------------------------------------------------
def log_results(fpr_results, tpr_results):
    """Pretty console summary for FPR / TPR dictionaries."""
    # ---------- FPR ------------------------------------------------
    logging.info("\n📊 Summary of FPR Results:")
    for method, res in fpr_results.items():
        logging.info(f"\n{method.upper()} Results:")
        for n_cls, vals in res.items():
            # vals is either a float (mean) or a dict with mean/std
            if isinstance(vals, dict):
                msg = f"{n_cls} classes: FPR = {vals['mean']:.3f} ± {vals['std']:.3f}"
            elif isinstance(vals, (list, tuple)):
                msg = f"{n_cls} classes: FPR = {np.mean(vals):.3f} ± {np.std(vals):.3f}"
            else:
                msg = f"{n_cls} classes: FPR = {vals:.3f}"
            logging.info(msg)

    # ---------- TPR ------------------------------------------------
    logging.info("\n📊 Summary of TPR Results:")
    for method, res in tpr_results.items():
        logging.info(f"\n{method.upper()} Results:")
        for corr_key, vals in res.items():
            dt = vals["detection_rate"]
            ad = vals["avg_delay"]
            ad_str = f"{ad:.1f}" if ad is not None else "—"
            logging.info(f"{corr_key}: TPR = {dt:.3f}, Avg Delay = {ad_str}")