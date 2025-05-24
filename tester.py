# ──────────────────────────────────────────────────────────────────────────────
#  quick_debug.py   – skip heavy eval, feed random data to the logger
# ──────────────────────────────────────────────────────────────────────────────
import os, shutil, json, numpy as np, torch, random, tempfile
from experiment_logger import setup_logging, log_experiment_start, log_progress
from experiment_logger import save_results, log_results
from plotting           import plot_tpr_comparison               # your helper

# -----------------------------------------------------------------------------
# -------------------------------------------------------------
#  Fake data generator that *looks* like the real evaluator output
# -------------------------------------------------------------
import numpy as np

def make_fake_results(methods      = ("pbrs", "w-cdf", "w-bbse", "w-bbseods"),
                      class_sizes  = range(1, 11),            # 1-10 classes
                      corruptions  = ("gaussian_noise", "brightness", "fog"),
                      severities   = range(1, 6),              # 1-5
                      num_seeds    = 3,                        # same as main()
                      rng_seed     = 42):

    rng = np.random.default_rng(rng_seed)
    fpr_dict, tpr_dict = {}, {}

    for m in methods:
        # ----------------------- FPR ----------------------------------------
        # one *list of per-seed scores* for every subset size
        fpr_dict[m] = {
            k: rng.uniform(0.0, 0.6, size=num_seeds).tolist()
            for k in class_sizes
        }

        # ----------------------- TPR ----------------------------------------
        tpr = {}
        for corr in corruptions:
            for sev in severities:
                raw_det = rng.integers(0, 2,   size=num_seeds).tolist()
                raw_del = rng.integers(20, 250, size=num_seeds).tolist()

                tpr[(corr, sev)] = dict(
                    detection_rate = float(np.mean(raw_det)),
                    avg_delay      = float(np.mean(raw_del)),
                    raw_detections = raw_det,
                    raw_delays     = [float(x) for x in raw_del],
                )
        tpr_dict[m] = tpr

    return fpr_dict, tpr_dict

# -----------------------------------------------------------------------------
def main():
    # ---- make a temporary log dir ------------------------------------------
    log_dir = "debug_logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # ---- set up basic logging ----------------------------------------------
    class DummyArgs: pass
    args = DummyArgs(); args.__dict__.update(method="debug")
    setup_logging(args)                    # writes experiment.log
    log_experiment_start(args)

    # ---- fabricate results --------------------------------------------------
    fpr_fake, tpr_fake = make_fake_results()
    log_progress("🧪 Fabricated fake results")

    # ---- persist / log / plot ----------------------------------------------
    save_results(fpr_fake, log_dir, "fpr")
    save_results(tpr_fake, log_dir, "tpr")

    log_results(fpr_fake, tpr_fake)

    # single-method comparison plot (just pick one for demo)
    plot_path = os.path.join(log_dir, "tpr_comparison.png")
    plot_tpr_comparison({"pbrs": tpr_fake["pbrs"]}, save_path=plot_path)
    log_progress(f"📈 fake plot written to {plot_path}")

    # show where stuff lives
    print("\nArtifacts written under:", os.path.abspath(log_dir))
    print(" –", os.listdir(log_dir))

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()