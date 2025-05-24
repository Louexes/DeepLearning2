# Label Shift & Covariate Shift Detection Experiments

This repository provides a framework for evaluating methods to detect distribution shifts (label shift and covariate/covariate shift) in image classification, with a focus on the CIFAR-10 dataset. The codebase supports several state-of-the-art and baseline methods, robust experiment logging, and result visualization.

---

## Table of Contents

- [Label Shift \& Covariate Shift Detection Experiments](#label-shift--covariate-shift-detection-experiments)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Methods](#methods)
  - [Repository Structure](#repository-structure)
  - [Setup \& Installation](#setup--installation)
  - [Running Experiments](#running-experiments)
  - [Experiment Arguments](#experiment-arguments)
  - [Results \& Logging](#results--logging)
  - [Extending the Framework](#extending-the-framework)
  - [Troubleshooting](#troubleshooting)
  - [Citations](#citations)
  - [Contact](#contact)

---

## Overview

The goal of this repository is to compare different statistical and learning-based methods for detecting when the test data distribution has shifted from the training distribution. The main focus is on:
- **Label Shift:** The class distribution changes between training and test data.
- **Covariate Shift:** The input distribution changes (e.g., via image corruptions).

Detection is performed using entropy-based martingale tests and various weighting/protection schemes.

---

## Methods

The following methods are implemented and can be selected via the `--method` argument:

- **baseline:** No adaptation; runs the martingale test on the raw entropy stream.
- **pbrs:** PBRS filtering (buffer and confidence threshold).
- **w-cdf:** Weighted CDF using importance weighting based on estimated label distributions.
- **w-bbse:** Black Box Shift Estimation (BBSE) for more accurate weighting.
- **w-bbseods:** BBSE with Online Distribution Shift (ODS) corrections.

Each method is evaluated for both False Positive Rate (FPR, under label shift) and True Positive Rate (TPR, under covariate shift).

---

## Repository Structure

```
.
├── Run_LabelShift_Experiments.py      # Main experiment runner
├── PBRS_LabelShift_Evaluation.py      # PBRS evaluation logic
├── WeightedCDF_LabelShift_Evaluation.py # Weighted CDF, BBSE, ODS logic
├── utilities.py                       # Data loading, model helpers, martingale, etc.
├── experiment_logger.py               # Logging and result saving
├── plotting.py                        # Plotting utilities
├── protector.py                       # Protector/CDF classes
├── utils_entropy_cache.py             # Entropy caching helpers
├── data/                              # CIFAR-10 and CIFAR-10-C data
└── logs/                              # Experiment logs and results
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install dependencies:**
   - Python 3.8+
   - PyTorch (with CUDA or MPS support if available)
   - torchvision
   - numpy, pandas, tqdm, matplotlib

   You can install the main dependencies with:
   ```bash
   pip install torch torchvision numpy pandas tqdm matplotlib
   ```

3. **Download CIFAR-10 and CIFAR-10-C:**
   - The code will automatically download CIFAR-10 if not present.
   - For CIFAR-10-C, download from [CIFAR-10-C GitHub](https://github.com/hendrycks/robustness) and place the files in `./data/CIFAR-10-C/`.

---

## Running Experiments

The main entry point is `Run_LabelShift_Experiments.py`. You can run experiments with different methods and settings using command-line arguments.

**Example: Run the baseline method**
```bash
python Run_LabelShift_Experiments.py --method baseline
```

**Example: Run PBRS with 5 seeds**
```bash
python Run_LabelShift_Experiments.py --method pbrs --seeds 0 1 2 3 4
```

**Example: Run weighted CDF**
```bash
python Run_LabelShift_Experiments.py --method w-cdf
```

**All arguments can be listed with:**
```bash
python Run_LabelShift_Experiments.py --help
```

---

## Experiment Arguments

Key arguments (with defaults):

- `--method`  
  Which method to use (`baseline`, `pbrs`, `w-cdf`, `w-bbse`, `w-bbseods`).

- `--batch_size`  
  Batch size for data loading (default: 64).

- `--device`  
  Device to use (`cuda`, `mps`, or `cpu`).

- `--buffer_capacity`  
  Buffer size for PBRS (default: 512).

- `--confidence_threshold`  
  Confidence threshold for PBRS (default: 0.5).

- `--seeds`  
  List of random seeds (default: `[0]`).

- `--number_classes`  
  List of class counts to test for label shift (default: `[1,2,3,4,5,6,7,8,9,10]`).

- `--corruptions`  
  List of corruption types for covariate shift (see code for defaults).

- `--severities`  
  List of severity levels for corruptions (default: `[1,2,3,4,5]`).

---

## Results & Logging

- **Logs and results** are saved in a timestamped directory under `./logs/`.
- **FPR and TPR results** are saved as CSV and JSON files.
- **Plots** (e.g., TPR comparison) are saved as PNG files.
- **Console output** provides progress, debug info, and summary statistics.

---

## Extending the Framework

- **Add new methods:**  
  Implement your method as a function and add it to the `METHODS` dictionary in `Run_LabelShift_Experiments.py`.

- **Add new datasets:**  
  Add new data loaders to `utilities.py` and update experiment logic as needed.

- **Change evaluation logic:**  
  Modify or extend the evaluation functions in the relevant files.

---

## Troubleshooting

- **Not enough samples for a class subset:**  
  The code oversamples with replacement to ensure enough data for each experiment.

- **Device errors (e.g., MPS):**  
  Some PyTorch operations may not be supported on MPS. The code includes workarounds for most cases.

- **Missing dependencies:**  
  Install all required Python packages as listed above.

- **Slow runs:**  
  Use a GPU (`cuda` or `mps`) for best performance.

---

## Citations

If you use this codebase in your research, please cite the relevant papers for PBRS, BBSE, and ODS methods as well as the CIFAR-10 and CIFAR-10-C datasets.

---

## Contact

For questions or contributions, please open an issue or pull request on the repository.

---

**Happy experimenting!**