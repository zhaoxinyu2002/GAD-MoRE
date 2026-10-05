# Environment and reproduction audit

Audit completed: 2026-10-05. Released source revision: `658b023`.

## What the environment pins establish

The original container currently imports Python 3.11.6, PyTorch 2.4.0+cu121,
PyTorch Geometric 2.6.1, Geoopt 0.5.0, NumPy 1.26.4, SciPy 1.12.0,
pandas 2.2.3, and scikit-learn 1.6.1. These are observed imports, rather than
versions inferred from a generic requirements list. This is the recovered
container environment; there was no dated package lock accompanying the
historical result CSVs.

A separate Python 3.11.6 environment was installed on an A100 machine with
these core versions. Its full dependency snapshot is
[`requirements-lock.txt`](../requirements-lock.txt). Dependency consistency
checks passed. The original container includes unrelated packages and a
mixture of system and user-site packages, which are deliberately not copied
into the clean environment.

## Controls

- All eleven released `.mat` files have identical SHA-256 hashes to the
  corresponding files in the original project.
- The original project's release commit `0eb20c6` and public revision
  `658b023` have the same model and training computations. Their differences
  concern device selection, paths, messages, and configuration lookup messages.
- Both old and new scientific-computing environments select randomized PCA
  for cora, citeseer, and Facebook. Matching the solver does not imply
  bitwise-identical features across numerical libraries.
- The current dataset loader always computes features from the raw files.
  Historical `.pt` caches are not loaded, and copying or deleting them does
  not change this code path.
- The default model configuration is used without optional JSON overrides.
  Source and target dataset order, 32 feature dimensions, and 40 training
  epochs remain unchanged.
- In the same pinned environment, the archived implementation and release
  produce exactly equal features, labels, and normalized adjacency matrices
  for all eleven graphs, despite using different implementations of Laplacian
  feature scoring. Their model parameter names and initialized values also
  match exactly at seed 0 with the historical model configuration. The
  machine-readable record, source hashes, and data hashes are in
  [`environment-audit.json`](environment-audit.json).
- With equal initialization and random seeds, two successive small synthetic
  CPU batches produce exactly equal losses and all parameter gradients for
  the archived model and the release with `w_message=0`. The second batch
  uses three identically populated memories per expert. This checks code equivalence separately
  from the full benchmark measurements; it is not a benchmark result.

## Recovered historical training objective

The release defaults to `--w_message 1.0`. This adds a sum of neighborhood
similarities to the training objective via `max_message`. It is separate
from the embedding, feature, structure, and router-entropy loss weights.

An archived `GRACE+pre` implementation associated with the same reference
CSV values does not contain this additional training loss. Its model and training entry point contain no `w_message` term. Its test-time score is the same embedding-reconstruction distance
used by the release. Setting `--w_message 0` therefore restores the
historical objective without editing model code. The reference CSV parameter
blocks do not record `w_message`, so source history is the evidence for this
setting, not a parameter recorded alongside the CSV. A complete five-trial run with
`--w_message 0` in the isolated pinned A100 environment finished all 40
epochs per trial. Its saved CSVs agree with independently recomputed log
means and population standard deviations.

## Completed measurements

Values below are unweighted averages over the seven target datasets. The
historical row summarizes the released CSVs; the other rows summarize
completed runs, not inferred scores.

| Setting | Completed seeds | Mean AUROC | Mean AUPRC |
| --- | --- | --- | --- |
| Historical reference CSVs | 0–4 (five-trial reference) | 0.8209 | 0.3696 |
| Released default, unpinned new environment, A100 | 0–4 | 0.7702 | 0.2844 |
| Released default, recovered original environment, V100 | 0–2 | 0.7711 | 0.2852 |
| Historical objective (`--w_message 0`), pinned environment, A100 | 0–4 | 0.8187 | 0.3647 |
| Archived `GRACE+pre` source, recovered original environment, V100 | 0 only | 0.8224 | 0.3754 |

The unpinned run used Python 3.12.3, PyTorch 2.14.1+cu130, PyTorch Geometric
2.8.0.post1, Geoopt 0.5.1, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, and
scikit-learn 1.9.1. The original-environment baseline was stopped after
three complete seeds established the same large discrepancy. Its incomplete
fourth trial is excluded. A separate pinned A100 default-objective run was
stopped before completing a seed and is not reported as a completed result.

These baselines show that restoring package versions alone does not recover
the reference metrics. They are not an exact hardware-controlled estimate
of the effect of changing any single package.


## Five-seed reproduction by dataset

All values are mean ± population standard deviation. Full per-seed values
and the final log digest are in [reproduction-results.json](reproduction-results.json).

| Dataset | Reference AUROC | Reproduced AUROC | Reference AUPRC | Reproduced AUPRC |
| --- | --- | --- | --- | --- |
| ACM | 0.8117 ± 0.0005 | 0.8120 ± 0.0007 | 0.3717 ± 0.0007 | 0.3713 ± 0.0009 |
| Amazon | 0.7690 ± 0.0180 | 0.7543 ± 0.0227 | 0.3155 ± 0.0484 | 0.2757 ± 0.0583 |
| BlogCatalog | 0.7309 ± 0.0006 | 0.7301 ± 0.0009 | 0.3531 ± 0.0042 | 0.3552 ± 0.0057 |
| citeseer | 0.9028 ± 0.0013 | 0.9043 ± 0.0021 | 0.4015 ± 0.0131 | 0.4112 ± 0.0166 |
| cora | 0.8639 ± 0.0020 | 0.8643 ± 0.0017 | 0.4525 ± 0.0088 | 0.4541 ± 0.0094 |
| Facebook | 0.7575 ± 0.0095 | 0.7558 ± 0.0081 | 0.0900 ± 0.0112 | 0.0835 ± 0.0155 |
| weibo | 0.9103 ± 0.0011 | 0.9104 ± 0.0007 | 0.6026 ± 0.0030 | 0.6019 ± 0.0023 |

The average differs from the reference by −0.0022 AUROC and −0.0049 AUPRC.
Amazon accounts for the largest remaining discrepancy (−0.0147 AUROC,
−0.0398 AUPRC). This recovers near-reference performance, not an exact
reproduction of every published number. Cross-hardware numerical effects
and stochastic training remain possible contributors; these measurements
do not isolate their individual effects.

The archived implementation also completed one full seed on the original
V100 environment; [archived-validation.json](archived-validation.json) records
its metrics. That one-seed check is not a replacement for the five-seed
reference or evidence of an exact match.

## Running controlled comparisons

Use the pinned environment, an idle GPU, and separate working directories
for runs whose outputs you want to retain. The program writes relative to
the current working directory and reuses result filenames. Include the
trailing slash in `--data_dir`.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Historical training objective from the archived implementation
CUDA_VISIBLE_DEVICES=0 python -u main.py --trials 5 --w_message 0

# Released default objective (the three-seed audit averaged 0.7711 / 0.2852)
# Save the preceding results elsewhere before running this in the same directory.
CUDA_VISIBLE_DEVICES=0 python -u main.py --trials 5 --w_message 1
```

Thread limits avoid excessive CPU threading on shared machines; record the
chosen values alongside results. CUDA kernels and numerical-library builds
can still produce numerical differences. Trial seeds are 0 through 4; the
program seeds training after feature preprocessing, while PCA explicitly
uses `random_state=0`.
