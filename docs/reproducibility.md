# Environment and reproduction audit

Audit date: 2026-10-04. Released source revision: `658b023`.

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

## Configuration drift and an open comparison

The release defaults to `--w_message 1.0`. This adds a sum of neighborhood
similarities to the training objective via `max_message`. It is separate
from the embedding, feature, structure, and router-entropy loss weights.

An archived `GRACE+pre` implementation retaining the same reference CSV
values does not contain this additional training loss. Its test-time score
is the same embedding-reconstruction distance used by the release. Setting
`--w_message 0` disables the added term in the released code without editing
the model. The reference CSV parameter blocks do not record `w_message`.
These observations motivate a controlled comparison; the CSVs alone do not
prove the exact historical execution provenance. A five-trial run with
`--w_message 0` was launched in the isolated pinned A100 environment, but
its outcome has not been retrieved and verified. Do not treat
`--w_message 0` as a verified way to recover the reference row.

## Completed baseline measurements

Values below are unweighted averages over the seven target datasets. The
historical row summarizes the released CSVs; the other rows summarize
completed runs, not inferred scores.

| Setting | Completed seeds | Mean AUROC | Mean AUPRC |
| --- | --- | --- | --- |
| Historical reference CSVs | 0–4 (five-trial reference) | 0.8209 | 0.3696 |
| Released default, unpinned new environment, A100 | 0–4 | 0.7702 | 0.2844 |
| Released default, recovered original environment, V100 | 0–2 | 0.7711 | 0.2852 |

The unpinned run used Python 3.12.3, PyTorch 2.14.1+cu130, PyTorch Geometric
2.8.0.post1, Geoopt 0.5.1, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, and
scikit-learn 1.9.1. The original-environment baseline was stopped after
three complete seeds established the same large discrepancy. Its incomplete
fourth trial is excluded. A separate pinned A100 default-objective run was
stopped before completing a seed and is not reported as a completed result.

These baselines show that restoring package versions alone does not recover
the reference metrics. They are not an exact hardware-controlled estimate
of the effect of changing any single package.

## Running controlled comparisons

Use the pinned environment, an idle GPU, and separate working directories
for runs whose outputs you want to retain. The program writes relative to
the current working directory and reuses result filenames. Include the
trailing slash in `--data_dir`.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Released default training objective
CUDA_VISIBLE_DEVICES=0 python -u main.py --trials 5 --w_message 1

# Historical objective without the added message loss
# Save the preceding results elsewhere before running this in the same directory.
CUDA_VISIBLE_DEVICES=0 python -u main.py --trials 5 --w_message 0
```

Thread limits avoid excessive CPU threading on shared machines; record the
chosen values alongside results. CUDA kernels and numerical-library builds
can still produce numerical differences. Trial seeds are 0 through 4; the
program seeds training after feature preprocessing, while PCA explicitly
uses `random_state=0`.

