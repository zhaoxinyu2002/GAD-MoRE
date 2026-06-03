# GAD-MoRE: Zero-shot Generalizable Graph Anomaly Detection with Mixture of Riemannian Experts

This repository contains the implementation of **GAD-MoRE**, a zero-shot generalizable graph anomaly detection framework based on a **Mixture of Riemannian Experts**.

## Abstract

Graph Anomaly Detection (GAD) aims to identify irregular patterns in graph data, and recent works have explored zero-shot generalist GAD to enable generalization to unseen graph datasets. However, existing zero-shot GAD methods largely ignore intrinsic geometric differences across diverse anomaly patterns, substantially limiting their cross-domain generalization. In this work, we reveal that anomaly detectability is highly dependent on the underlying geometric properties and that embedding graphs from different domains into a single static curvature space can distort the structural signatures of anomalies. To address the challenge that a single curvature space cannot capture geometry-dependent graph anomaly patterns, we propose **GAD-MoRE**, a novel framework for zero-shot Generalizable Graph Anomaly Detection with a Mixture of Riemannian Experts architecture. Specifically, to ensure that each anomaly pattern is modeled in the Riemannian space where it is most detectable, GAD-MoRE employs a set of specialized Riemannian expert networks, each operating in a distinct curvature space. To align raw node features with curvature-specific anomaly characteristics, we introduce an anomaly-aware multi-curvature feature alignment module that projects inputs into parallel Riemannian spaces, enabling the capture of diverse geometric characteristics. Finally, to facilitate better generalization beyond seen patterns, we design a memory-based dynamic router that adaptively assigns each input to the most compatible expert based on historical reconstruction performance on similar anomalies. Extensive experiments in the zero-shot setting demonstrate that GAD-MoRE significantly outperforms state-of-the-art generalist GAD baselines.

## Highlights

- **Zero-shot cross-domain GAD** without target-domain training, validation, or fine-tuning.
- **Mixture of Riemannian Experts** for modeling geometry-dependent anomaly patterns.
- **Anomaly-aware multi-curvature feature alignment** for constructing geometry-aware node representations.
- **Memory-based dynamic router** for reconstruction-quality-aware expert selection.
- **End-to-end training and evaluation code** for AUROC/AUPRC reporting over multiple trials.

## Repository Structure

```text
GAD-MoRE/
├── main.py              # main training and evaluation entry
├── model.py             # GAD-MoRE model, Riemannian experts, MoE scorer, memory router
├── train_test.py        # trainer and evaluator
├── utils.py             # dataset loading, preprocessing, metrics, and result saving
├── data/                # place graph datasets here
├── params/              # optional hyperparameter JSON files
├── checkpoints/         # saved model checkpoints
├── results/             # CSV result files
└── README.md
````

## Method Overview

GAD-MoRE contains three main components:

1. **Anomaly-aware Multi-curvature Feature Alignment**: projects raw node features into multiple curvature-aware spaces and constructs unified node representations.
2. **Mixture of Riemannian Experts Scorer**: uses multiple Riemannian expert networks with learnable curvatures to reconstruct node embeddings.
3. **Memory-based Dynamic Router**: routes nodes to suitable experts according to historical reconstruction quality.

## Environment

A typical environment is:

```bash
python >= 3.9
pytorch
torch-geometric
geoopt
numpy
scipy
pandas
scikit-learn
```

Install the main dependencies with:

```bash
pip install torch torch-geometric geoopt numpy scipy pandas scikit-learn tqdm
```

## Data Preparation

Place all `.mat` graph datasets under `./data/`.

Expected file structure:

```text
data/
├── pubmed.mat
├── Flickr.mat
├── Reddit.mat
├── YelpChi.mat
├── ACM.mat
├── Amazon.mat
├── BlogCatalog.mat
├── citeseer.mat
├── cora.mat
├── Facebook.mat
└── weibo.mat
```

Each `.mat` file should contain:

* `Network`: adjacency matrix
* `Attributes`: node attributes
* `Label` or `gnd`: anomaly labels

The first run will automatically preprocess raw `.mat` files and cache processed files such as:

```text
data/ACM_processed_dim32.pt
```

## Training and Evaluation

Run the default zero-shot experiment:

```bash
python main.py --data_dir ./data/ --trials 5
```

By default, the model is trained on four source graphs:

```text
pubmed, Flickr, Reddit, YelpChi
```

and evaluated on seven unseen target graphs:

```text
ACM, Amazon, BlogCatalog, citeseer, cora, Facebook, weibo
```

Target labels are used only for evaluation.

## Main Options

```bash
python main.py \
  --data_dir ./data/ \
  --trials 5 \
  --dims 32 \
  --gate_temperature 0.7 \
  --contrastive_weight 0.1 \
  --w_embed 1.0 \
  --w_feature 0.5 \
  --w_structure 0.1 \
  --w_gate 0.01
```

## Save and Load Checkpoints

Save trained models:

```bash
python main.py --data_dir ./data/ --trials 5 --save_model
```

Load a saved checkpoint for evaluation:

```bash
python main.py \
  --data_dir ./data/ \
  --load_model_path checkpoints/GAD-MoRE_seed0.pth
```

## Outputs

The code reports AUROC and AUPRC for each target dataset and saves CSV results under:

```text
results/
```

Example output files:

```text
results/32_multitrain_GAD-MoRE_AUC.csv
results/32_multitrain_GAD-MoRE_AP.csv
```

## Notes

* The paper refers to the method as **GAD-MoRE**. The implementation now uses GAD-MoRE naming consistently.
* The default setting follows zero-shot cross-domain evaluation.
* The first run may be slower because feature alignment and dataset caching are performed.
* If visualization utilities are used, please include `visualization.py`; otherwise, visualization-related code can be ignored.

## Citation

If you use this codebase in your research, please cite our paper:

```bibtex
@inproceedings{gadmore2026,
  title     = {Zero-shot Generalizable Graph Anomaly Detection with Mixture of Riemannian Experts},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining},
  year      = {2026}
}
```
