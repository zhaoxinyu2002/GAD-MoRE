# Datasets

This directory contains the benchmark `.mat` files used by GAD-MoRE. After cloning the repository, no extra download step is required.

Expected filenames:

```text
pubmed.mat
Flickr.mat
Reddit.mat
YelpChi.mat
ACM.mat
Amazon.mat
BlogCatalog.mat
citeseer.mat
cora.mat
Facebook.mat
weibo.mat
```

Each file provides the graph structure, node attributes, and anomaly labels used by the loader in `utils.py` (typically through fields such as `Network`, `Attributes`, and `Label` or `gnd`).

Please still cite the original dataset papers when using these graphs. Preprocessed feature caches (`*_processed_dim*.pt`) may be created locally and are not part of the release.
