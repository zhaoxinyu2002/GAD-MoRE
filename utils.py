import os
import json
import random
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import roc_auc_score, average_precision_score
import geoopt

def test_eval(labels, probs):
    score = {}
    with torch.no_grad():
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        probs = probs.cpu().numpy() if torch.is_tensor(probs) else probs
        score["AUROC"] = roc_auc_score(labels, probs)
        score["AUPRC"] = average_precision_score(labels, probs)
    return score

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    if sp.issparse(features):
        return features.toarray()
    return features

def feat_alignment(X, edges, dims):
    X_np = X.cpu().numpy() if torch.is_tensor(X) else X
    X_tensor = torch.FloatTensor(X_np)

    num_nodes = X_np.shape[0]
    edge_src, edge_dst = edges[0].cpu().numpy(), edges[1].cpu().numpy()
    adj = sp.coo_matrix((np.ones(len(edge_src)), (edge_src, edge_dst)),
                        shape=(num_nodes, num_nodes))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    degrees = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    lap_mx = sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    lap_mx = lap_mx.tocsr()

    def _laplacian_scores(feature_matrix: torch.Tensor, laplacian: sp.spmatrix):
        feature_np = feature_matrix.detach().cpu().numpy()
        scores = []
        for i in range(feature_np.shape[1]):
            feature_col = feature_np[:, i]
            feature_col_normalized = (feature_col - feature_col.mean()) / (feature_col.std() + 1e-9)
            scores.append(float(feature_col_normalized @ (laplacian.dot(feature_col_normalized))))
        return np.asarray(scores)

    total_ratio = 8
    euclidean_dims = (dims * 4) // total_ratio
    hyperbolic_dims = dims // total_ratio

    remaining_dims = dims - euclidean_dims - 4 * hyperbolic_dims
    euclidean_dims += remaining_dims

    euclidean_features = X_tensor

    if euclidean_features.shape[1] > euclidean_dims * 4:
        pca_euclidean = PCA(n_components=euclidean_dims * 4, random_state=0)
        euclidean_pca = torch.FloatTensor(pca_euclidean.fit_transform(euclidean_features.detach().numpy()))
    else:
        euclidean_pca = euclidean_features

    euclidean_scores = _laplacian_scores(euclidean_pca, lap_mx)
    euclidean_indices = torch.from_numpy(np.argsort(euclidean_scores)[:euclidean_dims]).long()
    euclidean_final = euclidean_pca[:, euclidean_indices]

    curvatures = [-1.0, 1.0, -0.5, 0.5]
    hyperbolic_features_list = []

    for curv in curvatures:

        manifold = geoopt.Stereographic(k=curv)
        x_proj = manifold.proju(manifold.origin(X_tensor.shape), X_tensor)
        x_exp = manifold.expmap0(x_proj)
        x_log = manifold.logmap0(x_exp)

        if x_log.shape[1] > hyperbolic_dims * 4:
            pca_hyperbolic = PCA(n_components=hyperbolic_dims * 4, random_state=0)
            hyperbolic_pca = torch.FloatTensor(pca_hyperbolic.fit_transform(x_log.detach().numpy()))
        else:
            hyperbolic_pca = x_log

        hyperbolic_scores = _laplacian_scores(hyperbolic_pca, lap_mx)
        hyperbolic_indices = torch.from_numpy(np.argsort(hyperbolic_scores)[:hyperbolic_dims]).long()
        hyperbolic_final = hyperbolic_pca[:, hyperbolic_indices]
        hyperbolic_features_list.append(hyperbolic_final)

    all_features = [euclidean_final] + hyperbolic_features_list
    final_features = torch.cat(all_features, dim=1)

    print(
        f"Feature split - Euclidean: {euclidean_dims}, hyperbolic: {hyperbolic_dims}x4, total: {final_features.shape[1]}"
    )

    return final_features

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def max_message(feature, adj_matrix):
    feature = feature / (torch.norm(feature, dim=-1, keepdim=True) + 1e-9)

    if torch.is_tensor(adj_matrix) and getattr(adj_matrix, "is_sparse", False):
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense().flatten()
        aggregated = torch.sparse.mm(adj_matrix, feature)
    else:
        if not torch.is_tensor(adj_matrix):
            adj_matrix = torch.tensor(adj_matrix, dtype=feature.dtype, device=feature.device)
        row_sum = torch.sum(adj_matrix, dim=1).flatten()
        aggregated = torch.matmul(adj_matrix, feature)

    message = torch.sum(feature * aggregated, dim=1)

    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    message = message * r_inv

    return -torch.sum(message), message

class Dataset:
    def __init__(self, dims, name="cora", prefix="./data/"):
        self.name = name
        self.dims = dims
        self.prefix = prefix

        self._load_and_process_data()

    def _load_and_process_data(self):
        mat_file = f"{self.prefix}{self.name}.mat"
        data = sio.loadmat(mat_file)

        adj = data["Network"]
        feat = data["Attributes"]
        labels = data["Label"] if "Label" in data else data["gnd"]

        if self.name in ["Amazon", "YelpChi", "tolokers", "tfinance"]:
            if sp.issparse(feat):
                feat = preprocess_features(feat)
        else:
            if sp.issparse(feat):
                feat = feat.toarray()

        feat = torch.FloatTensor(feat)

        adj_sp = sp.csr_matrix(adj)
        self.edge_index = torch.tensor(adj_sp.nonzero(), dtype=torch.long)
        self.feat = feat_alignment(feat, self.edge_index, self.dims)

        if self.name in ["YelpChi", "Facebook"]:
            adj_norm = normalize_adj(adj)
        else:
            adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
        self.adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

        self.graph = Data(
            x=self.feat,
            adj=self.adj_norm,
            ano_labels=torch.tensor(np.squeeze(np.array(labels)), dtype=torch.float),
        )
        self.original_feat = self.feat.clone()
        self.original_adj = self.adj_norm.clone()

        try:
            _, message = self.compute_max_message()

        except Exception as e:
            print(f"Failed to compute max_message: {e}")

    def propagated(self, k, device: str | torch.device | None = None):
        x = self.feat if device is None else self.feat.to(device)
        adj = self.adj_norm if device is None else self.adj_norm.to(device)
        h_list = [x]
        for _ in range(k):
            h_list.append(torch.spmm(adj, h_list[-1]))
        self.graph.x_list = h_list

    def compute_max_message(self):
        loss, message = max_message(self.feat, self.adj_norm)

        self.graph.max_message = message
        return loss, message

def read_json(model, shot, json_dir):
    filename = f"{json_dir}/{model}_{shot}.json"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON file {filename}: {e}")
    return None

def save_results_to_csv(auc_mean_dict, auc_std_dict, pre_mean_dict, pre_std_dict, datasets_test, args, dims):
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    method_name = args.model

    def create_df(mean_dict, std_dict, avg_metric_name):
        csv_data = {"Method": method_name}
        for dataset in datasets_test:
            mean_val = mean_dict.get(dataset, 0)
            std_val = std_dict.get(dataset, 0)
            csv_data[dataset] = f"{mean_val:.4f}±{std_val:.4f}"

        valid_means = [mean for mean in mean_dict.values() if mean > 0]
        avg_val = np.mean(valid_means) if valid_means else 0
        csv_data[avg_metric_name] = f"{avg_val:.4f}"
        return pd.DataFrame([csv_data])

    df_auc = create_df(auc_mean_dict, auc_std_dict, "AVG_AUC")
    df_ap = create_df(pre_mean_dict, pre_std_dict, "AVG_AP")

    param_text = "\nParameters:\n" + "\n".join([f"{arg}: {val}" for arg, val in vars(args).items()])

    def write_to_file(df, path):
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=["Method"], keep="last")
        df.to_csv(path, index=False)
        with open(path, "a") as f:
            f.write(param_text)

    auc_csv_path = os.path.join(results_dir, f"{dims}_multitrain_{method_name}_AUC.csv")
    ap_csv_path = os.path.join(results_dir, f"{dims}_multitrain_{method_name}_AP.csv")

    write_to_file(df_auc, auc_csv_path)
    write_to_file(df_ap, ap_csv_path)

    print(f"AUC results saved to {auc_csv_path}")
    print(df_auc)
    print(f"AP results saved to {ap_csv_path}")
    print(df_ap)