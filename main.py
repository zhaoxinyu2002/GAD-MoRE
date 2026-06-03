import argparse
import warnings
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import random
import numpy as np
import pandas as pd
import torch

from utils import Dataset, read_json, save_results_to_csv
from train_test import GADMoREDetector

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(args, dims):
    model_config = read_json(args.model, args.shot, args.json_dir)
    if model_config is None:
        print("使用默认模型配置。")
        model_config = {
            "in_feats": dims,
            "h_feats": 1024,
            "num_layers": 4,
            "dropout_rate": 0,
            "activation": "ELU",
            "num_hops": 2,
            "scorer_type": "moe",
            "num_experts": 5,
            "expert_hidden_dim": 256,
            "top_k": 2,
            "init_curvs": None,
            "eps": args.eps,
            "router_type": "expert_memory",
        }

        model_config["gate_temperature"] = args.gate_temperature
        model_config["gate_noise_type"] = args.gate_noise_type
        model_config["gate_noise_std"] = args.gate_noise_std
        model_config["memory_size"] = args.memory_size
    else:
        print("使用已保存的最佳模型配置。")
        model_config["scorer_type"] = "moe"
        model_config["in_feats"] = dims
        if "init_curvs" not in model_config:
            model_config["init_curvs"] = None

        model_config["gate_temperature"] = args.gate_temperature
        model_config["gate_noise_type"] = args.gate_noise_type
        model_config["gate_noise_std"] = args.gate_noise_std
        model_config["router_type"] = "expert_memory"
        model_config["memory_size"] = args.memory_size
    print(model_config)
    return model_config

def main():
    parser = argparse.ArgumentParser(description="GAD-MoRE - Anomaly Detection")
    parser.add_argument("--trials", type=int, default=5, help="实验重复次数")
    parser.add_argument("--model", type=str, default="GAD-MoRE", help="模型名称")
    parser.add_argument(
        "--shot", type=int, default=10, help="Few-shot setting中的shot数量"
    )
    parser.add_argument(
        "--json_dir", type=str, default="./params", help="存放超参数配置的JSON文件目录"
    )
    parser.add_argument(
        "--dims", type=int, default=32, help="输入特征维度（各数据集向量维度）"
    )
    parser.add_argument("--eps", type=float, default=4e-3, help="Epsilon for infmatrix")
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        default=True,
        help="使用无监督训练进行零样本检测",
    )

    parser.add_argument(
        "--gate_temperature",
        type=float,
        default=0.7,
        help="门控softmax温度（仅在top-k内）",
    )
    parser.add_argument(
        "--gate_noise_type",
        type=str,
        default="gumbel",
        choices=["none", "gaussian", "gumbel"],
        help="门控噪声类型（训练期）",
    )
    parser.add_argument(
        "--gate_noise_std",
        type=float,
        default=0.0,
        help="门控高斯噪声标准差（仅当 noise_type=gaussian 时生效）",
    )

    parser.add_argument(
        "--contrastive_weight", type=float, default=0.1, help="对比损失权重"
    )

    parser.add_argument("--w_embed", type=float, default=1.0, help="嵌入重构损失权重")
    parser.add_argument("--w_feature", type=float, default=0.5, help="特征重构损失权重")
    parser.add_argument(
        "--w_structure", type=float, default=0.1, help="结构重构(BCE)损失权重"
    )
    parser.add_argument("--w_gate", type=float, default=0.01, help="记忆路由器熵正则损失权重")
    parser.add_argument("--w_message", type=float, default=1.0, help="max_message 损失权重 (默认关闭)")
    parser.add_argument("--memory_size", type=int, default=32, help="专家记忆库容量（每个专家的最大记忆条目数）")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/",
        help="数据集 .mat 文件所在目录（以 / 结尾）",
    )
    args = parser.parse_args()
    datasets_test = [
        "ACM",
        "Amazon",
        "BlogCatalog",
        "citeseer",
        "cora",
        "Facebook",
        "weibo",
    ]
    datasets_train = ["pubmed", "Flickr", "Reddit", "YelpChi"]
    dims = args.dims

    print(f"在 {len(datasets_train)} 个数据集上进行训练: {datasets_train}")
    print(f"在 {len(datasets_test)} 个数据集上进行测试: {datasets_test}")
    print("使用主线配置: scorer=moe, router=expert_memory")

    train_config = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "epochs": 40,
        "testdsets": datasets_test,
        "lr": 5e-5,
        "weight_decay": 5e-5,
        "contrastive_weight": args.contrastive_weight,
        "temperature": 0.1,
        "w_embed": args.w_embed,
        "w_feature": args.w_feature,
        "w_structure": args.w_structure,
        "w_gate": args.w_gate,
        "w_message": args.w_message,
    }

    data_train = [Dataset(dims, name, prefix=args.data_dir) for name in datasets_train]
    data_test = [Dataset(dims, name, prefix=args.data_dir) for name in datasets_test]

    model_config = load_config(args, dims)

    for tr_data in data_train:
        tr_data.propagated(model_config["num_hops"], device=train_config["device"])
    for te_data in data_test:
        te_data.propagated(model_config["num_hops"], device=train_config["device"])

    auc_dict, pre_dict = {}, {}
    for t in range(args.trials):
        seed = t
        set_seed(seed)
        print(f"模型 {args.model}, 第 {seed} 次实验")
        train_config["seed"] = seed

        data = {"train": data_train, "test": data_test}
        detector = GADMoREDetector(train_config, model_config, data)
        test_score_list = detector.train()

        for test_data_name, test_score in test_score_list.items():
            if test_data_name not in auc_dict:
                auc_dict[test_data_name] = []
                pre_dict[test_data_name] = []
            auc_dict[test_data_name].append(test_score["AUROC"])
            pre_dict[test_data_name].append(test_score["AUPRC"])
            print(f"测试集 {test_data_name}, AUROC: {auc_dict[test_data_name]}")
            print(f"测试集 {test_data_name}, AUPRC: {pre_dict[test_data_name]}")

    auc_mean_dict, auc_std_dict, pre_mean_dict, pre_std_dict = {}, {}, {}, {}
    for test_data_name in auc_dict:
        auc_mean_dict[test_data_name] = np.mean(auc_dict[test_data_name])
        auc_std_dict[test_data_name] = np.std(auc_dict[test_data_name])
        pre_mean_dict[test_data_name] = np.mean(pre_dict[test_data_name])
        pre_std_dict[test_data_name] = np.std(pre_dict[test_data_name])

    for test_data_name in auc_mean_dict:
        str_result = (
            f"AUROC:{auc_mean_dict[test_data_name]:.4f}±{auc_std_dict[test_data_name]:.4f}, "
            f"AUPRC:{pre_mean_dict[test_data_name]:.4f}±{pre_std_dict[test_data_name]:.4f}"
        )
        print("-" * 50 + test_data_name + "-" * 50)
        print(f"结果: {str_result}")

    save_results_to_csv(
        auc_mean_dict,
        auc_std_dict,
        pre_mean_dict,
        pre_std_dict,
        datasets_test,
        args,
        dims,
    )

if __name__ == "__main__":
    main()
