import torch
from model import GADMoRE
from utils import test_eval

class GADMoREDetector:
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.device = self.train_config["device"]

        original_feature_dim = data["train"][0].original_feat.shape[1]
        model_config["original_feature_dim"] = original_feature_dim

        self.model = GADMoRE(**model_config).to(self.device)

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config["lr"],
            weight_decay=self.train_config["weight_decay"],
        )

        for e in range(self.train_config["epochs"]):
            effective_cw = self.train_config["contrastive_weight"]

            if hasattr(self.model.anomaly_scorer, 'router') and hasattr(self.model.anomaly_scorer.router, 'set_epoch'):
                self.model.anomaly_scorer.router.set_epoch(e)

            for didx, train_data in enumerate(self.data["train"]):
                self.model.train()

                train_graph = train_data.graph.to(self.device)
                original_features = train_data.original_feat.to(self.device)
                original_adj = train_data.original_adj.to(self.device)

                node_embeddings = self.model(train_graph, use_residual=True)

                loss = self.model.anomaly_scorer.get_unsupervised_loss(
                    original_features,
                    original_adj,
                    node_embeddings,
                    contrastive_weight=effective_cw,
                    temperature=self.train_config.get("temperature", 0.1),
                    w_embed=self.train_config.get("w_embed", 1.0),
                    w_feature=self.train_config.get("w_feature", 0.5),
                    w_structure=self.train_config.get("w_structure", 0.1),
                    w_entropy=self.train_config.get("w_gate", 0.01),
                    w_message=self.train_config.get("w_message", 0.0),
                )
                print(f"Epoch {e}, Dataset {didx}, Loss={loss.item():.4f}")

                if self.model_config.get("scorer_type") == "moe":
                    stats = getattr(self.model.anomaly_scorer, 'get_latest_routing_stats', lambda: None)()
                    if stats:
                        usage_str = ",".join([f"{u:.2f}" for u in stats['expert_usage_dist']])
                        print(
                            f"  RoutingStats -> entropy:{stats['entropy_mean']:.4f} "
                            f"load_cv:{stats['load_balance_cv']:.4f} avg_topk_w:{stats['avg_topk_weight']:.4f} usage:[{usage_str}]"
                        )

                    if hasattr(self.model.anomaly_scorer, 'router') and hasattr(self.model.anomaly_scorer.router, 'get_memory_stats'):
                        memory_stats = self.model.anomaly_scorer.router.get_memory_stats()
                        if e % 5 == 0:
                            util_str = ",".join([f"{u:.2f}" for u in memory_stats['memory_utilization']])
                            score_str = ",".join([f"{s:.3f}" for s in memory_stats['avg_scores']])
                            coldstart_remaining = memory_stats['coldstart_remaining'][0] if memory_stats['coldstart_remaining'] else 0
                            print(
                                f"  MemoryStats -> exploration_ratio:{memory_stats['exploration_ratio']:.3f} "
                                f"coldstart_remaining:{coldstart_remaining} "
                                f"utilization:[{util_str}] avg_scores:[{score_str}]"
                            )
                            pass

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"Finished {self.train_config['epochs']} training epochs.")
        return self.evaluate()

    def evaluate(self):
        test_score_list = {}
        self.model.eval()
        with torch.no_grad():
            for didx, test_data in enumerate(self.data["test"]):
                test_graph = test_data.graph.to(self.device)
                adj_matrix = test_graph.adj
                labels = test_graph.ano_labels.to(self.device)
                original_features = self.data["test"][didx].original_feat.to(self.device)

                node_embeddings = self.model(test_graph, use_residual=True)

                zero_shot_mask = torch.zeros_like(labels, dtype=torch.bool)

                if self.model_config["scorer_type"] == "moe":
                    query_scores = self.model.anomaly_scorer.get_test_score(
                        node_embeddings,
                        adj=adj_matrix,
                        prompt_mask=zero_shot_mask,
                        y=labels,
                        original_features=original_features,
                    )
                else:
                    query_scores = self.model.anomaly_scorer.get_test_score(
                        node_embeddings,
                        prompt_mask=zero_shot_mask,
                        y=labels,
                    )

                test_score = test_eval(labels, query_scores)
                test_data_name = self.train_config["testdsets"][didx]
                test_score_list[test_data_name] = {
                    "AUROC": test_score["AUROC"],
                    "AUPRC": test_score["AUPRC"],
                }
        return test_score_list