import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import numpy as np
from typing import Optional, Dict, List
from utils import max_message

class NodeMemoryEntry:
    def __init__(self, node_embedding: torch.Tensor, performance_score: float):
        self.node_embedding = node_embedding.clone().detach()
        self.performance_score = performance_score
        self.access_count = 1
        self.last_access_epoch = 0

    def update_access(self, epoch: int):
        self.access_count += 1
        self.last_access_epoch = epoch

    def compute_similarity(self, query_embedding: torch.Tensor) -> float:
        return F.cosine_similarity(
            self.node_embedding.unsqueeze(0),
            query_embedding.unsqueeze(0)
        ).item()

class ExpertMemoryBank:
    def __init__(self, memory_size: int, embedding_dim: int, coldstart_epochs: int = 5):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.coldstart_epochs = coldstart_epochs
        self.quality_warmup_epochs = 10
        self.current_epoch = 0

        self.memories: List[NodeMemoryEntry] = []
        self.behavior_vector = None

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def add_memory(self, node_embedding: torch.Tensor, performance_score: float):
        if self.current_epoch < self.coldstart_epochs:
            return

        if self.current_epoch < self.coldstart_epochs + self.quality_warmup_epochs:
            progress = (self.current_epoch - self.coldstart_epochs) / self.quality_warmup_epochs
            min_quality_threshold = 0.3 + 0.4 * max(0.0, min(1.0, progress))
        else:
            min_quality_threshold = 0.7

        if performance_score < min_quality_threshold:
            return

        memory = NodeMemoryEntry(node_embedding, performance_score)

        if len(self.memories) < self.memory_size:
            self.memories.append(memory)
        else:
            worst_idx = min(range(len(self.memories)),
                          key=lambda i: self.memories[i].performance_score)
            worst_score = self.memories[worst_idx].performance_score

            if performance_score - worst_score > 0.1:
                self.memories[worst_idx] = memory

        self._update_behavior_profile()

    def _update_behavior_profile(self):
        if not self.memories:
            self.behavior_vector = torch.zeros(self.embedding_dim)
            return

        weights = F.softmax(torch.tensor([m.performance_score for m in self.memories]), dim=0)
        embeddings = torch.stack([m.node_embedding for m in self.memories])

        if embeddings.device != weights.device:
            weights = weights.to(embeddings.device)

        self.behavior_vector = torch.sum(weights.unsqueeze(1) * embeddings, dim=0)

    def get_similarity(self, query_embedding: torch.Tensor) -> float:
        if not self.memories:
            return 0.5

        max_similarity = 0.0
        for memory in self.memories:
            similarity = memory.compute_similarity(query_embedding)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def get_behavior_vector(self) -> torch.Tensor:
        if self.behavior_vector is None:
            return torch.zeros(self.embedding_dim)
        return self.behavior_vector.clone().detach()

    def is_empty(self) -> bool:
        return not self.memories

    def get_stats(self) -> dict:
        if not self.memories:
            return {
                'memory_count': 0,
                'avg_score': 0.0,
                'coldstart_remaining': max(0, self.coldstart_epochs - self.current_epoch)
            }

        return {
            'memory_count': len(self.memories),
            'avg_score': np.mean([m.performance_score for m in self.memories]),
            'coldstart_remaining': max(0, self.coldstart_epochs - self.current_epoch)
        }

class ExpertMemoryRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, experts: nn.ModuleList, memory_size: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.memory_size = memory_size
        self.experts = experts

        self.warmup_epochs = 15
        self.current_epoch = 0
        self.exploration_ratio = 1.0

        self.expert_memories = [
            ExpertMemoryBank(memory_size, embedding_dim, coldstart_epochs=self.warmup_epochs)
            for _ in range(num_experts)
        ]

        self.output_proj = nn.Linear(num_experts, num_experts)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if epoch < 5:
            self.exploration_ratio = 1.0
        elif epoch < self.warmup_epochs:
            progress = (epoch - 5) / (self.warmup_epochs - 5)
            self.exploration_ratio = 0.8 * (1.0 - progress) + 0.2
        elif epoch < self.warmup_epochs + 10:
            progress = (epoch - self.warmup_epochs) / 10
            self.exploration_ratio = 0.2 * (1.0 - progress) + 0.05
        else:
            self.exploration_ratio = 0.05

        for memory_bank in self.expert_memories:
            memory_bank.set_epoch(epoch)

    def get_logits(self, node_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = node_embeds.size(0)

        memory_fullness = sum(len(mem.memories) for mem in self.expert_memories) / (self.num_experts * self.memory_size)
        adaptive_exploration = self.exploration_ratio + (1.0 - memory_fullness) * 0.3
        adaptive_exploration = min(adaptive_exploration, 0.9)

        if self.training and torch.rand(1).item() < adaptive_exploration:
            return self._get_exploration_logits(batch_size, node_embeds.device)
        else:
            return self._get_memory_based_logits(node_embeds)

    def _get_exploration_logits(self, batch_size: int, device: torch.device) -> torch.Tensor:
        logits = torch.ones(batch_size, self.num_experts, device=device)
        noise = torch.randn_like(logits) * 0.1
        return logits + noise

    def _get_memory_based_logits(self, node_embeds: torch.Tensor) -> torch.Tensor:
            batch_size = node_embeds.size(0)
            logits = torch.zeros(batch_size, self.num_experts, device=node_embeds.device)
            chunk_size = 512

            for expert_id in range(self.num_experts):
                if self.expert_memories[expert_id].is_empty():
                    logits[:, expert_id] = 0.0
                else:
                    manifold = self.experts[expert_id].manifold

                    with torch.no_grad():
                        memory_embeds_euclid = torch.stack(
                            [m.node_embedding for m in self.expert_memories[expert_id].memories]
                        ).to(node_embeds.device)
                        memory_embeds_proj = manifold.expmap0(memory_embeds_euclid)

                        all_min_dists = []

                        for i in range(0, batch_size, chunk_size):
                            end_i = min(i + chunk_size, batch_size)
                            node_embeds_chunk = node_embeds[i:end_i].detach()
                            node_embeds_chunk_proj = manifold.expmap0(node_embeds_chunk)

                            dists_chunk = manifold.dist(
                                node_embeds_chunk_proj.unsqueeze(1),
                                memory_embeds_proj.unsqueeze(0)
                            )

                            min_dists_chunk, _ = dists_chunk.min(dim=1)
                            all_min_dists.append(min_dists_chunk)

                        if all_min_dists:
                            min_dists = torch.cat(all_min_dists, dim=0)
                            logits[:, expert_id] = -min_dists

            return self.output_proj(logits)

    def update_memory(self, node_embeds: torch.Tensor, expert_errors: torch.Tensor, expert_assignments: torch.Tensor):
        if not self.training or node_embeds.size(0) < 10:
            return

        with torch.no_grad():
            for expert_id in range(self.num_experts):
                expert_mask = expert_assignments[:, expert_id] > 0.1

                if expert_mask.sum() == 0:
                    continue

                assigned_embeds = node_embeds[expert_mask]
                assigned_errors = expert_errors[expert_mask, expert_id]

                scores = 1.0 - (assigned_errors / (assigned_errors.max() + 1e-6))

                quality_threshold = 0.7
                high_quality_mask = scores >= quality_threshold

                if high_quality_mask.sum() == 0:
                    quality_threshold = torch.quantile(scores, 0.7)
                    high_quality_mask = scores >= quality_threshold

                if high_quality_mask.sum() > 0:
                    high_quality_embeds = assigned_embeds[high_quality_mask]
                    high_quality_scores = scores[high_quality_mask]

                    sorted_indices = torch.argsort(high_quality_scores, descending=True)

                    num_samples = min(3, len(high_quality_embeds), self.expert_memories[expert_id].memory_size // 4)

                    for i in range(num_samples):
                        idx = sorted_indices[i]
                        self.expert_memories[expert_id].add_memory(
                            high_quality_embeds[idx],
                            high_quality_scores[idx].item()
                        )

    def get_memory_stats(self) -> dict:
        stats = {
            'memory_utilization': [len(mem.memories) / self.memory_size for mem in self.expert_memories],
            'avg_scores': [mem.get_stats()['avg_score'] for mem in self.expert_memories],
            'coldstart_remaining': [mem.get_stats()['coldstart_remaining'] for mem in self.expert_memories],
            'exploration_ratio': self.exploration_ratio,
            'current_epoch': self.current_epoch
        }
        return stats

class ExpertMemoryRouterWrapper(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, experts: nn.ModuleList, memory_size: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.expert_router = ExpertMemoryRouter(embedding_dim, num_experts, experts, memory_size)

    def get_logits(self, node_embeds: torch.Tensor) -> torch.Tensor:
        return self.expert_router.get_logits(node_embeds)

    def update_memory(self, node_embeds: torch.Tensor, expert_errors: torch.Tensor,
                     expert_assignments: torch.Tensor = None):
        if expert_assignments is None:
            with torch.no_grad():
                weights = 1.0 / (expert_errors + 1e-6)
                expert_assignments = F.softmax(weights, dim=-1)

        self.expert_router.update_memory(node_embeds, expert_errors, expert_assignments)

    def set_epoch(self, epoch: int):
        self.expert_router.set_epoch(epoch)

    def get_memory_stats(self) -> dict:
        return self.expert_router.get_memory_stats()

class LinearRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, num_experts)

    def get_logits(self, node_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(node_embeds)

    def update_memory(self, *args, **kwargs):
        return None

class GADMoRE(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=32,
        num_layers=2,
        dropout_rate=0,
        activation="ReLU",
        num_hops=4,
        scorer_type="moe",
        original_feature_dim=None,
        **kwargs,
    ):
        super(GADMoRE, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        if num_layers > 0:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for _ in range(1, num_layers - 1):
                self.layers.append(nn.Linear(h_feats, h_feats))

        scorer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in [
                "num_experts",
                "expert_hidden_dim",
                "top_k",
                "init_curvs",
                "use_structure_aware_gate",
                "gate_temperature",
                "gate_noise_type",
                "gate_noise_std",
                "policy_entropy_coef",
                "policy_loss_coef",
                "memory_size",
                "memory_hidden_dim",
            ]
        }

        embedding_dim = h_feats * num_hops
        self.anomaly_scorer = AnomalyScorerMoE(
            embedding_dim=embedding_dim,
            original_feature_dim=original_feature_dim,
            **scorer_kwargs,
        )

    def forward(self, h, use_residual=True):
        x_list = h.x_list
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
            if i != len(self.layers) - 1:
                x_list = [self.act(x) for x in x_list]

        if not use_residual:
            return torch.hstack(x_list)

        residual_list = [x_list[i] - x_list[0] for i in range(1, len(x_list))]
        return torch.hstack(residual_list)

class AnomalyScorerAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(AnomalyScorerAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.node_proj = nn.Linear(embedding_dim, embedding_dim)
        self.ref_proj = nn.Linear(embedding_dim, embedding_dim)

    def reconstruction_attention(self, node_embeds, reference_embeds):
        node_proj = self.node_proj(node_embeds)
        ref_proj = self.ref_proj(reference_embeds)
        sim_scores = torch.matmul(node_proj, ref_proj.transpose(0, 1)) / torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32)
        )
        attention_weights = F.softmax(sim_scores, dim=1)
        return torch.matmul(attention_weights, reference_embeds)

    def get_test_score(self, X, prompt_mask, y):
        reference_embed = X
        target_embed = X
        target_recon = self.reconstruction_attention(target_embed, reference_embed)
        diff = target_embed - target_recon
        return torch.sqrt(torch.sum(diff**2, dim=1))

    def get_unsupervised_loss(self, *args, **kwargs):
        node_embeddings = args[2]
        original_adj = args[1]
        recon_embed = self.reconstruction_attention(node_embeddings, node_embeddings)
        base_loss = F.mse_loss(node_embeddings, recon_embed)
        w_message = kwargs.get("w_message", 0.0)
        if w_message and w_message > 0:
            mm_loss, _ = max_message(node_embeddings, original_adj)
            return base_loss + w_message * mm_loss
        return base_loss

class AnomalyScorerMoE(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_experts: int = 5,
        expert_hidden_dim: int = 128,
        top_k: int = 2,
        init_curvs=None,
        original_feature_dim: int | None = None,
        gate_temperature: float = 1.0,
        gate_noise_type: str = "none",
        gate_noise_std: float = 0.0,
        memory_size: int = 32,
        memory_hidden_dim: int = 64,
        router_type: str = "expert_memory",
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate_temperature = gate_temperature
        self.gate_noise_type = gate_noise_type
        self.gate_noise_std = gate_noise_std

        self.router_type = "expert_memory"

        if init_curvs is None:
            init_curvs = (
                [0.0]
                + [-0.5 * (i + 1) for i in range(num_experts // 2)]
                + [0.5 * (i + 1) for i in range(num_experts - num_experts // 2 - 1)]
            )
        self.experts = nn.ModuleList(
            [
                RiemannianExpert(
                    initial_curvature=curv,
                    in_dim=embedding_dim,
                    hidden_dim=expert_hidden_dim,
                    out_dim=embedding_dim,
                    dropout=0.1,
                )
                for curv in init_curvs
            ]
        )

        self.router = ExpertMemoryRouterWrapper(
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            experts=self.experts,
            memory_size=memory_size,
            hidden_dim=memory_hidden_dim,
        )

        if original_feature_dim is None:
            raise ValueError("必须提供 original_feature_dim 用于特征重构。")
        self.feature_decoder = nn.Linear(embedding_dim, original_feature_dim)

    def _apply_noise(self, logits):
        if not self.training:
            return logits
        if self.gate_noise_type == "gaussian" and self.gate_noise_std > 0:
            return logits + self.gate_noise_std * torch.randn_like(logits)
        if self.gate_noise_type == "gumbel":
            U = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(U))
            return logits + gumbel
        return logits

    def moe_reconstruction(self, node_embeds):

        gate_logits = self.router.get_logits(node_embeds)
        noisy_gate_logits = self._apply_noise(gate_logits)

        gate_weights = F.softmax(noisy_gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(noisy_gate_logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals / max(self.gate_temperature, 1e-6), dim=-1)

        with torch.no_grad():
            unique_experts = torch.unique(topk_idx)
        expert_outputs_unique = []
        for e_id in unique_experts.tolist():

            if node_embeds.size(0) > 512:
                chunk_size = 256
                out_e_chunks = []
                for i in range(0, node_embeds.size(0), chunk_size):
                    chunk = node_embeds[i:i+chunk_size]
                    with torch.cuda.amp.autocast(enabled=False):
                        out_chunk = self.experts[e_id](chunk)
                        out_chunk_euclid = self.experts[e_id].manifold.logmap0(out_chunk)
                    out_e_chunks.append(out_chunk_euclid)
                out_e_euclid = torch.cat(out_e_chunks, dim=0)
            else:
                out_e = self.experts[e_id](node_embeds)
                out_e_euclid = self.experts[e_id].manifold.logmap0(out_e)
            expert_outputs_unique.append(out_e_euclid)

        if len(expert_outputs_unique) == 0:
            reconstructed_embeds = torch.zeros_like(node_embeds)
        else:
            stacked_unique = torch.stack(expert_outputs_unique, dim=0)
            mapping_tensor = torch.full(
                (self.num_experts,), -1, dtype=torch.long, device=node_embeds.device
            )
            mapping_tensor[unique_experts] = torch.arange(
                unique_experts.size(0), device=node_embeds.device
            )
            mapped_pos = mapping_tensor[topk_idx]
            stacked_unique_NUD = stacked_unique.permute(1, 0, 2)
            gathered = torch.gather(
                stacked_unique_NUD,
                1,
                mapped_pos.unsqueeze(-1).expand(-1, -1, self.embedding_dim),
            )
            weighted = gathered * topk_weights.unsqueeze(-1)
            reconstructed_embeds = weighted.sum(dim=1)

        reconstructed_features = self.feature_decoder(reconstructed_embeds)
        reconstructed_adj_logits = reconstructed_embeds @ reconstructed_embeds.t()
        reconstructed_adj = torch.sigmoid(reconstructed_adj_logits)

        gate_entropy = - (gate_weights * torch.log(gate_weights + 1e-9)).sum(dim=-1)

        if self.training and hasattr(self.router, 'update_memory'):

            batch_size = min(64, len(node_embeds))
            expert_errors = torch.zeros(len(node_embeds), self.num_experts, device=node_embeds.device)

            for start_idx in range(0, len(node_embeds), batch_size):
                end_idx = min(start_idx + batch_size, len(node_embeds))
                batch_embeds = node_embeds[start_idx:end_idx]

                with torch.no_grad():
                    for expert_id in range(self.num_experts):
                        expert_output = self.experts[expert_id](batch_embeds)
                        expert_output_euclid = self.experts[expert_id].manifold.logmap0(expert_output)
                        expert_error = F.mse_loss(expert_output_euclid, batch_embeds, reduction='none').mean(dim=1)
                        expert_errors[start_idx:end_idx, expert_id] = expert_error

            expert_assignments = torch.zeros(len(node_embeds), self.num_experts, device=node_embeds.device)
            expert_assignments.scatter_(1, topk_idx, topk_weights)

            self.router.update_memory(node_embeds, expert_errors, expert_assignments)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        policy_info = {
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "gate_logits": gate_logits,
            "noisy_gate_logits": noisy_gate_logits,
            "entropy": gate_entropy,
        }

        return (
            reconstructed_embeds,
            gate_weights,
            reconstructed_features,
            reconstructed_adj,
            policy_info,
        )

    def get_test_score(
        self,
        X,
        adj,
        prompt_mask,
        y,
        original_features: torch.Tensor | None = None,
        embed_weight: float = 1.0,
        feature_weight: float = 0.5,
    ):

        recon_embed, _, recon_features, _, _ = self.moe_reconstruction(X)

        embed_err = torch.sqrt(torch.sum((X - recon_embed) ** 2, dim=1))

        feat_err = F.mse_loss(recon_features, original_features, reduction="none").mean(
            dim=1
        )

        embed_s, feat_s = embed_err, feat_err

        score = embed_weight * embed_s
        return score

    def get_unsupervised_loss(
        self,
        original_features,
        original_adj,
        node_embeddings,
        contrastive_weight=0.1,
        temperature=0.1,
        w_embed: float = 1.0,
        w_feature: float = 0.5,
        w_structure: float = 0.1,
        w_entropy: float = 0.01,
        w_message: float = 0.0,
    ):
        recon_embed, gate_weights, recon_features, recon_adj, policy_info = self.moe_reconstruction(
            node_embeddings
        )

        recon_embed_loss = F.mse_loss(recon_embed, node_embeddings)
        feature_recon_loss = F.mse_loss(recon_features, original_features)
        structure_recon_loss = F.binary_cross_entropy(
            recon_adj, original_adj.to_dense()
        )

        contrastive_loss = (
            structure_contrastive_loss(node_embeddings, original_adj, temperature)
            if contrastive_weight > 0
            else 0.0
        )

        base_loss = (
            w_embed * recon_embed_loss
            + w_feature * feature_recon_loss
            + w_structure * structure_recon_loss
            + contrastive_weight * contrastive_loss
        )

        if self.training and hasattr(self, '_update_step_counter'):
            self._update_step_counter += 1
        elif self.training:
            self._update_step_counter = 0

        if self.training and getattr(self, '_update_step_counter', 0) % 20 == 0:

            sample_size = min(100, node_embeddings.size(0))
            indices = torch.randperm(node_embeddings.size(0))[:sample_size]
            sample_embeds = node_embeddings[indices]

            with torch.no_grad():
                expert_errors = []
                for i, expert in enumerate(self.experts):
                    expert_output = expert(sample_embeds)
                    expert_output_euclid = expert.manifold.logmap0(expert_output)
                    expert_error = F.mse_loss(expert_output_euclid, sample_embeds, reduction='none').mean(dim=1)
                    expert_errors.append(expert_error)
                expert_errors = torch.stack(expert_errors, dim=1)

            self.router.update_memory(sample_embeds, expert_errors)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        entropy_mean = policy_info['entropy'].mean()

        if w_message and w_message > 0:
            mm_loss, _ = max_message(node_embeddings, original_adj)
            base_loss = base_loss + w_message * mm_loss

        total_loss = base_loss + w_entropy * entropy_mean

        with torch.no_grad():
            topk_idx = policy_info['topk_idx']
            usage_counts = torch.bincount(topk_idx.view(-1), minlength=self.num_experts).float()
            usage_dist = usage_counts / usage_counts.sum().clamp_min(1.0)
            load_balance = usage_dist.std() / (usage_dist.mean().clamp_min(1e-9))
            avg_topk_weight = policy_info['topk_weights'].mean().item()
            self.latest_routing_stats = {
                'entropy_mean': float(entropy_mean.item()),
                'expert_usage_dist': usage_dist.cpu().tolist(),
                'load_balance_cv': float(load_balance.item()),
                'avg_topk_weight': avg_topk_weight,
            }

        return total_loss

    def get_latest_routing_stats(self) -> Optional[Dict]:
        return getattr(self, 'latest_routing_stats', None)

class kappaLinear(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, dropout=0.0, use_bias=True):
        super(kappaLinear, self).__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x)
        if self.use_bias:
            bias = self.manifold.proju(self.manifold.origin(self.bias.shape), self.bias)
            kappa_bias = self.manifold.expmap0(bias)
            res = self.manifold.mobius_add(res, kappa_bias)
        return res

class RiemannianExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0, initial_curvature=0.0):
        super(RiemannianExpert, self).__init__()
        self.manifold = geoopt.Stereographic(k=initial_curvature, learnable=True)
        self.linear1 = kappaLinear(
            self.manifold, in_dim, hidden_dim, dropout, use_bias=True
        )
        self.linear2 = kappaLinear(
            self.manifold, hidden_dim, out_dim, dropout, use_bias=True
        )
        self.act = nn.ReLU()

    def forward(self, x):

        x_proj = self.manifold.proju(self.manifold.origin(x.shape), x)
        x_exp = self.manifold.expmap0(x_proj)

        h = self.linear1(x_exp)
        h_log = self.manifold.logmap0(h)
        h_act = self.act(h_log)
        h_act_exp = self.manifold.expmap0(h_act)

        output = self.linear2(h_act_exp)
        return output

def structure_contrastive_loss(features, adj, temperature=0.5, chunk_size=1000, sample_ratio=0.3):
    N = features.size(0)
    device = features.device

    adj = adj.coalesce()

    if N > 8000:
        sample_size = int(N * sample_ratio)
        sample_size = max(sample_size, 1000)
        sample_idx = torch.randperm(N, device=device)[:sample_size]

        sampled_features = features[sample_idx]

        adj_indices = adj.indices()
        adj_values = adj.values()

        old_to_new = torch.full((N,), -1, dtype=torch.long, device=device)
        old_to_new[sample_idx] = torch.arange(sample_size, device=device)

        src_mask = old_to_new[adj_indices[0]] >= 0
        dst_mask = old_to_new[adj_indices[1]] >= 0
        edge_mask = src_mask & dst_mask

        if edge_mask.sum() > 0:
            new_src = old_to_new[adj_indices[0][edge_mask]]
            new_dst = old_to_new[adj_indices[1][edge_mask]]
            new_edges = torch.stack([new_src, new_dst])
            new_values = adj_values[edge_mask]

            sampled_adj = torch.sparse_coo_tensor(
                new_edges, new_values,
                (sample_size, sample_size), device=device
            ).coalesce()
        else:

            sampled_adj = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                (sample_size, sample_size), device=device
            )

        features = sampled_features
        adj = sampled_adj
        N = sample_size

    features = F.normalize(features, p=2, dim=1)

    adj_indices = adj.indices()
    pos_pairs = torch.zeros((N, N), dtype=torch.bool, device=device)

    if adj_indices.size(1) > 0:
        pos_pairs[adj_indices[0], adj_indices[1]] = True
        pos_pairs[adj_indices[1], adj_indices[0]] = True

    pos_pairs.fill_diagonal_(True)

    total_loss = 0.0
    num_chunks = 0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_features = features[start:end]
        chunk_pos_pairs = pos_pairs[start:end]

        sim_matrix = torch.mm(chunk_features, features.T) / temperature

        exp_sim = torch.exp(sim_matrix)

        denominator = exp_sim.sum(dim=1, keepdim=True)

        numerator = (exp_sim * chunk_pos_pairs.float()).sum(dim=1, keepdim=True)

        numerator = torch.clamp(numerator, min=1e-8)
        denominator = torch.clamp(denominator, min=1e-8)

        log_prob = torch.log(numerator / denominator)
        chunk_loss = -log_prob.mean()

        total_loss += chunk_loss
        num_chunks += 1

    return total_loss / num_chunks
