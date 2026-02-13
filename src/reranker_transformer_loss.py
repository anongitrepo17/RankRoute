"""
Transformer-based Ranking Evaluator Loss

Replaces the MLP-based evaluator with a Transformer architecture.
Uses self-attention to score candidate rankings based on inference-available signals.

Key components:
1. TransformerEvaluatorNonLeaky: Transformer-based evaluator
2. RankingEvaluatorTransformerLoss: Loss function using transformer evaluator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import unified NDCG calculation
from ndcg_utils import compute_ndcg
from ndcg_torch import compute_ndcg_batch


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# Transformer Evaluator
# ============================================================

class TransformerEvaluatorNonLeaky(nn.Module):
    """
    Transformer-based evaluator for ranking quality.
    
    Scores a candidate ranking using ONLY inference-available signals:
      - base_scores: model scores for each doc
      - optional doc_features: features per doc
      - position info: derived from rank and positional encoding
    
    No relevance labels are used.
    """
    
    def __init__(self,
                 doc_feat_dim: int = 0,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 training_signals: str = 'both'):
        super().__init__()
        self.doc_feat_dim = doc_feat_dim
        self.d_model = d_model
        self.training_signals = training_signals  # 'scores', 'features', or 'both'
        
        # Validate training_signals
        if training_signals not in ('scores', 'features', 'both'):
            raise ValueError(f"training_signals must be 'scores', 'features', or 'both', got {training_signals}")
        
        # Calculate input dimension based on training_signals
        base_input_dim = 0
        if training_signals in ('scores', 'both'):
            base_input_dim += 5  # score_z, score_rank, gap_z, pos_norm, topk
        if training_signals in ('features', 'both'):
            base_input_dim += doc_feat_dim
        
        self.input_proj = nn.Linear(base_input_dim, d_model)
        
        # Learnable CLS token for list-level representation
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: aggregate and score
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
    
    @staticmethod
    def _safe_z(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Z-normalize a tensor with improved numerical stability."""
        mu = x.mean()
        std = x.std(unbiased=False).clamp_min(eps)
        z = (x - mu) / std
        # Clamp extreme values to prevent NaN propagation
        z = torch.clamp(z, min=-10.0, max=10.0)
        return z
    
    def _build_features(self,
                        base_scores_ranked: torch.Tensor,
                        doc_features_ranked: torch.Tensor | None,
                        k_ref: int = 10) -> torch.Tensor:
        """
        Build per-document features for transformer input.
        
        Args:
            base_scores_ranked: (N,) scores in candidate ranking order
            doc_features_ranked: (N, D) or None
            k_ref: reference k for top-k indicator
        
        Returns:
            Feature matrix with shape depending on training_signals:
            - 'scores': (N, 5)
            - 'features': (N, D)
            - 'both': (N, 5+D)
        """
        device = base_scores_ranked.device
        N = base_scores_ranked.numel()
        eps = 1e-6
        
        features_list = []
        
        # Include score-based features if training_signals is 'scores' or 'both'
        if self.training_signals in ('scores', 'both'):
            # 1) score z-normalized within the list
            score_z = self._safe_z(base_scores_ranked, eps=eps)
            
            # 2) score rank position within the list (0 best .. 1 worst)
            score_order = base_scores_ranked.argsort(descending=True)
            score_rank = score_order.argsort().float()
            score_rank = score_rank / (max(N - 1, 1))
            
            # 3) local score gap to next document
            gap = torch.zeros(N, device=device)
            if N > 1:
                gap[:-1] = base_scores_ranked[:-1] - base_scores_ranked[1:]
            gap_z = self._safe_z(gap, eps=eps)
            
            # 4) normalized position (0..1)
            pos = torch.arange(N, device=device).float()
            pos_norm = pos / (max(N - 1, 1))
            
            # 5) top-k indicator
            topk = (pos < min(k_ref, N)).float()
            
            score_features = torch.stack([score_z, score_rank, gap_z, pos_norm, topk], dim=-1)  # (N, 5)
            features_list.append(score_features)
        
        # Include document features if training_signals is 'features' or 'both'
        if self.training_signals in ('features', 'both'):
            if doc_features_ranked is not None:
                # Safeguard: clamp extreme feature values to prevent NaN propagation
                doc_feat_safe = torch.clamp(doc_features_ranked, min=-1e3, max=1e3)
                # Replace any NaN values with 0
                doc_feat_safe = torch.where(torch.isnan(doc_feat_safe), torch.zeros_like(doc_feat_safe), doc_feat_safe)
                features_list.append(doc_feat_safe)
            elif self.training_signals == 'features':
                raise ValueError("training_signals='features' but doc_features_ranked is None")
        
        if not features_list:
            raise ValueError(f"No features to concatenate for training_signals={self.training_signals}")
        
        z = torch.cat(features_list, dim=-1)
        return z
    
    def forward(self,
                base_scores_ranked: torch.Tensor,
                doc_features_ranked: torch.Tensor | None = None) -> torch.Tensor:
        """
        Score a candidate ranking using transformer.
        
        Args:
            base_scores_ranked: (N,) scores in ranking order
            doc_features_ranked: (N, D) optional features in ranking order
        
        Returns:
            Scalar score for this ranking
        """
        # Build features
        z = self._build_features(base_scores_ranked, doc_features_ranked)  # (N, 5+D)
        
        # Project to d_model
        x = self.input_proj(z)  # (N, d_model)
        
        # Add batch dimension
        x = x.unsqueeze(0)  # (1, N, d_model)
        
        # Prepend CLS token
        cls = self.cls.expand(1, 1, self.d_model)  # (1, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (1, N+1, d_model)
        
        # Add positional encoding (CRITICAL for position-awareness)
        x = self.pos_encoding(x)  # (1, N+1, d_model)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x)  # (1, N+1, d_model)
        
        # Extract CLS token representation
        h = transformer_out[:, 0, :]  # (1, d_model)
        
        # Score
        score = self.output_head(h)  # (1, 1)
        score = score.squeeze(-1).squeeze(0)  # scalar
        
        # Safeguard against NaN scores
        if torch.isnan(score):
            score = torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        return score


# ============================================================
# Transformer-based Ranking Evaluator Loss
# ============================================================

class RankingEvaluatorTransformerLoss(nn.Module):
    """
    Non-leaky transformer-based evaluator loss.
    
    Uses transformer architecture instead of MLP for scoring candidate rankings.
    Labels are used ONLY to compute best_idx (supervision).
    Evaluator scores candidates using base_scores (+ optional doc_features), never labels.
    """
    
    def __init__(self,
                 doc_feat_dim: int = 0,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 ndcg_k: int = 10,
                 num_candidates: int = 10,
                 temperature: float = 0.1,
                 training_signals: str = 'both'):
        """
        Args:
            doc_feat_dim: Dimension of optional document features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network in transformer
            dropout: Dropout rate
            ndcg_k: k for NDCG@k computation
            num_candidates: Number of candidate rankings to generate
            temperature: Temperature for soft target distribution from NDCG scores
            training_signals: 'scores', 'features', or 'both' - which signals to use for training
        """
        super().__init__()
        self.evaluator = TransformerEvaluatorNonLeaky(
            doc_feat_dim=doc_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            training_signals=training_signals
        )
        self.ndcg_k = ndcg_k
        self.num_candidates = num_candidates
        self.temperature = temperature
        
        # Pairwise training attributes
        self.num_pairs_per_query = 64
        self.pair_margin = 0.0
        self.pair_weight_by_delta = True
    
    def generate_candidates(self, num_docs: int, device: torch.device, 
                          base_scores: torch.Tensor = None) -> list[torch.Tensor]:
        """
        Generate structured candidate rankings via perturbations.
        
        Instead of random permutations, we generate candidates that are plausible
        at inference time by perturbing the base ranking.
        
        Args:
            num_docs: Number of documents
            device: Device to create tensors on
            base_scores: Base model scores (if available) to create base ranking
        
        Returns:
            List of candidate rankings as index tensors
        """
        candidates = []
        
        # Get base ranking if scores provided
        if base_scores is not None:
            base_ranking = torch.argsort(base_scores, descending=True)
        else:
            base_ranking = torch.arange(num_docs, device=device)
        
        # 1. Original base ranking
        candidates.append(base_ranking.clone())
        
        # 2-N. Structured perturbations
        num_perturbations = self.num_candidates - 1
        
        for i in range(num_perturbations):
            perturbed = base_ranking.clone()
            
            # Mix of perturbation strategies:
            strategy = i % 4
            
            if strategy == 0:
                # Swap within sliding window (top-10)
                window_size = min(10, num_docs)
                if window_size > 1:
                    idx1 = torch.randint(0, window_size, (1,), device=device).item()
                    idx2 = torch.randint(0, window_size, (1,), device=device).item()
                    if idx1 != idx2:
                        perturbed[idx1], perturbed[idx2] = perturbed[idx2].clone(), perturbed[idx1].clone()
            
            elif strategy == 1:
                # Swap within larger window (top-20)
                window_size = min(20, num_docs)
                if window_size > 1:
                    idx1 = torch.randint(0, window_size, (1,), device=device).item()
                    idx2 = torch.randint(0, window_size, (1,), device=device).item()
                    if idx1 != idx2:
                        perturbed[idx1], perturbed[idx2] = perturbed[idx2].clone(), perturbed[idx1].clone()
            
            elif strategy == 2:
                # Temperature-based sort with noise
                temperature = 0.1 + 0.3 * torch.rand(1, device=device).item()
                noise = torch.randn(num_docs, device=device) * temperature
                noisy_scores = base_scores[base_ranking] + noise if base_scores is not None else noise
                perturbed = base_ranking[torch.argsort(noisy_scores, descending=True)].clone()
            
            else:
                # Multiple random swaps (2-5 swaps)
                num_swaps = torch.randint(2, 6, (1,), device=device).item()
                for _ in range(num_swaps):
                    idx1 = torch.randint(0, num_docs, (1,), device=device).item()
                    idx2 = torch.randint(0, num_docs, (1,), device=device).item()
                    if idx1 != idx2:
                        perturbed[idx1], perturbed[idx2] = perturbed[idx2].clone(), perturbed[idx1].clone()
            
            candidates.append(perturbed)
        
        # Validation: ensure all candidates are valid permutations
        for i, cand in enumerate(candidates):
            if len(cand) != num_docs:
                raise ValueError(f"Candidate {i} has wrong length: {len(cand)} != {num_docs}")
            # Check all elements 0..num_docs-1 are present exactly once
            sorted_cand = torch.sort(cand)[0]
            expected = torch.arange(num_docs, device=device)
            if not torch.equal(sorted_cand, expected):
                raise ValueError(
                    f"Candidate {i} is not a valid permutation. "
                    f"Expected elements {expected.tolist()}, got {sorted_cand.tolist()}"
                )
        
        return candidates
    
    def score_candidate(self,
                        ranking_indices: torch.Tensor,
                        base_scores: torch.Tensor,
                        doc_features: torch.Tensor | None) -> torch.Tensor:
        """
        Score a single candidate ranking.
        
        Args:
            ranking_indices: (N,) permutation indices
            base_scores: (N,) in original doc order
            doc_features: (N, D) in original doc order, optional
        
        Returns:
            Scalar score for this ranking
        """
        scores_ranked = base_scores[ranking_indices]  # (N,)
        feats_ranked = doc_features[ranking_indices] if doc_features is not None else None
        return self.evaluator(scores_ranked, feats_ranked)
    
    def score_candidates_batch(self,
                               candidate_rankings: list[torch.Tensor],
                               base_scores: torch.Tensor,
                               doc_features: torch.Tensor | None) -> torch.Tensor:
        """
        Score multiple candidate rankings in a single batched forward pass.
        
        This is ~20x faster than calling score_candidate() in a loop.
        
        Args:
            candidate_rankings: List of K candidate rankings, each (N,) tensor
            base_scores: (N,) base scores in original doc order
            doc_features: (N, D) doc features in original doc order, optional
        
        Returns:
            (K,) tensor of scores for each candidate
        """
        K = len(candidate_rankings)
        N = base_scores.numel()
        device = base_scores.device
        
        # Stack all rankings: (K, N)
        rankings_stacked = torch.stack(candidate_rankings, dim=0)
        
        # Gather scores for all candidates at once: (K, N)
        base_scores_expanded = base_scores.unsqueeze(0).expand(K, -1)
        scores_ranked_batch = torch.gather(base_scores_expanded, dim=1, index=rankings_stacked)
        
        # Gather features if available: (K, N, D)
        if doc_features is not None:
            # Expand features: (1, N, D) -> (K, N, D)
            doc_features_expanded = doc_features.unsqueeze(0).expand(K, -1, -1)
            # Gather along document dimension using rankings
            feats_ranked_batch = torch.gather(
                doc_features_expanded, 
                dim=1, 
                index=rankings_stacked.unsqueeze(-1).expand(-1, -1, doc_features.size(-1))
            )
        else:
            feats_ranked_batch = None
        
        # Build features for all candidates: (K, N, feature_dim)
        z_batch = []
        for k in range(K):
            scores_k = scores_ranked_batch[k]  # (N,)
            feats_k = feats_ranked_batch[k] if feats_ranked_batch is not None else None
            z_k = self.evaluator._build_features(scores_k, feats_k)  # (N, feature_dim)
            z_batch.append(z_k)
        z_batch = torch.stack(z_batch, dim=0)  # (K, N, feature_dim)
        
        # Project to d_model: (K, N, d_model)
        x = self.evaluator.input_proj(z_batch)
        
        # Prepend CLS token for each candidate: (K, 1, d_model)
        cls = self.evaluator.cls.expand(K, 1, self.evaluator.d_model)
        x = torch.cat([cls, x], dim=1)  # (K, N+1, d_model)
        
        # Add positional encoding
        x = self.evaluator.pos_encoding(x)  # (K, N+1, d_model)
        
        # Transformer encoder processes all K candidates in batch
        transformer_out = self.evaluator.transformer_encoder(x)  # (K, N+1, d_model)
        
        # Extract CLS token representation for each candidate
        h = transformer_out[:, 0, :]  # (K, d_model)
        
        # Score all candidates
        scores = self.evaluator.output_head(h)  # (K, 1)
        scores = scores.squeeze(-1)  # (K,)
        
        # Safeguard against NaN scores
        nan_mask = torch.isnan(scores)
        if nan_mask.any():
            scores = torch.where(nan_mask, torch.zeros_like(scores), scores)
        
        return scores
    
    def _sample_pairs(self, ndcg_scores: torch.Tensor, num_pairs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample pairs of candidates for pairwise preference learning.
        
        Returns:
          i_idx: (P,) candidate indices
          j_idx: (P,) candidate indices
          y:     (P,) labels in {0,1} where 1 means i beats j
        """
        K = ndcg_scores.numel()
        device = ndcg_scores.device

        # sample with replacement
        i_idx = torch.randint(0, K, (num_pairs,), device=device)
        j_idx = torch.randint(0, K, (num_pairs,), device=device)

        # avoid i==j (resample those positions)
        same = (i_idx == j_idx)
        if same.any():
            j_idx[same] = (j_idx[same] + torch.randint(1, K, (same.sum(),), device=device)) % K

        # y = 1 if ndcg_i > ndcg_j else 0
        y = (ndcg_scores[i_idx] > ndcg_scores[j_idx]).float()
        return i_idx, j_idx, y
    
    def forward(self,
                labels: torch.Tensor,
                base_scores: torch.Tensor,
                doc_features: torch.Tensor | None = None,
                candidate_rankings: list | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss and return evaluator's choice.
        
        Args:
            labels: (N,) ground-truth (used only to compute best_idx/ndcg_scores)
            base_scores: (N,) model scores (available at inference)
            doc_features: (N, D) optional (available at inference)
            candidate_rankings: optional list of permutations
        
        Returns:
            loss: Cross-entropy loss
            chosen_idx: Index of ranking evaluator picked
            ndcg_scores: (K,) NDCG@k for each candidate
            correct: Boolean - did evaluator pick the best?
        """
        device = labels.device
        N = labels.numel()
        
        if candidate_rankings is None:
            candidate_rankings = self.generate_candidates(N, device, base_scores)
        
        K = len(candidate_rankings)
        
        # 1) Oracle NDCG for each candidate (supervision only)
        # Use batched GPU computation instead of Python loop
        # This is ~50-100x faster for large datasets like WEB10K
        ndcg_scores = compute_ndcg_batch(labels, candidate_rankings, k=self.ndcg_k)
        best_idx = ndcg_scores.argmax()
        spread = (ndcg_scores.max() - ndcg_scores.min()).item()
        std = ndcg_scores.std().item()
        #print(f"NDCG spread={spread:.4f} std={std:.4f} K={K}")

        
        # 2) Evaluator scores candidates WITHOUT labels
        # Use batched scoring instead of loop for ~20x speedup
        eval_scores = self.score_candidates_batch(candidate_rankings, base_scores, doc_features)  # (K,)
        
        # -----------------------------
        # Pairwise preference loss
        # -----------------------------
        num_pairs = getattr(self, "num_pairs_per_query", None)
        if num_pairs is None:
            # fallback if you didn't store it as attribute; set default
            num_pairs = 64

        i_idx, j_idx, y = self._sample_pairs(ndcg_scores, num_pairs=num_pairs)

        # score difference: s_i - s_j
        s_i = eval_scores[i_idx]
        s_j = eval_scores[j_idx]
        diff = s_i - s_j

        # optional margin (usually 0.0)
        margin = getattr(self, "pair_margin", 0.0)
        if margin != 0.0:
            diff = diff - margin

        # base RankNet / Bradley-Terry
        pair_loss = F.binary_cross_entropy_with_logits(diff, y)


        t = self.temperature
        p_target = torch.softmax(ndcg_scores / t, dim=0).detach()
        p_pred   = torch.log_softmax(eval_scores / t, dim=0)
        listwise_kl = F.kl_div(p_pred, p_target, reduction="batchmean") * (t * t)

        # optional: weight by ndcg gap to focus meaningful preferences
        pair_weight_by_delta = getattr(self, "pair_weight_by_delta", True)
        if pair_weight_by_delta:
            delta = (ndcg_scores[i_idx] - ndcg_scores[j_idx]).abs()
            # normalize weights to keep loss scale stable
            # Adaptive clipping to prevent gradient explosion with small deltas
            delta_mean = delta.mean()
            # Clamp the denominator to a reasonable range based on delta magnitude
            scale_factor = torch.clamp(delta_mean, min=0.01, max=0.5)
            w = delta / (scale_factor + 1e-8)
            # Clamp weights to prevent extreme values
            w = torch.clamp(w, min=0.1, max=10.0)
            pair_loss = (F.binary_cross_entropy_with_logits(diff, y, reduction="none") * w).mean()

        loss = pair_loss  + 0.2 * listwise_kl #0.05 * listwise_kl
        
        # Safeguard against NaN loss - use eval_scores to maintain gradient flow
        if torch.isnan(loss):
            # Create a small loss from eval_scores to maintain gradient connection
            loss = (eval_scores.mean() * 0.0) + 0.1  # 0.7 is a typical loss value
        
        # Pairwise training accuracy (debug)
        with torch.no_grad():
            # evaluate accuracy on the sampled pairs
            pred = (diff > 0).float()
            pair_acc = (pred == y).float().mean()
        
        # Choice + correctness (for monitoring)
        chosen_idx = eval_scores.argmax()
        best_idx = ndcg_scores.argmax()
        correct = (chosen_idx == best_idx)
        
        return loss, chosen_idx, ndcg_scores, correct


# ============================================================
# Batch Wrapper
# ============================================================

class BatchRankingEvaluatorTransformerLoss(nn.Module):
    """Wrapper to handle batches of queries."""
    
    def __init__(self,
                 doc_feat_dim: int = 0,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 ndcg_k: int = 10,
                 num_candidates: int = 10,
                 temperature: float = 0.1,
                 training_signals: str = 'both'):
        super().__init__()
        self.loss_fn = RankingEvaluatorTransformerLoss(
            doc_feat_dim=doc_feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            ndcg_k=ndcg_k,
            num_candidates=num_candidates,
            temperature=temperature,
            training_signals=training_signals
        )
    
    def forward(self,
                batch_labels: list[torch.Tensor],
                batch_base_scores: list[torch.Tensor],
                batch_doc_features: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, float, float]:
        """
        Process a batch of queries.
        
        Args:
            batch_labels: List of B tensors, each (N_i,) with relevance labels
            batch_base_scores: List of B tensors, each (N_i,) with base scores
            batch_doc_features: List of B tensors, each (N_i, D) with doc features, optional
        
        Returns:
            total_loss: Average loss across batch
            accuracy: Fraction of queries where evaluator picked best
            mean_ndcg: Mean NDCG@k of picked rankings
        """
        losses = []
        correct = 0
        picked_ndcgs = []
        
        if batch_doc_features is None:
            batch_doc_features = [None] * len(batch_labels)
        
        for y, s, x in zip(batch_labels, batch_base_scores, batch_doc_features):
            loss, chosen_idx, ndcg_scores, is_correct = self.loss_fn(y, s, x)
            losses.append(loss)
            correct += int(is_correct.item())
            picked_ndcgs.append(ndcg_scores[chosen_idx].item())
        
        total_loss = torch.stack(losses).mean()
        acc = correct / max(len(batch_labels), 1)
        mean_ndcg = float(np.mean(picked_ndcgs)) if picked_ndcgs else 0.0
        return total_loss, acc, mean_ndcg
