"""
Modular Training Framework with Integrated Reranking

Extends modular_train_allrank.py with a reranking block that:
1. Captures model predictions during training on each fold
2. Constructs ideal rankings (sorted by labels) and candidate rankings (random shuffles)
3. Trains a transformer-based reranker on base scores (not document features)
4. At test time, loads fold predictions and applies reranking
5. Reports NDCG@5, NDCG@10, MRR@5, MRR@10 before and after reranking

Usage:
    python modular_reranker.py --config configs/modular_config.json --ranker transformer --loss ranknet
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import argparse
import os
import json
from os import path
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy

# Import from modular_train_allrank
from modular_train_allrank import (
    Logger, BaseRankerFactory, TransformerRankerFactory, BaseLossFactory,
    RankNetLossFactory, LRCLLossFactory, SoftNDCGLossFactory, NeuralNDCGLossFactory,
    NeuralLossFactory, LambdaLossFactory, LambdaRankFactory, ModularTrainingFramework, 
    SoftNDCGLossFactory,
    load_modular_config, _patch_ranker_for_test_slate_length
)

# Import reranker components
from reranker_transformer_loss import RankingEvaluatorTransformerLoss

# Import unified NDCG calculation
from ndcg_utils import compute_ndcg as compute_ndcg_unified


# ============================================================
# Reranker Configuration
# ============================================================

class RerankerConfig:
    """Configuration for the reranker component"""
    
    # Reranker architecture
    DOC_FEAT_DIM = 46  # MQ2008 has 46 features, will be auto-detected from data
    TRANSFORMER_D_MODEL = 128
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_NUM_LAYERS = 4
    TRANSFORMER_DIM_FEEDFORWARD = 256
    TRANSFORMER_DROPOUT = 0.0
    
    # Training
    NUM_CANDIDATES = 20  # Increased from 10 to 32 for better signal
    LEARNING_RATE = 1e-3  # Increased from 1e-5 to 1e-4
    NUM_EPOCHS = 20
    SEED = 42
    TEMPERATURE = 0.01 #0.25  # Temperature for soft target distribution
    BATCH_SIZE = 128  # Number of queries to process per batch during training and evaluation
    TRAINING_SIGNALS = 'both'  # 'scores', 'features', or 'both' - which signals to use for training
    
    # Pairwise training config
    NUM_PAIRS_PER_QUERY = 256     # 16–128 depending on speed
    PAIR_MARGIN = 0.0            # keep 0.0 initially
    PAIR_WEIGHT_BY_DELTA = True  # emphasize bigger NDCG gaps
    DELTA_EPS = 1e-6
    
    
    # Confidence-based safety (valid for publication - no oracle)
    CONFIDENCE_BASED_SAFETY = False   # If True, fallback to original when model has low confidence
    CONFIDENCE_THRESHOLD = 0.0001       # Minimum score gap between best and 2nd best candidate (normalized)
    CONFIDENCE_ENTROPY_THRESHOLD = 10.0 # Maximum entropy in predicted distribution (lower = more confident)
    
    # Evaluation
    NDCG_K_VALUES = [1, 3, 5, 10]
    MRR_K_VALUES = [1, 3, 5, 10]


# ============================================================
# Metrics Computation
# ============================================================

# Use unified NDCG function from ndcg_utils
compute_ndcg = compute_ndcg_unified


def compute_mrr(labels: np.ndarray, k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank@k
    
    Matches allrank implementation: finds the position of the FIRST OCCURRENCE of the 
    MAXIMUM relevance value in the ranking, then returns 1/(position+1) if within top-k.
    
    This is NOT "first relevant document" - it's the position of the highest relevance.
    
    Args:
        labels: Relevance labels already in ranking order (top-k)
        k: Position cutoff (for bounds checking)
    
    Returns:
        MRR@k value (0.0 if max relevance position is beyond top-k)
    """
    labels = np.array(labels, dtype=np.float32)
    
    # Find the maximum relevance value
    max_relevance = np.max(labels)
    
    # If all zeros, return 0
    if max_relevance == 0.0:
        return 0.0
    
    # Find first position where relevance equals max (first occurrence of max)
    max_positions = np.where(labels == max_relevance)[0]
    first_max_pos = max_positions[0]
    
    # Only count if within top-k (position is 0-indexed, so position < k)
    if first_max_pos < k:
        return 1.0 / (first_max_pos + 1)
    else:
        return 0.0


def compute_metrics_before_after(labels: np.ndarray, original_ranking: np.ndarray, 
                                 reranked_ranking: np.ndarray, k_values: List[int]) -> Dict[str, float]:
    """
    Compute metrics before and after reranking
    
    Args:
        labels: relevance labels (all documents)
        original_ranking: indices of original ranking
        reranked_ranking: indices of reranked ranking
        k_values: list of k values to compute metrics at
    
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    for k in k_values:
        # Original ranking metrics
        orig_labels_topk = labels[original_ranking[:k]]
        orig_ndcg = compute_ndcg(orig_labels_topk, labels_all=labels, k=k)
        orig_mrr = compute_mrr(orig_labels_topk, k=k)
        
        # Reranked ranking metrics
        reranked_labels_topk = labels[reranked_ranking[:k]]
        reranked_ndcg = compute_ndcg(reranked_labels_topk, labels_all=labels, k=k)
        reranked_mrr = compute_mrr(reranked_labels_topk, k=k)
        
        metrics[f'ndcg@{k}_before'] = orig_ndcg
        metrics[f'ndcg@{k}_after'] = reranked_ndcg
        metrics[f'ndcg@{k}_improvement'] = reranked_ndcg - orig_ndcg
        
        metrics[f'mrr@{k}_before'] = orig_mrr
        metrics[f'mrr@{k}_after'] = reranked_mrr
        metrics[f'mrr@{k}_improvement'] = reranked_mrr - orig_mrr
    
    return metrics


# ============================================================
# Reranker Trainer
# ============================================================

class RerankerTrainer:
    """Trains a transformer-based reranker on model predictions"""
    
    def __init__(self, device: str = 'cpu', logger=None, config: RerankerConfig = None, doc_feat_dim: int = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or RerankerConfig()
        
        # Use provided doc_feat_dim or default from config
        self.doc_feat_dim = doc_feat_dim if doc_feat_dim is not None else self.config.DOC_FEAT_DIM
        
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        
        # Reranker model: configurable training signals (scores, features, or both)
        self.model = RankingEvaluatorTransformerLoss(
            doc_feat_dim=self.doc_feat_dim,
            d_model=self.config.TRANSFORMER_D_MODEL,
            nhead=self.config.TRANSFORMER_NHEAD,
            num_layers=self.config.TRANSFORMER_NUM_LAYERS,
            dim_feedforward=self.config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=self.config.TRANSFORMER_DROPOUT,
            ndcg_k=10,
            num_candidates=self.config.NUM_CANDIDATES,
            temperature=self.config.TEMPERATURE,
            training_signals=self.config.TRAINING_SIGNALS,
        ).to(device)
        
        self.logger.info(f"Initialized reranker with doc_feat_dim={self.doc_feat_dim}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
    
    def _build_candidates(self, labels: torch.Tensor, predictions: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Generate structured candidate rankings for training.
        
        Uses the model's generate_candidates method which creates plausible
        perturbations of the base ranking instead of random permutations.
        This ensures candidates are realistic and learnable from base scores.
        
        Args:
            labels: relevance labels (not used for candidate generation)
            predictions: base model predictions (used to create base ranking)
        
        Returns:
            List of candidate rankings (as index tensors)
        
        Note: We no longer include the label-ideal permutation since it's not
        available at inference time and creates a train-test mismatch.
        """
        N = labels.numel()
        device = labels.device
        
        # Use the model's structured candidate generation
        # This creates perturbations of the base ranking that are plausible at inference
        if predictions is not None:
            candidates = self.model.generate_candidates(N, device, predictions)
        else:
            # If no predictions, use identity ranking as base
            candidates = self.model.generate_candidates(N, device, None)
        
        return candidates
    
    def train_on_predictions(self, queries_with_predictions: List[Dict]) -> Dict[str, float]:
        """
        Train reranker on model predictions using batch processing
        
        Args:
            queries_with_predictions: List of dicts with 'labels', 'predictions' keys
        
        Returns:
            Training statistics
        """
        self.model.train()
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Process queries in batches
            for batch_start in range(0, len(queries_with_predictions), self.config.BATCH_SIZE):
                batch_end = min(batch_start + self.config.BATCH_SIZE, len(queries_with_predictions))
                batch = queries_with_predictions[batch_start:batch_end]
                
                # Zero gradients once per batch
                self.optimizer.zero_grad()
                batch_loss = 0.0
                
                for q in batch:
                    labels = torch.from_numpy(q['labels']).float().to(self.device)
                    predictions = torch.from_numpy(q['predictions']).float().to(self.device)
                    
                    # Get document features if available
                    doc_features = None
                    if 'features' in q and q['features'] is not None and len(q['features']) > 0:
                        doc_features = torch.from_numpy(q['features']).float().to(self.device)
                    
                    # Build candidates including the original ranking from predictions
                    candidate_rankings = self._build_candidates(labels, predictions)
                    
                    # Train on predictions AND document features
                    loss, chosen_idx, ndcg_scores, correct = self.model(
                        labels=labels,
                        base_scores=predictions,
                        doc_features=doc_features,
                        candidate_rankings=candidate_rankings
                    )
                    
                    # Scale loss by batch size to maintain correct gradient magnitude
                    # This ensures gradients are averaged across the batch
                    scaled_loss = loss / len(batch)
                    batch_loss += scaled_loss
                    
                    epoch_loss += loss.item()
                    epoch_correct += int(correct.item())
                
                # Backpropagate accumulated gradients once per batch
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            avg_epoch_loss = epoch_loss / max(len(queries_with_predictions), 1)
            avg_epoch_acc = epoch_correct / max(len(queries_with_predictions), 1)
            
            if epoch % 1 == 0 or epoch == self.config.NUM_EPOCHS:
                self.logger.info(f"Reranker Epoch {epoch}/{self.config.NUM_EPOCHS} | loss={avg_epoch_loss:.4f} acc={avg_epoch_acc:.4f}")
        
        return {
            'final_loss': avg_epoch_loss,
            'final_accuracy': avg_epoch_acc
        }
    
    def rerank_predictions(self, labels: np.ndarray, predictions: np.ndarray, 
                          doc_features: np.ndarray = None, num_candidates: int = None, 
                          debug: bool = False) -> Tuple[np.ndarray, float, bool]:
        """
        Rerank predictions using trained reranker
        
        Args:
            labels: relevance labels
            predictions: base model scores
            doc_features: document features (N x feat_dim)
            num_candidates: number of candidate rankings to generate (default: use config)
            debug: whether to print debug information
        
        Returns:
            Tuple of (reranked_indices, reranker_score, confidence_fallback_used)
            - reranked_indices: ranking to use (may be original if confidence check fails)
            - reranker_score: score from reranker model
            - confidence_fallback_used: True if we fell back to original ranking due to low confidence
        """
        self.model.eval()
        
        if num_candidates is None:
            num_candidates = self.config.NUM_CANDIDATES
        
        N = len(predictions)
        device = self.device
        
        # Original ranking (candidate 0)
        original_ranking = np.argsort(-predictions)
        
        # Generate structured candidates using the model's method
        pred_t = torch.from_numpy(predictions).to(device)
        candidates_t = self.model.generate_candidates(N, device, pred_t)
        
        # Convert to numpy for compatibility
        candidates = [cand.cpu().numpy() for cand in candidates_t[:num_candidates]]
        
        # Score each candidate
        with torch.no_grad():
            pred_t = torch.from_numpy(predictions).to(device)
            doc_feat_t = torch.from_numpy(doc_features).to(device) if doc_features is not None else None
            scores = []
            
            for cand in candidates:
                cand_t = torch.as_tensor(cand, device=device, dtype=torch.long)
                score = self.model.score_candidate(cand_t, pred_t, doc_feat_t).item()
                scores.append(score)
        
        # Select best candidate according to model
        best_idx = int(np.argmax(scores))
        reranked = candidates[best_idx]
        reranker_score = float(scores[best_idx])
        
        # Confidence-based safety: fallback if model is uncertain (NO ORACLE)
        confidence_fallback_used = False
        if self.config.CONFIDENCE_BASED_SAFETY:
            scores_array = np.array(scores)
            
            # Method 1: Score gap between best and second-best
            sorted_scores = np.sort(scores_array)[::-1]  # Descending order
            if len(sorted_scores) >= 2:
                score_gap = sorted_scores[0] - sorted_scores[1]
                # Normalize by score range to make threshold interpretable
                score_range = sorted_scores[0] - sorted_scores[-1]
                normalized_gap = score_gap / (score_range + 1e-8)
            else:
                normalized_gap = 1.0  # Only one candidate, use it
            
            # Method 2: Entropy of predicted distribution
            # Convert scores to probabilities using softmax
            exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
            probs = exp_scores / np.sum(exp_scores)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            # Fallback to original if low confidence
            low_confidence = (normalized_gap < self.config.CONFIDENCE_THRESHOLD or 
                            entropy > self.config.CONFIDENCE_ENTROPY_THRESHOLD)
            
            if low_confidence and best_idx != 0:  # Don't fallback if already chose original
                reranked = original_ranking
                confidence_fallback_used = True
                
                if debug:
                    print(f"  CONFIDENCE FALLBACK: gap={normalized_gap:.4f} < {self.config.CONFIDENCE_THRESHOLD}, "
                          f"entropy={entropy:.4f} > {self.config.CONFIDENCE_ENTROPY_THRESHOLD}")
        
        if debug:
            # Compute actual NDCG for each candidate for comparison
            from modular_reranker import compute_ndcg as compute_ndcg_fn
            actual_ndcgs = []
            for cand in candidates:
                labels_ranked = labels[cand]
                ndcg = compute_ndcg_fn(labels_ranked, labels_all=labels, k=10)
                actual_ndcgs.append(ndcg)
            
            print(f"  Model scores:  {[f'{s:.4f}' for s in scores]}")
            print(f"  Actual NDCG@10: {[f'{n:.4f}' for n in actual_ndcgs]}")
            print(f"  Best candidate index: {best_idx} (0=original, 1-{num_candidates-1}=shuffled)")
            print(f"  Model picked NDCG={actual_ndcgs[best_idx]:.4f}, Best possible={max(actual_ndcgs):.4f}")
            print(f"  Original kept: {best_idx == 0 or confidence_fallback_used}")
            if confidence_fallback_used:
                print(f"  Confidence fallback: YES")
        
        return reranked, reranker_score, confidence_fallback_used


# ============================================================
# Main Training with Reranking
# ============================================================

class ModularRerankerFramework:
    """Extends ModularTrainingFramework with reranking"""
    
    def __init__(self, logger=None):
        self.base_framework = ModularTrainingFramework()
        self.reranker_config = RerankerConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def train_with_reranking(self, ranker_type: str, loss_type: str, rank_config, logger, 
                            device: str, tag: str, loss_params: Optional[Dict[str, Any]] = None,
                            wandb_logger=None, dataset_name: str = None) -> Dict[str, Any]:
        """
        Train ranker and reranker, return results before and after reranking
        """
        
        # Step 1: Train base ranker
        logger.info("=" * 80)
        logger.info("STEP 1: Training base ranker")
        logger.info("=" * 80)
        
        ranking_results = self.base_framework.train(
            ranker_type=ranker_type,
            loss_type=loss_type,
            rank_config=rank_config,
            logger=logger,
            device=device,
            tag=tag,
            loss_params=loss_params,
            wandb_logger=wandb_logger,
            dataset_name=dataset_name
        )
        
        logger.info(f"Base ranker results: {ranking_results}")
        
        # Step 2: Collect training predictions for reranker training
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Collecting training predictions for reranker")
        logger.info("=" * 80)
        
        # Load train predictions saved by the ranker
        train_predictions = self._load_train_predictions(rank_config, dataset_name, loss_type, tag)
        
        # Auto-detect feature dimension from first query
        doc_feat_dim = 0
        if train_predictions and 'features' in train_predictions[0] and train_predictions[0]['features'] is not None:
            doc_feat_dim = train_predictions[0]['features'].shape[1]
            logger.info(f"Auto-detected document feature dimension: {doc_feat_dim}")
        
        # Step 3: Train reranker
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Training reranker")
        logger.info("=" * 80)
        
        reranker = RerankerTrainer(device=device, logger=logger, config=self.reranker_config, doc_feat_dim=doc_feat_dim)
        reranker_stats = reranker.train_on_predictions(train_predictions)
        logger.info(f"Reranker training stats: {reranker_stats}")
        
        # Step 4: Evaluate on test set with reranking
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Evaluating on test set with reranking")
        logger.info("=" * 80)
        
        test_predictions = self._load_test_predictions(rank_config, dataset_name, loss_type, tag)
        reranking_results = self._evaluate_with_reranking(
            test_predictions, reranker, logger, self.reranker_config
        )
        
        # Save results
        self._save_results(reranking_results, dataset_name, loss_type, tag)
        
        return {
            'base_ranker_results': ranking_results,
            'reranker_stats': reranker_stats,
            'reranking_results': reranking_results
        }
    
    def _load_features_from_dataset(self, rank_config, split: str = 'train') -> Dict[int, np.ndarray]:
        """
        Load document features directly from dataset files
        
        Returns:
            Dictionary mapping qid -> features array (num_docs x feat_dim)
        """
        try:
            from sklearn.datasets import load_svmlight_file
            
            # Get dataset path from config
            dataset_path = rank_config.data.path
            
            # Construct file path for the split
            if split == 'train':
                file_path = os.path.join(dataset_path, 'train.txt')
            elif split == 'vali':
                file_path = os.path.join(dataset_path, 'vali.txt')
            elif split == 'test':
                file_path = os.path.join(dataset_path, 'test.txt')
            else:
                raise ValueError(f"Unknown split: {split}")
            
            if not os.path.exists(file_path):
                print(f"Warning: dataset file not found at {file_path}")
                return {}
            
            # Load features and query IDs from LibSVM file
            X, y, query_ids = load_svmlight_file(file_path, query_id=True)
            
            # Convert sparse matrix to dense if needed
            if not isinstance(X, np.ndarray):
                X = X.toarray()
            
            # Group by query ID
            unique_qids, indices, counts = np.unique(query_ids, return_index=True, return_counts=True)
            groups = np.cumsum(counts[np.argsort(indices)])
            
            X_by_qid = np.split(X, groups)[:-1]
            qids_sorted = unique_qids[np.argsort(indices)]
            
            # Create dictionary mapping qid -> features
            features_dict = {}
            for qid, features in zip(qids_sorted, X_by_qid):
                features_dict[int(qid)] = features.astype(np.float32)
            
            print(f"Loaded features for {len(features_dict)} queries from {file_path}")
            return features_dict
            
        except Exception as e:
            print(f"Error loading features from dataset: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _load_train_predictions(self, rank_config, dataset_name: str, loss_type: str, tag: str) -> List[Dict]:
        """Load training predictions saved by the ranker (used for reranker training)"""
        try:
            # Load features from dataset
            features_dict = self._load_features_from_dataset(rank_config, split='train')
            
            # Load train predictions saved by the ranker
            # The predictions are saved in preds/dataset/loss_type/Fold{N}_train.txt
            preds_dir = os.path.join('preds', dataset_name, loss_type)
            fold_num = tag if tag else '1'
            pred_file = os.path.join(preds_dir, f'Fold{fold_num}_train.txt')
            
            if not os.path.exists(pred_file):
                print(f"Warning: train predictions file not found at {pred_file}")
                return []
            
            queries = []
            current_query = None
            
            # Parse the predictions file
            with open(pred_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('qid:'):
                        # Save previous query if exists
                        if current_query is not None:
                            # Convert features list to numpy array
                            if current_query['features']:
                                current_query['features'] = np.array(current_query['features'], dtype=np.float32)
                            queries.append(current_query)
                        
                        # Start new query
                        qid = line.split(':')[1]
                        current_query = {'qid': qid, 'labels': [], 'predictions': [], 'features': []}
                    
                    elif line.startswith('labels:'):
                        # Parse labels
                        labels_str = line.split('[')[1].split(']')[0]
                        labels = np.array([float(x) for x in labels_str.split()], dtype=np.float32)
                        current_query['labels'] = labels
                    
                    elif line.startswith('predictions:'):
                        # Parse predictions
                        preds_str = line.split('[')[1].split(']')[0]
                        predictions = np.array([float(x) for x in preds_str.split()], dtype=np.float32)
                        current_query['predictions'] = predictions
                    
                    elif line.startswith('features:'):
                        # Start reading features (multi-line)
                        current_query['features'] = []
                    
                    elif line.startswith('[') and current_query and 'features' in current_query and isinstance(current_query['features'], list):
                        # Parse a feature vector
                        feat_str = line[1:-1]  # Remove [ and ]
                        if feat_str.strip():
                            feat_vec = [float(x) for x in feat_str.split()]
                            current_query['features'].append(feat_vec)
                
                # Don't forget the last query
                if current_query is not None:
                    # Convert features list to numpy array
                    if current_query['features']:
                        current_query['features'] = np.array(current_query['features'], dtype=np.float32)
                    queries.append(current_query)
            
            # Merge in features from dataset for queries that don't have them
            for q in queries:
                qid = int(q['qid'])
                features = q.get('features')
                if qid in features_dict and (features is None or len(features) == 0):
                    dataset_features = features_dict[qid]
                    num_preds = len(q['predictions'])
                    num_feats = len(dataset_features)
                    
                    # Ensure features match predictions length
                    if num_feats == num_preds:
                        q['features'] = dataset_features
                    elif num_feats > num_preds:
                        # Dataset has more docs, take first num_preds
                        q['features'] = dataset_features[:num_preds]
                    else:
                        # Dataset has fewer docs, pad with zeros
                        feat_dim = dataset_features.shape[1]
                        padded = np.zeros((num_preds, feat_dim), dtype=np.float32)
                        padded[:num_feats] = dataset_features
                        q['features'] = padded
            
            return queries
        except Exception as e:
            print(f"Error loading train predictions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _load_test_predictions(self, rank_config, dataset_name: str, loss_type: str, tag: str) -> List[Dict]:
        """Load test predictions from saved file"""
        # Load test predictions saved by the ranker
        try:
            # Load features from dataset
            features_dict = self._load_features_from_dataset(rank_config, split='test')
            
            # Construct the path to the saved predictions
            # The predictions are saved in preds/dataset/loss_type/Fold{N}_test.txt
            preds_dir = os.path.join('preds', dataset_name, loss_type)
            # tag should contain the fold number (e.g., '1'), or default to '1'
            fold_num = tag if tag else '1'
            pred_file = os.path.join(preds_dir, f'Fold{fold_num}_test.txt')
            
            if not os.path.exists(pred_file):
                print(f"Warning: predictions file not found at {pred_file}")
                return []
            
            queries = []
            current_query = None
            
            # Parse the predictions file
            with open(pred_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('qid:'):
                        # Save previous query if exists
                        if current_query is not None:
                            # Convert features list to numpy array
                            if current_query['features']:
                                current_query['features'] = np.array(current_query['features'], dtype=np.float32)
                            queries.append(current_query)
                        
                        # Start new query
                        qid = line.split(':')[1]
                        current_query = {'qid': qid, 'labels': [], 'predictions': [], 'features': []}
                    
                    elif line.startswith('labels:'):
                        # Parse labels
                        labels_str = line.split('[')[1].split(']')[0]
                        labels = np.array([float(x) for x in labels_str.split()], dtype=np.float32)
                        current_query['labels'] = labels
                    
                    elif line.startswith('predictions:'):
                        # Parse predictions
                        preds_str = line.split('[')[1].split(']')[0]
                        predictions = np.array([float(x) for x in preds_str.split()], dtype=np.float32)
                        current_query['predictions'] = predictions
                    
                    elif line.startswith('features:'):
                        # Start reading features (multi-line)
                        current_query['features'] = []
                    
                    elif line.startswith('[') and current_query and 'features' in current_query and isinstance(current_query['features'], list):
                        # Parse a feature vector
                        feat_str = line[1:-1]  # Remove [ and ]
                        if feat_str.strip():
                            feat_vec = [float(x) for x in feat_str.split()]
                            current_query['features'].append(feat_vec)
                
                # Don't forget the last query
                if current_query is not None:
                    # Convert features list to numpy array
                    if current_query['features']:
                        current_query['features'] = np.array(current_query['features'], dtype=np.float32)
                    queries.append(current_query)
            
            # Merge in features from dataset for queries that don't have them
            for q in queries:
                qid = int(q['qid'])
                features = q.get('features')
                if qid in features_dict and (features is None or len(features) == 0):
                    dataset_features = features_dict[qid]
                    num_preds = len(q['predictions'])
                    num_feats = len(dataset_features)
                    
                    # Ensure features match predictions length
                    if num_feats == num_preds:
                        q['features'] = dataset_features
                    elif num_feats > num_preds:
                        # Dataset has more docs, take first num_preds
                        q['features'] = dataset_features[:num_preds]
                    else:
                        # Dataset has fewer docs, pad with zeros
                        feat_dim = dataset_features.shape[1]
                        padded = np.zeros((num_preds, feat_dim), dtype=np.float32)
                        padded[:num_feats] = dataset_features
                        q['features'] = padded
            
            return queries
        except Exception as e:
            print(f"Error loading test predictions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _evaluate_with_reranking(self, test_predictions: List[Dict], reranker: RerankerTrainer,
                                logger, config: RerankerConfig) -> Dict[str, Any]:
        """Evaluate test set with and without reranking using batch processing"""
        
        # Store only per-query metrics temporarily for averaging
        metrics_before_list = {f'ndcg@{k}': [] for k in config.NDCG_K_VALUES}
        metrics_before_list.update({f'mrr@{k}': [] for k in config.MRR_K_VALUES})
        
        metrics_after_list = {f'ndcg@{k}': [] for k in config.NDCG_K_VALUES}
        metrics_after_list.update({f'mrr@{k}': [] for k in config.MRR_K_VALUES})
        
        queries_evaluated = 0
        queries_changed = 0  # Track how many queries had rankings changed
        
        # Process test predictions in batches
        for batch_start in range(0, len(test_predictions), config.BATCH_SIZE):
            batch_end = min(batch_start + config.BATCH_SIZE, len(test_predictions))
            batch = test_predictions[batch_start:batch_end]
            
            for batch_idx, q in enumerate(batch):
                q_idx = batch_start + batch_idx
                try:
                    labels = q['labels']
                    predictions = q['predictions']
                    doc_features = q.get('features', None)
                    
                    # Validate shapes match
                    if len(labels) != len(predictions):
                        print(f"Warning: labels size {len(labels)} != predictions size {len(predictions)}")
                        continue
                    
                    # Debug first 3 queries
                    debug_mode = q_idx < 3
                    if debug_mode:
                        logger.info(f"\nQuery {q_idx + 1} (qid: {q.get('qid', 'unknown')}):")
                    
                    # Original ranking
                    orig_ranking = np.argsort(-predictions)
                    
                    # Reranked ranking (with features)
                    reranked_ranking, reranker_score, confidence_fallback = reranker.rerank_predictions(
                        labels, predictions, doc_features=doc_features, debug=debug_mode
                    )
                    
                    # Track statistics
                    ranking_changed = not np.array_equal(orig_ranking, reranked_ranking)
                    if ranking_changed:
                        queries_changed += 1
                    
                    if debug_mode:
                        logger.info(f"  Ranking changed: {ranking_changed}")
                    
                    # Compute metrics
                    query_metrics = compute_metrics_before_after(
                        labels, orig_ranking, reranked_ranking, 
                        config.NDCG_K_VALUES + config.MRR_K_VALUES
                    )
                    
                    for metric_name, value in query_metrics.items():
                        if 'before' in metric_name:
                            key = metric_name.replace('_before', '')
                            metrics_before_list[key].append(value)
                        elif 'after' in metric_name:
                            key = metric_name.replace('_after', '')
                            metrics_after_list[key].append(value)
                    
                    queries_evaluated += 1
                except Exception as e:
                    print(f"Error evaluating query: {e}")
                    continue
        
        # Print diagnostic information
        logger.info(f"\nReranking Statistics:")
        logger.info(f"  Total queries evaluated: {queries_evaluated}")
        if queries_evaluated > 0:
            logger.info(f"  Queries where ranking changed: {queries_changed} ({100*queries_changed/queries_evaluated:.1f}%)")
            logger.info(f"  Queries where ranking kept original: {queries_evaluated - queries_changed} ({100*(queries_evaluated-queries_changed)/queries_evaluated:.1f}%)")
        else:
            logger.info("  WARNING: No queries were successfully evaluated!")
        logger.info("")
        
        # Compute averages only (don't store per-query metrics)
        results = {
            'queries_evaluated': queries_evaluated,
            'queries_changed': queries_changed,
            'summary': {}
        }
        
        for k in config.NDCG_K_VALUES:
            key = f'ndcg@{k}'
            if metrics_before_list[key]:
                before_avg = float(np.mean(metrics_before_list[key]))
                after_avg = float(np.mean(metrics_after_list[key]))
                results['summary'][f'ndcg@{k}_before'] = before_avg
                results['summary'][f'ndcg@{k}_after'] = after_avg
                results['summary'][f'ndcg@{k}_improvement'] = after_avg - before_avg
                logger.info(f"NDCG@{k} - Before: {before_avg:.6f}, After: {after_avg:.6f}, Improvement: {after_avg - before_avg:.6f}")
        
        for k in config.MRR_K_VALUES:
            key = f'mrr@{k}'
            if metrics_before_list[key]:
                before_avg = float(np.mean(metrics_before_list[key]))
                after_avg = float(np.mean(metrics_after_list[key]))
                results['summary'][f'mrr@{k}_before'] = before_avg
                results['summary'][f'mrr@{k}_after'] = after_avg
                results['summary'][f'mrr@{k}_improvement'] = after_avg - before_avg
                logger.info(f"MRR@{k} - Before: {before_avg:.6f}, After: {after_avg:.6f}, Improvement: {after_avg - before_avg:.6f}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any], dataset_name: str, loss_type: str, tag: str):
        """Save reranking results to file"""
        output_dir = f"preds/{dataset_name}/{loss_type}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"reranking_results_fold{tag}.json")
        with open(output_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = self._make_serializable(results)
            json.dump(results_serializable, f, indent=2)
        
        print(f"Saved reranking results to {output_file}")
    
    def train_reranker_only(self, rank_config, loss_type: str, dataset_name: str, 
                           tag: str, logger, device: str) -> Dict[str, Any]:
        """
        Train and evaluate reranker using existing predictions (skip ranker training)
        
        Args:
            rank_config: Config object (not used but kept for consistency)
            loss_type: Loss type used in prediction file paths
            dataset_name: Dataset name
            tag: Fold identifier
            logger: Logger instance
            device: Device to run on
        
        Returns:
            Dictionary with reranker stats and evaluation results
        """
        
        # Step 1: Load training predictions
        logger.info("=" * 80)
        logger.info("STEP 1: Loading training predictions for reranker")
        logger.info("=" * 80)
        
        train_predictions = self._load_train_predictions(rank_config, dataset_name, loss_type, tag)
        
        if not train_predictions:
            raise FileNotFoundError(
                f"No training predictions found at preds/{dataset_name}/{loss_type}/Fold{tag}_train.txt. "
                f"Please run the full pipeline first to generate predictions."
            )
        
        logger.info(f"Loaded {len(train_predictions)} training queries")
        
        # Auto-detect feature dimension from first query
        doc_feat_dim = 0
        if train_predictions and 'features' in train_predictions[0] and train_predictions[0]['features'] is not None:
            doc_feat_dim = train_predictions[0]['features'].shape[1]
            logger.info(f"Auto-detected document feature dimension: {doc_feat_dim}")
        
        # Step 2: Train reranker
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Training reranker")
        logger.info("=" * 80)
        
        reranker = RerankerTrainer(device=device, logger=logger, config=self.reranker_config, doc_feat_dim=doc_feat_dim)
        reranker_stats = reranker.train_on_predictions(train_predictions)
        logger.info(f"Reranker training stats: {reranker_stats}")
        
        # Step 3: Load test predictions
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Loading test predictions")
        logger.info("=" * 80)
        
        test_predictions = self._load_test_predictions(rank_config, dataset_name, loss_type, tag)
        
        if not test_predictions:
            raise FileNotFoundError(
                f"No test predictions found at preds/{dataset_name}/{loss_type}/Fold{tag}_test.txt. "
                f"Please run the full pipeline first to generate predictions."
            )
        
        logger.info(f"Loaded {len(test_predictions)} test queries")
        
        # Step 4: Evaluate on test set with reranking
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Evaluating on test set with reranking")
        logger.info("=" * 80)
        
        reranking_results = self._evaluate_with_reranking(
            test_predictions, reranker, logger, self.reranker_config
        )
        
        # Save results
        self._save_results(reranking_results, dataset_name, loss_type, tag)
        
        return {
            'reranker_stats': reranker_stats,
            'reranking_results': reranking_results
        }
    
    @staticmethod
    def _make_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, dict):
            return {k: ModularRerankerFramework._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModularRerankerFramework._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main entry point"""
    print("=" * 80)
    print("MODULAR RERANKER STARTING")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    parser = argparse.ArgumentParser(description='Modular LTR Training with Reranking')
    parser.add_argument('--config', required=True, help='Modular config file path')
    parser.add_argument('--ranker', help='Ranker type override (transformer, sbert)')
    parser.add_argument('--loss', help='Loss type override (ranknet, lrcl, neural, softndcg, neuralndcg)')
    parser.add_argument('--model', help='Model architecture', default=None)
    parser.add_argument('--fold', help='Fold id (1..N)', default=None)
    parser.add_argument('--wandb', help='Enable wandb logging', action='store_true')
    parser.add_argument('--wandb_project', help='WandB project name', default='modular-reranker')
    
    # Reranker-only mode
    parser.add_argument('--reranker-only', action='store_true', 
                        help='Skip ranker training and use existing predictions')
    
    # Reranker-specific parameters
    parser.add_argument('--reranker_d_model', type=int, default=256, help='Reranker d_model')
    parser.add_argument('--reranker_nhead', type=int, default=4, help='Reranker nhead')
    parser.add_argument('--reranker_num_layers', type=int, default=2, help='Reranker num_layers')
    parser.add_argument('--reranker_epochs', type=int, default=5, help='Reranker training epochs')
    parser.add_argument('--reranker_lr', type=float, default=1e-4, help='Reranker learning rate')
    
    args = parser.parse_args()
    print(f"Config file: {args.config}")
    print(f"Ranker: {args.ranker}, Loss: {args.loss}")
    
    if args.reranker_only:
        print("\n" + "=" * 80)
        print("RERANKER-ONLY MODE: Skipping ranker training, using existing predictions")
        print("=" * 80 + "\n")
    
    # Load config
    print("Loading modular config...")
    modular_config = load_modular_config(args.config)
    base_config_path = modular_config['base_config']
    print(f"Base config path (before resolution): {base_config_path}")
    
    # Resolve relative paths relative to config file directory
    config_dir = os.path.dirname(os.path.abspath(args.config))
    print(f"Config directory: {config_dir}")
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(config_dir, base_config_path)
    print(f"Base config path (after resolution): {base_config_path}")
    
    rank_job_dir = modular_config.get('dataset_name', 'MQ2008')
    ranker_type = args.ranker or modular_config.get('ranker_type', 'transformer')
    loss_type = args.loss or modular_config.get('loss_type', 'ranknet')
    cross_validation_config = modular_config.get('cross_validation', {})
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config_dict = json.load(f)
    
    # Extract dataset name from data.path if not explicitly set
    if 'dataset_name' not in modular_config and 'data' in base_config_dict and 'path' in base_config_dict['data']:
        data_path = base_config_dict['data']['path']
        # Extract dataset name from path like "dataset/MQ2008/Fold{fold}" -> "MQ2008"
        dataset_name = data_path.split('/')[1]
        rank_job_dir = dataset_name
    
    # Remove modular-specific keys that shouldn't go to base config
    modular_only_keys = ['ranker_type', 'loss_type', 'base_config', 'description', 'cross_validation', 'dataset_name']
    base_config_dict_clean = {k: v for k, v in base_config_dict.items() if k not in modular_only_keys}
    
    # Setup logging
    logger = Logger()
    
    # Update reranker config from args
    #RerankerConfig.TRANSFORMER_D_MODEL = args.reranker_d_model
    #RerankerConfig.TRANSFORMER_NHEAD = args.reranker_nhead
    #RerankerConfig.TRANSFORMER_NUM_LAYERS = args.reranker_num_layers
    #RerankerConfig.NUM_EPOCHS = args.reranker_epochs
    #RerankerConfig.LEARNING_RATE = args.reranker_lr
    
    # Initialize framework
    framework = ModularRerankerFramework(logger=logger)
    
    # Process folds
    if cross_validation_config and cross_validation_config.get('enabled', False):
        folds = cross_validation_config.get('folds', [1])
        seeds = cross_validation_config.get('seeds', [1])
        
        all_results = {}
        
        for fold in folds:
            print('\n' + '=' * 80)
            print(f'PROCESSING FOLD {fold}')
            print('=' * 80)
            
            temp_config_file = base_config_path.replace('.json', f'_temp_fold{fold}.json')
            with open(temp_config_file, 'w') as f:
                json.dump(base_config_dict_clean, f, indent=2)
            
            rank_run_id = 'run_1'
            fold_config = framework.base_framework.load_config(ranker_type, temp_config_file, rank_job_dir, rank_run_id)
            os.remove(temp_config_file)
            
            # Format fold placeholder in dataset path
            if hasattr(fold_config.data, 'path') and '{fold}' in fold_config.data.path:
                fold_config.data.path = fold_config.data.path.format(fold=fold)
            print(f'Dataset path: {fold_config.data.path}')
            
            fold_results = {}
            for seed in seeds:
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                
                print('\n' + '=' * 80)
                print(f'Fold: {fold}, Seed: {seed}')
                print('=' * 80)
                
                if args.reranker_only:
                    results = framework.train_reranker_only(
                        rank_config=fold_config,
                        loss_type=loss_type,
                        dataset_name=rank_job_dir,
                        tag=fold,
                        logger=logger,
                        device=device
                    )
                else:
                    results = framework.train_with_reranking(
                        ranker_type=ranker_type,
                        loss_type=loss_type,
                        rank_config=fold_config,
                        logger=logger,
                        device=device,
                        tag=fold,
                        loss_params={},
                        wandb_logger=None,
                        dataset_name=rank_job_dir
                    )
                fold_results[f'seed-{seed}'] = results
            
            all_results[f'fold-{fold}'] = fold_results
        
        # Print summary
        print('\n' + '=' * 80)
        print('SUMMARY RESULTS')
        print('=' * 80)
        for fold_key, fold_data in all_results.items():
            print(f'\n{fold_key}:')
            for seed_key, result in fold_data.items():
                print(f'  {seed_key}: {result}')
    
    # Single-fold mode
    else:
        print('\n' + '=' * 80)
        print('SINGLE-FOLD MODE')
        print('=' * 80)
        
        seeds = [1]
        
        temp_config_file = base_config_path.replace('.json', '_temp_single.json')
        with open(temp_config_file, 'w') as f:
            json.dump(base_config_dict_clean, f, indent=2)
        
        rank_run_id = 'run_1'
        rank_config = framework.base_framework.load_config(ranker_type, temp_config_file, rank_job_dir, rank_run_id)
        os.remove(temp_config_file)
        
        # Format fold placeholder in dataset path
        fold_id = args.fold or 1
        if hasattr(rank_config.data, 'path') and '{fold}' in rank_config.data.path:
            rank_config.data.path = rank_config.data.path.format(fold=fold_id)
        print(f"Dataset path: {rank_config.data.path}")
        
        results = {}
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            print('\n' + '=' * 80)
            print(f'Seed: {seed}')
            print('=' * 80)
            
            fold = args.fold or '1'
            
            if args.reranker_only:
                result = framework.train_reranker_only(
                    rank_config=rank_config,
                    loss_type=loss_type,
                    dataset_name=rank_job_dir,
                    tag=fold,
                    logger=logger,
                    device=device
                )
            else:
                result = framework.train_with_reranking(
                    ranker_type=ranker_type,
                    loss_type=loss_type,
                    rank_config=rank_config,
                    logger=logger,
                    device=device,
                    tag=fold,
                    loss_params={},
                    wandb_logger=None,
                    dataset_name=rank_job_dir
                )
            results[f'seed-{seed}'] = result
        
        print('\n' + '=' * 80)
        print('RESULTS')
        print('=' * 80)
        for seed, result in results.items():
            print(f'{seed}: {result}')


if __name__ == '__main__':
    main()
