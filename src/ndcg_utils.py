"""
Unified NDCG Calculation Utilities

This module provides a single, consistent NDCG@k implementation used across
the entire pipeline (training, evaluation, reranking).

Key principle: IDCG must be computed from ALL available labels, not just
the labels in the current ranking. This ensures NDCG is always in [0, 1].
"""

import numpy as np


def compute_ndcg(labels_ranked: np.ndarray, labels_all: np.ndarray = None, k: int = 10) -> float:
    """
    Compute NDCG@k using the standard gain function: 2^rel - 1
    
    This is the ONLY NDCG function that should be used throughout the codebase.
    
    Args:
        labels_ranked: Relevance labels in ranking order (the ranking to evaluate)
        labels_all: ALL available relevance labels for computing IDCG. 
                   If None, uses labels_ranked (assumes you're passing all labels)
        k: Cutoff position for NDCG@k
    
    Returns:
        NDCG@k value in range [0, 1]
    
    Examples:
        # Case 1: Evaluating a full ranking
        >>> labels = np.array([2, 1, 0, 1, 0])
        >>> compute_ndcg(labels, k=5)  # labels_all defaults to labels_ranked
        
        # Case 2: Evaluating a partial ranking (e.g., top-10 from 50 docs)
        >>> all_labels = np.array([...])  # 50 labels
        >>> ranking_indices = np.argsort(-scores)[:10]
        >>> labels_ranked = all_labels[ranking_indices]
        >>> compute_ndcg(labels_ranked, labels_all=all_labels, k=10)
    
    Note:
        - Gain function: 2^rel - 1 (consistent with allrank)
        - Discount: 1/log2(i+2) for position i (0-indexed)
        - IDCG is computed from ALL labels to ensure NDCG <= 1.0
        - Returns 1.0 if IDCG is 0 (no relevant documents)
    """
    if labels_all is None:
        labels_all = labels_ranked
    
    # Gain function: 2^rel - 1 (same as allrank)
    def gain(rel):
        return np.power(2.0, rel) - 1.0
    
    # Compute DCG@k from the ranking
    dcg = 0.0
    for i in range(min(k, len(labels_ranked))):
        rel = labels_ranked[i]
        dcg += gain(rel) / np.log2(i + 2)
    
    # Compute IDCG@k from ALL labels (ideal ranking)
    # This is critical: we must use labels_all, not labels_ranked
    ideal = np.sort(labels_all)[::-1]
    idcg = 0.0
    for i in range(min(k, len(ideal))):
        rel = ideal[i]
        idcg += gain(rel) / np.log2(i + 2)
    
    # Handle zero IDCG (no relevant documents)
    if idcg == 0.0:
        return 1.0
    
    ndcg = dcg / idcg
    
    # Sanity check: NDCG should always be in [0, 1]
    if ndcg > 1.0 + 1e-6:  # Allow small numerical errors
        # Debug information
        import sys
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"NDCG > 1.0 ERROR DETAILS:", file=sys.stderr)
        print(f"  NDCG = {ndcg:.6f}", file=sys.stderr)
        print(f"  DCG = {dcg:.6f}, IDCG = {idcg:.6f}", file=sys.stderr)
        print(f"  k = {k}", file=sys.stderr)
        print(f"  len(labels_ranked) = {len(labels_ranked)}", file=sys.stderr)
        print(f"  len(labels_all) = {len(labels_all)}", file=sys.stderr)
        print(f"  labels_ranked[:k] = {labels_ranked[:min(k, len(labels_ranked))]}", file=sys.stderr)
        print(f"  labels_all (sorted) = {np.sort(labels_all)[::-1][:min(k, len(labels_all))]}", file=sys.stderr)
        print(f"  Are they the same set? {sorted(labels_ranked) == sorted(labels_all)}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        raise ValueError(
            f"NDCG={ndcg:.6f} > 1.0! This indicates a bug in IDCG calculation. "
            f"DCG={dcg:.6f}, IDCG={idcg:.6f}, "
            f"len(labels_ranked)={len(labels_ranked)}, len(labels_all)={len(labels_all)}"
        )
    
    return min(ndcg, 1.0)  # Clamp to [0, 1] to handle numerical errors
