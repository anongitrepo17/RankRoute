# Modular Reranker

## Overview

`modular_reranker.py` extends `modular_train_allrank.py` with an integrated reranking pipeline. It:

1. **Trains a base ranker** using the modular framework (transformer, loss function, etc.)
2. **Trains a transformer-based reranker** on the base ranker's predictions
3. **Evaluates on test set** with metrics before and after reranking
4. **Saves results** to `preds/{dataset}/{loss_type}/reranking_results_fold{N}.json`

## Key Differences from Base Training

### Training Phase
- **Base ranker**: Trained normally using specified loss function
- **Reranker training**: 
  - Takes base ranker predictions as input
  - Constructs ideal rankings (sorted by labels) + N-1 random shuffles
  - Trains transformer to identify ideal rankings
  - Uses only base scores (not document features)

### Test Phase
- **Original ranking**: From base ranker predictions
- **Reranked ranking**: Selected by trained reranker from candidates
- **Metrics**: NDCG@5, NDCG@10, MRR@5, MRR@10 before and after reranking

## Architecture

### RerankerConfig
Centralized configuration for reranker hyperparameters:
```python
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 256
TRANSFORMER_DROPOUT = 0.0
NUM_CANDIDATES = 10
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
```

### RerankerTrainer
Handles reranker training and inference:
- `train_on_predictions()`: Train on captured predictions
- `rerank_predictions()`: Apply reranking to test predictions

### ModularRerankerFramework
Orchestrates the full pipeline:
- `train_with_reranking()`: Main training method
- `_evaluate_with_reranking()`: Compute metrics before/after
- `_save_results()`: Save results to JSON

## Usage

### Basic Usage
```bash
python modular_reranker.py --config configs/modular_config.json --ranker transformer --loss ranknet
```

### With Custom Reranker Parameters
```bash
python modular_reranker.py \
    --config configs/modular_config.json \
    --ranker transformer \
    --loss ranknet \
    --reranker_d_model 256 \
    --reranker_nhead 4 \
    --reranker_num_layers 2 \
    --reranker_epochs 5 \
    --reranker_lr 1e-4
```

### Cross-Validation
Enable in config file:
```json
{
    "cross_validation": {
        "enabled": true,
        "folds": [1, 2, 3, 4, 5],
        "seeds": [1]
    }
}
```

## Output Format

Results saved to `preds/{dataset}/{loss_type}/reranking_results_fold{N}.json`:

```json
{
    "queries_evaluated": 157,
    "metrics_before": {
        "ndcg@5": [0.5, 0.6, ...],
        "ndcg@10": [0.7, 0.8, ...],
        "mrr@5": [0.8, 0.9, ...],
        "mrr@10": [0.85, 0.95, ...]
    },
    "metrics_after": {
        "ndcg@5": [0.52, 0.62, ...],
        "ndcg@10": [0.72, 0.82, ...],
        "mrr@5": [0.82, 0.92, ...],
        "mrr@10": [0.87, 0.97, ...]
    },
    "summary": {
        "ndcg@5_before": 0.5500,
        "ndcg@5_after": 0.5700,
        "ndcg@5_improvement": 0.0200,
        "ndcg@10_before": 0.7500,
        "ndcg@10_after": 0.7700,
        "ndcg@10_improvement": 0.0200,
        "mrr@5_before": 0.8200,
        "mrr@5_after": 0.8400,
        "mrr@5_improvement": 0.0200,
        "mrr@10_before": 0.8700,
        "mrr@10_after": 0.8900,
        "mrr@10_improvement": 0.0200
    }
}
```

## Integration with Existing Pipeline

The script reuses components from:
- `modular_train_allrank.py`: Base training framework
- `reranker_transformer_loss.py`: Transformer evaluator and loss
- `ranker.py`: Base ranker implementation

## Future Enhancements

1. **Capture training predictions**: Modify ranker to save predictions during training
2. **Validation set reranker training**: Use validation predictions for reranker training
3. **Multiple reranker architectures**: Support MLP, attention-based alternatives
4. **Hyperparameter tuning**: Grid search over reranker parameters
5. **Ensemble reranking**: Combine multiple reranker predictions

## Notes

- Reranker trains on base scores only (not document features) for efficiency
- Ideal ranking always placed as first candidate during training
- Test-time reranking generates original ranking + N-1 random shuffles
- Results automatically saved with proper JSON serialization
