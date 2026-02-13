# Reranker-Only Mode Usage

## Overview
The `--reranker-only` flag allows you to skip ranker training and train/test only the reranker using existing prediction files from previous runs.

## Requirements
Before using `--reranker-only`, you must have already run the full pipeline at least once to generate:
- `preds/{dataset}/{loss_type}/Fold{N}_train.txt` - Training predictions
- `preds/{dataset}/{loss_type}/Fold{N}_test.txt` - Test predictions

## Usage

### Basic Command
```bash
python modular_reranker.py \
    --config configs/softndcg_transformer_MQ2008.json \
    --loss listnet \
    --fold 1 \
    --reranker-only
```

### With Custom Reranker Configuration
```bash
python modular_reranker.py \
    --config configs/softndcg_transformer_MQ2008.json \
    --loss listnet \
    --fold 1 \
    --reranker-only \
    --reranker_d_model 512 \
    --reranker_nhead 8 \
    --reranker_num_layers 4 \
    --reranker_epochs 20 \
    --reranker_lr 1e-5
```

### SLURM Script Example
```bash
#!/bin/bash
#SBATCH --job-name=reranker-only
#SBATCH --output=slurm-mod-rerank/reranker-%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

python modular_reranker.py \
    --config configs/softndcg_transformer_MQ2008.json \
    --loss listnet \
    --fold 1 \
    --reranker-only
```

## What Happens in Reranker-Only Mode

1. **STEP 1**: Load training predictions from `preds/{dataset}/{loss_type}/Fold{N}_train.txt`
2. **STEP 2**: Train reranker on loaded predictions
3. **STEP 3**: Load test predictions from `preds/{dataset}/{loss_type}/Fold{N}_test.txt`
4. **STEP 4**: Evaluate reranker on test set (before/after metrics)
5. Save results to `preds/{dataset}/{loss_type}/reranking_results_fold{N}.json`

## Benefits

- **Fast iteration**: Experiment with different reranker architectures without retraining the base ranker
- **Hyperparameter tuning**: Quickly test different learning rates, model sizes, etc.
- **Resource efficient**: No need to retrain the expensive base ranker
- **Reproducibility**: Use the exact same base predictions across experiments

## Configuration in Code

The reranker configuration is in `modular_reranker.py` (lines 46-64):

```python
class RerankerConfig:
    # Architecture
    TRANSFORMER_D_MODEL = 32          # Current: small/fast
    TRANSFORMER_NHEAD = 2
    TRANSFORMER_NUM_LAYERS = 1
    TRANSFORMER_DIM_FEEDFORWARD = 32
    TRANSFORMER_DROPOUT = 0.0
    
    # Training
    NUM_CANDIDATES = 10
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    SEED = 42
```

## Example Workflow

1. **First run** (full pipeline):
   ```bash
   python modular_reranker.py --config configs/softndcg_transformer_MQ2008.json --loss listnet --fold 1
   ```
   This generates predictions in `preds/MQ2008/listnet/`

2. **Subsequent runs** (reranker-only):
   ```bash
   # Try different architectures
   python modular_reranker.py --config configs/softndcg_transformer_MQ2008.json --loss listnet --fold 1 --reranker-only --reranker_d_model 64
   
   # Try different learning rates
   python modular_reranker.py --config configs/softndcg_transformer_MQ2008.json --loss listnet --fold 1 --reranker-only --reranker_lr 1e-4
   
   # Try more epochs
   python modular_reranker.py --config configs/softndcg_transformer_MQ2008.json --loss listnet --fold 1 --reranker-only --reranker_epochs 20
   ```

## Error Handling

If prediction files don't exist, you'll get a clear error message:
```
FileNotFoundError: No training predictions found at preds/MQ2008/listnet/Fold1_train.txt. 
Please run the full pipeline first to generate predictions.
```
