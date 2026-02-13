# Query-Adaptive Listwise Reranking (RankRoute)

## Overview

This repository implements the framework proposed in our paper on
**query-adaptive listwise reranking**. Instead of deploying a single
loss-trained ranker globally, we reframe inference as a **per-query
selection problem** over a finite candidate set of complete ranked
lists.

For each query: 1. A **base ranker** is trained using the modular
framework. 2. A **candidate set of ranked lists** is constructed. 3. A
**Transformer-based selector** evaluates each query--list pair. 4. The
selector chooses the ranking expected to maximize a target metric (e.g.,
NDCG or MRR). 5. Metrics are reported **before and after selection**.

This framework supports:

-   **Intra-model reranking**: structured perturbations of a single
    baseline ranking.
-   **Inter-model routing**: selection among outputs of multiple rankers
    trained with different surrogate losses.

The method separates two concepts emphasized in the paper:

-   **Coverage**: whether a better ranking exists in the candidate set.
-   **Selection**: whether the learned selector can identify that
    ranking.

------------------------------------------------------------------------

## Key Differences from Base Training

### Training Phase

-   **Base ranker**:
    -   Trained normally using the specified surrogate loss.
    -   Produces predicted scores and induced ranked lists per query.
-   **Selector training**:
    -   Takes candidate ranked lists as input.
    -   Learns to approximate the candidate-oracle (the best ranking
        within the candidate set under the target metric).
    -   Operates over complete ranked lists.
    -   Uses inference-time features derived from ranker scores and
        available query/document signals.

### Test Phase

For each query:

-   **Original ranking**: Produced by the baseline ranker.
-   **Selected ranking**: Chosen by the trained selector from the
    candidate set.
-   **Metrics computed**:
    -   NDCG@5
    -   NDCG@10
    -   MRR@5
    -   MRR@10

Performance is evaluated using per-query paired comparisons as described
in the paper.

------------------------------------------------------------------------

## Architecture

### RerankerConfig

The selector is implemented as a Transformer encoder operating over
document sequences induced by candidate rankings.

Hyperparameters (unchanged from the original README):

``` python
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 256
TRANSFORMER_DROPOUT = 0.0
NUM_CANDIDATES = 10
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
```

The Transformer produces a scalar score per candidate list.\
The highest-scoring candidate is selected per query.

------------------------------------------------------------------------

## Core Components

### RerankerTrainer

Responsible for selector training and inference:

-   `train_on_predictions()` --- trains the selector on generated
    candidate lists.
-   `rerank_predictions()` --- applies selection at inference time.

### ModularRerankerFramework

Coordinates the end-to-end pipeline:

-   `train_with_reranking()` --- trains baseline + selector.
-   `_evaluate_with_reranking()` --- computes metrics before/after
    selection.
-   `_save_results()` --- serializes results to JSON.

------------------------------------------------------------------------

## Usage

### Basic Usage

``` bash
python modular_reranker.py --config configs/modular_config.json --ranker transformer --loss ranknet
```

### With Custom Selector Parameters

``` bash
python modular_reranker.py     --config configs/modular_config.json     --ranker transformer     --loss ranknet     --reranker_d_model 256     --reranker_nhead 4     --reranker_num_layers 2     --reranker_epochs 5     --reranker_lr 1e-4
```

### Cross-Validation

Enable in config file:

``` json
{
    "cross_validation": {
        "enabled": true,
        "folds": [1, 2, 3, 4, 5],
        "seeds": [1]
    }
}
```

------------------------------------------------------------------------

## Output Format

Results are saved to:

    preds/{dataset}/{loss_type}/reranking_results_fold{N}.json

Example structure:

``` json
{
    "queries_evaluated": 157,
    "metrics_before": {
        "ndcg@5": [],
        "ndcg@10": [],
        "mrr@5": [],
        "mrr@10": []
    },
    "metrics_after": {
        "ndcg@5": [],
        "ndcg@10": [],
        "mrr@5": [],
        "mrr@10": []
    },
    "summary": {
        "ndcg@5_before": 0.5500,
        "ndcg@5_after": 0.5700,
        "ndcg@5_improvement": 0.0200
    }
}
```

------------------------------------------------------------------------

## Integration with Existing Pipeline

This module builds upon:

-   `modular_train_allrank.py` --- baseline ranker training
-   `reranker_transformer_loss.py` --- selector architecture and loss
-   `ranker.py` --- base ranking models

------------------------------------------------------------------------

## Notes

-   The selector operates at the level of complete ranked lists.
-   Baseline containment is enforced in candidate construction.
-   The framework supports both structured perturbation (intra-model)
    and multi-ranker routing (inter-model).
-   Statistical testing follows the methodology described in the paper.
