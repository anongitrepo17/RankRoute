import torch
import os
from .training.train_utils import compute_metrics
from .data.dataset_loading import PADDED_Y_VALUE
# from .utils.ltr_logging import get_logger

# logger = get_logger()

def save_predictions(model, data_dl, device, logger, dataset_name, fold, loss_type, split='train'):
    """
    Save model predictions for a given data split (train/val/test)
    
    Args:
        model: trained model
        data_dl: dataloader for the split
        device: device to run on
        logger: logger instance
        dataset_name: name of dataset (e.g., 'MQ2008')
        fold: fold number
        loss_type: loss function name (e.g., 'listnet')
        split: 'train', 'val', or 'test'
    """
    model.eval()
    
    # Collect predictions, labels, and features
    all_predictions = []
    all_labels = []
    all_features = []
    all_qids = []
    batch_idx = 0
    
    with torch.no_grad():
        for xb, yb, indices in data_dl:
            mask = (yb == PADDED_Y_VALUE)
            pred_b = model.score(xb.to(device=device), mask.to(device=device), indices.to(device=device))
            
            # Store predictions, labels, and features
            all_predictions.append(pred_b.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
            all_features.append(xb.cpu().numpy())
            
            # Get query IDs
            batch_size = xb.shape[0]
            for i in range(batch_size):
                qid_idx = batch_idx + i
                if qid_idx < len(data_dl.dataset.query_ids):
                    all_qids.append(int(data_dl.dataset.query_ids[qid_idx]))
            batch_idx += batch_size
    
    # Save to file
    import numpy as np
    
    # Create output directory
    output_dir = os.path.join('preds', dataset_name, loss_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file path
    output_file = os.path.join(output_dir, f'Fold{fold}_{split}.txt')
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_features = np.concatenate(all_features, axis=0)
    
    # Write to file
    with open(output_file, 'w') as f:
        for i in range(len(all_qids)):
            qid = all_qids[i]
            labels = all_labels[i]
            preds = all_predictions[i]
            features = all_features[i]
            
            # Filter out padded values
            valid_mask = labels != PADDED_Y_VALUE
            valid_labels = labels[valid_mask]
            valid_preds = preds[valid_mask]
            valid_features = features[valid_mask]
            
            # Write query ID
            f.write(f"qid:{qid}\n")
            
            # Write labels
            f.write("labels: [" + " ".join([str(label) for label in valid_labels]) + "]\n")
            
            # Write predictions
            f.write("predictions: [" + " ".join([str(pred) for pred in valid_preds]) + "]\n")
            
            # Write features (one row per document)
            f.write("features:\n")
            for doc_feat in valid_features:
                f.write("  [" + " ".join([str(feat) for feat in doc_feat]) + "]\n")
            
            # Add blank line
            f.write("\n")
    
    logger.info(f'Saved {split} predictions to {output_file}')


def do_test(model, test_dl, config, device, logger, dataset_name=None, fold=None, loss_type=None):
    model.eval()
    
    # Collect predictions and labels for saving
    all_predictions = []
    all_labels = []
    all_qids = []
    batch_idx = 0
    
    with torch.no_grad():
        logger.info('testing ...')
        
        # Collect predictions and labels
        for xb, yb, indices in test_dl:
            mask = (yb == PADDED_Y_VALUE)
            pred_b = model.score(xb.to(device=device), mask.to(device=device), indices.to(device=device))
            
            # Store predictions and labels (move to CPU and convert to numpy)
            all_predictions.append(pred_b.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
            
            # Get actual query IDs from the dataset
            batch_size = xb.shape[0]
            for i in range(batch_size):
                qid_idx = batch_idx + i
                if qid_idx < len(test_dl.dataset.query_ids):
                    all_qids.append(int(test_dl.dataset.query_ids[qid_idx]))
            batch_idx += batch_size
        
        # Compute metrics
        test_metrics = compute_metrics(config.test_metrics, model, test_dl, device)

    test_summary = 'Test:'
    for metric_name, metric_value in test_metrics.items():
        test_summary += " {metric_name} {metric_value:.8} ".format(
            metric_name=metric_name, metric_value=metric_value)
    logger.info('\033[94m' + test_summary + '\033[0m')

    # Log test metrics to wandb if available
    if hasattr(logger, 'wandb_logger') and logger.wandb_logger:
        logger.wandb_logger.log_metrics(test_metrics, 0, "test")
    
    # Save predictions and labels if dataset_name and fold are provided
    if dataset_name is not None and fold is not None:
        import numpy as np
        
        # Create output directory - include loss_type if provided
        if loss_type is not None:
            output_dir = os.path.join('preds', dataset_name, loss_type)
        else:
            output_dir = os.path.join('preds', dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path
        output_file = os.path.join(output_dir, f'Fold{fold}_test.txt')
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Write to file
        with open(output_file, 'w') as f:
            for i in range(len(all_qids)):
                qid = all_qids[i]
                labels = all_labels[i]
                preds = all_predictions[i]
                
                # Filter out padded values (-1.0)
                valid_mask = labels != PADDED_Y_VALUE
                valid_labels = labels[valid_mask]
                valid_preds = preds[valid_mask]
                
                # Write query ID
                f.write(f"qid:{qid}\n")
                
                # Write labels (ground truth) in brackets
                f.write("labels: [" + " ".join([str(label) for label in valid_labels]) + "]\n")
                
                # Write predictions in brackets
                f.write("predictions: [" + " ".join([str(pred) for pred in valid_preds]) + "]\n")
                
                # Add blank line for readability
                f.write("\n")
        
        logger.info(f'Saved predictions to {output_file}')
    
    return {"test_metrics": test_metrics}
