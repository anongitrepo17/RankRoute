import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    data_size = 0
    with torch.no_grad():
        for batch in dataloader:
            scores, ranks, masks, targets = [x.to(device) for x in batch]
            outputs = model(scores, ranks, masks)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * scores.shape[0]
            data_size += scores.shape[0]
    return total_loss / data_size

def train_loss_model(config, model, train_dl, val_dl, test_dl, device, logger, tag):
    logger.info(f'target metric: {config.metric}')
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Loss function
    def loss_fn(preds, targets):
        return F.mse_loss(preds, targets)

    # Early stopping & best model tracking
    best_val_loss = float('inf')
    best_state = None
    patience = 5
    wait = 0
    ckpt_path = f'best_loss_model_{tag}.pth'

    for epoch in range(1, config.epochs + 1):
        logger.epoch = epoch
        model.train()
        train_loss = 0.0

        data_size = 0
        for batch in train_dl:
            scores, ranks, masks, targets = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(scores, ranks, masks)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * scores.shape[0]
            data_size += scores.shape[0]

        scheduler.step()

        # validation stage
        val_loss = evaluate(model, val_dl, device, loss_fn)

        logger.info(f"Train Loss: {train_loss/data_size:.10f} | Val Loss: {val_loss:.10f}")

        # Early stopping & checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
    # load best model
    logger.info(f"Best checkpoint: {ckpt_path} with val loss = {best_val_loss:.10f}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    logger.epoch = None

    # testing stage
    if test_dl:
        logger.stage = "Test"
        test_loss = evaluate(model, test_dl, device, loss_fn)
        logger.info(f"Test Loss: {test_loss:.10f}")
        logger.stage = None

    return model
