# from allrank.utils.ltr_logging import get_logger

# logger = get_logger()
import torch


class EarlyStop:
    def __init__(self, patience, logger, tag=''):
        self.patience = patience
        self.best_value = 0.0
        self.best_epoch = 0
        self.logger = logger
        self.tag = tag

    def step(self, current_value, current_epoch, model):
        self.logger.info("Current: {:.8}   Best: {:.8}".format(current_value, self.best_value))
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_epoch = current_epoch

    def stop_training(self, current_epoch) -> bool:
        return current_epoch - self.best_epoch > self.patience

    def get_best_model(self):
        return None
