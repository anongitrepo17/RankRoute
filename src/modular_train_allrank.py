"""
Modular Training Framework for Learning-to-Rank
Allows configurable combination of rankers and loss functions.

This framework supports:
- Multiple ranker types: transformer-based (ranker.py) and SBERT-based (sbert_ranker.py)
- Multiple loss types: RankNet, LRCL (blackbox_loss.py), and custom losses
- Cross-validation and single-fold training
- WandB logging compatible with existing scripts
- JSON configuration for easy experimentation

Usage:
    python modular_train.py --config configs/modular_config.json --ranker transformer --loss ranknet
    python modular_train.py --config configs/modular_config.json --ranker sbert --loss lrcl --wandb
"""

import random
import numpy as np
import torch
import torch.nn as nn
import logging
import sys
import argparse
import os
import json
from os import path
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

# Import existing components
from ranker import Ranker, load_config as load_transformer_config

# Import allrank loss functions
from allrank.models.losses.listMLE import listMLE
from allrank.models.losses.listNet import listNet
from allrank.models.losses.pointwise import pointwise_rmse
from allrank.models.losses.approxNDCG import approxNDCGLoss
from allrank.models.losses.lambdaLoss import lambdaLoss

# Import neural loss components
from neural_loss import train_loss_model, NeuralRankNDCG, NeuralRankRecall, LossModelConfig
from neural_loss import load_dataset, RankDataset, RankDataLoader, Model
from copy import deepcopy

# Import neuralNDCG loss
from allrank.models.losses.neuralNDCG import neuralNDCG, neuralNDCG_transposed

# Import dataset loading utilities for test slate length override
from allrank.data.dataset_loading import load_libsvm_role
from torchvision import transforms
from allrank.data.dataset_loading import FixLength, ToTensor


def load_test_dataset_with_slate_length(input_path: str, slate_length: int):
    """
    Load test dataset and apply the training slate_length to it.
    This ensures test data uses the same slate length as training/validation.
    
    :param input_path: directory containing the LibSVM files
    :param slate_length: target slate length to apply to test dataset
    :return: loaded test dataset with FixLength transform applied
    """
    test_ds = load_libsvm_role(input_path, 'test')
    test_ds.transform = transforms.Compose([FixLength(slate_length), ToTensor()])
    return test_ds


def _patch_ranker_for_test_slate_length():
    """
    Monkey-patch the Ranker class to apply training slate_length to test dataset.
    This ensures test data uses the same slate length as training/validation.
    """
    from ranker import Ranker
    from allrank.data.dataset_loading import load_libsvm_dataset_role as original_load_libsvm_dataset_role
    
    original_init = Ranker.__init__
    
    def patched_init(self, config, alter_model=None, device='cpu', logger=None):
        # Call original init
        original_init(self, config, alter_model, device, logger)
        
        # Override test dataset loading to use training slate_length
        if self.config.test_metrics is not None and config.data.ds_type != 'npz':
            test_ds = load_test_dataset_with_slate_length(config.data.path, config.data.slate_length)
            from allrank.data.dataset_loading import create_test_dataloader
            self.test_dl = create_test_dataloader(test_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)
    
    Ranker.__init__ = patched_init


_patch_ranker_for_test_slate_length()


class Logger:
    """Console logger compatible with existing scripts"""
    def __init__(self):
        log_format = "[%(levelname)s] %(asctime)s - %(message)s"
        log_dateformat = "%H:%M:%S"
        logging.basicConfig(format=log_format, datefmt=log_dateformat,
                            stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.episode = None
        self.phase = None
        self.stage = None
        self.epoch = None

    def info(self, message):
        msg = ''
        if self.epoch:
            msg = f"Epoch {self.epoch:<2}"
        if self.stage:
            msg = f"{self.stage:<5}, {msg}" if msg else f"{self.stage:<5}"
        if self.phase:
            msg = f"{self.phase:<10}, {msg}" if msg else f"{self.phase:<10}"
        self.logger.info(f"{msg}  --  {message}" if msg else message)


class BaseRankerFactory(ABC):
    """Abstract factory for creating rankers"""
    
    @abstractmethod
    def create_ranker(self, config, device, logger):
        """Create a ranker instance"""
        pass
    
    @abstractmethod
    def load_config(self, config_file, job_dir, run_id):
        """Load configuration for this ranker type"""
        pass


class TransformerRankerFactory(BaseRankerFactory):
    """Factory for transformer-based rankers"""
    
    def __init__(self, alter_model=None):
        self.alter_model = alter_model
    
    def create_ranker(self, config, device, logger):
        return Ranker(config, alter_model=self.alter_model, device=device, logger=logger)
    
    def load_config(self, config_file, job_dir, run_id):
        return load_transformer_config(config_file, job_dir, run_id)

class BaseLossFactory(ABC):
    """Abstract factory for creating loss functions"""
    
    @abstractmethod
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        """Create a loss function instance"""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this loss type"""
        pass


class RankNetLossFactory(BaseLossFactory):
    """Factory for RankNet loss"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        loss_func = RankNetLoss(sigma=params['sigma'])
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {'sigma': 1.0}


class CustomizableLRCLLoss(nn.Module):
    """Custom LRCL loss that allows parameter customization without modifying blackbox_loss.py"""
    
    def __init__(self, use_gap_weight=True, pair_subsample=800, random_state=0,
                 margin_num_bases=8, penalty_num_bases=8, weight_num_bases=6,
                 weight_floor=1e-3, slope_range=(0.5, 4.0), bias_range=(-2.0, 2.0)):
        super().__init__()
        
        # Import here to avoid circular imports
        from blackbox_loss import LearnedMargin, LearnedPenalty, LearnedGapWeight, MonotoneBasis
        
        # Create custom margin function
        self.tau = LearnedMargin(num_bases=margin_num_bases)
        # Override the basis with custom parameters
        self.tau.basis = MonotoneBasis(num_bases=margin_num_bases, kind="softplus",
                                       slope_range=slope_range, bias_range=bias_range)
        
        # Create custom penalty function  
        self.g = LearnedPenalty(num_bases=penalty_num_bases)
        # Override the basis with custom parameters
        self.g.basis = MonotoneBasis(num_bases=penalty_num_bases, kind="softplus",
                                     slope_range=slope_range, bias_range=bias_range)
        
        # Create custom gap weight function if needed
        if use_gap_weight:
            self.w = LearnedGapWeight(num_bases=weight_num_bases, floor=weight_floor)
            # Override the basis with custom parameters
            self.w.basis = MonotoneBasis(num_bases=weight_num_bases, kind="sigmoid",
                                         slope_range=slope_range, bias_range=bias_range)
        else:
            self.w = None
            
        self.pair_subsample = pair_subsample
        self._seed = int(random_state)
        self._gen = None
        
    def _get_generator(self, device):
        # (Re)create generator on the correct device with a stable seed
        if (self._gen is None) or (self._gen.device != torch.device(device)):
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self._seed)
        return self._gen

    @staticmethod
    def _normalize_scores(scores, mask, eps=1e-6):
        # zero-mean, unit-std per query over valid items
        scores = scores.clone()
        scores = scores.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean = scores.sum(dim=1, keepdim=True) / denom
        scores = scores - mean
        var = ((scores ** 2) * mask).sum(dim=1, keepdim=True) / denom
        std = (var + eps).sqrt()
        return scores / std

    def forward(self, predictions, targets):
        # infer mask: targets == -1 => invalid (common in LTR datasets)
        if targets.dtype.is_floating_point:
            mask = ~torch.isclose(targets, torch.tensor(-1.0, device=targets.device))
        else:
            mask = targets != -1
        targets_f = targets.float()
        
        # Use the same logic as LearnedPairwiseLoss.forward
        device = predictions.device
        B, N = predictions.shape
        s = self._normalize_scores(predictions, mask)
        r = targets_f

        losses = []
        for b in range(B):
            valid = mask[b]  # (N,)
            idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            nb = idx.numel()
            if nb <= 1:
                continue

            sb = s[b, idx]     # (nb,)
            rb = r[b, idx]     # (nb,)

            i_idx, j_idx = torch.triu_indices(nb, nb, offset=1, device=device)
            dr = rb[i_idx] - rb[j_idx]         # (P,)
            t = torch.sign(dr)                 # {-1,0,1}
            keep = t != 0
            if keep.sum() == 0:
                continue
            i_idx = i_idx[keep]; j_idx = j_idx[keep]
            dr = dr[keep]; t = t[keep]

            if self.pair_subsample is not None and self.pair_subsample < i_idx.numel():
                gen = self._get_generator(device)
                perm = torch.randperm(i_idx.numel(), generator=gen, device=device)[:self.pair_subsample]
                i_idx = i_idx[perm]; j_idx = j_idx[perm]
                dr = dr[perm]; t = t[perm]

            ds = sb[i_idx] - sb[j_idx]        # (P,)
            abs_dr = dr.abs()

            tau = self.tau(abs_dr)            # required margin
            m = tau - t * ds                  # violation
            w = self.w(abs_dr) if self.w is not None else 1.0
            pair_loss = self.g(m) * w
            losses.append(pair_loss.mean())

        if len(losses) == 0:
            return torch.zeros((), device=predictions.device, requires_grad=True)

        return torch.stack(losses).mean()


class LRCLLossFactory(BaseLossFactory):
    """Factory for LRCL (Learned Ranking with Contrastive Loss) loss"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        loss_func = CustomizableLRCLLoss(
            use_gap_weight=params['use_gap_weight'],
            pair_subsample=params['pair_subsample'],
            random_state=params['random_state'],
            margin_num_bases=params['margin_num_bases'],
            penalty_num_bases=params['penalty_num_bases'],
            weight_num_bases=params['weight_num_bases'],
            weight_floor=params['weight_floor'],
            slope_range=params['slope_range'],
            bias_range=params['bias_range']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'use_gap_weight': True,
            'pair_subsample': None,
            'random_state': 0,
            'margin_num_bases': 8,
            'penalty_num_bases': 8,
            'weight_num_bases': 6,
            'weight_floor': 1e-3,
            'slope_range': (0.5, 4.0),
            'bias_range': (-2.0, 2.0)
        }


class SoftNDCGLossFactory(BaseLossFactory):
    """Factory for SoftNDCG and Learnable loss functions"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        loss_func = SoftNDCGLoss(
                temperature=params['temperature'],
                gain_base=params['gain_base'],
                discount_base=params['discount_base'],
                sinkhorn_iters=params['sinkhorn_iters']
            )
        
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'loss_type': 'softndcg',
            'temperature': 0.5,
            'gain_base': 2.0,
            'discount_base': 2.0,
            'sinkhorn_iters': 20
        }


class ListMLELossFactory(BaseLossFactory):
    """Factory for ListMLE loss function"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        class ListMLEWrapper(nn.Module):
            def __init__(self, eps, padded_value_indicator):
                super().__init__()
                self.eps = eps
                self.padded_value_indicator = padded_value_indicator
            
            def forward(self, y_pred, y_true):
                return listMLE(y_pred, y_true, self.eps, self.padded_value_indicator)
        
        loss_func = ListMLEWrapper(
            eps=params['eps'],
            padded_value_indicator=params['padded_value_indicator']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'eps': 1e-10,
            'padded_value_indicator': -1
        }


class ListNetLossFactory(BaseLossFactory):
    """Factory for ListNet loss function"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        class ListNetWrapper(nn.Module):
            def __init__(self, eps, padded_value_indicator):
                super().__init__()
                self.eps = eps
                self.padded_value_indicator = padded_value_indicator
            
            def forward(self, y_pred, y_true):
                return listNet(y_pred, y_true, self.eps, self.padded_value_indicator)
        
        loss_func = ListNetWrapper(
            eps=params['eps'],
            padded_value_indicator=params['padded_value_indicator']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'eps': 1e-8,
            'padded_value_indicator': -1
        }


class RMSELossFactory(BaseLossFactory):
    """Factory for RMSE (pointwise) loss function"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        class RMSEWrapper(nn.Module):
            def __init__(self, no_of_levels, padded_value_indicator):
                super().__init__()
                self.no_of_levels = no_of_levels
                self.padded_value_indicator = padded_value_indicator
            
            def forward(self, y_pred, y_true):
                return pointwise_rmse(y_pred, y_true, self.no_of_levels, self.padded_value_indicator)
        
        loss_func = RMSEWrapper(
            no_of_levels=params['no_of_levels'],
            padded_value_indicator=params['padded_value_indicator']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'no_of_levels': 5,  # Default for typical relevance levels (0-4)
            'padded_value_indicator': -1
        }


class ApproxNDCGLossFactory(BaseLossFactory):
    """Factory for ApproxNDCG loss function"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        class ApproxNDCGWrapper(nn.Module):
            def __init__(self, eps, padded_value_indicator, alpha):
                super().__init__()
                self.eps = eps
                self.padded_value_indicator = padded_value_indicator
                self.alpha = alpha
            
            def forward(self, y_pred, y_true):
                return approxNDCGLoss(y_pred, y_true, self.eps, self.padded_value_indicator, self.alpha)
        
        loss_func = ApproxNDCGWrapper(
            eps=params['eps'],
            padded_value_indicator=params['padded_value_indicator'],
            alpha=params['alpha']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'eps': 1e-10,
            'padded_value_indicator': -1,
            'alpha': 1.0  # Score difference weight for sigmoid function
        }


class NeuralNDCGLossFactory(BaseLossFactory):
    """Factory for NeuralNDCG loss function"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        # Create a wrapper that handles the neuralNDCG function
        class NeuralNDCGWrapper(nn.Module):
            def __init__(self, variant='standard', temperature=1.0, powered_relevancies=True, 
                         k=None, stochastic=False, n_samples=32, beta=0.1, log_scores=True,
                         max_iter=50, tol=1e-6, padded_value_indicator=-1):
                super().__init__()
                self.variant = variant
                self.temperature = temperature
                self.powered_relevancies = powered_relevancies
                self.k = k
                self.stochastic = stochastic
                self.n_samples = n_samples
                self.beta = beta
                self.log_scores = log_scores
                self.max_iter = max_iter
                self.tol = tol
                self.padded_value_indicator = padded_value_indicator
            
            def forward(self, y_pred, y_true):
                if self.variant == 'transposed':
                    return neuralNDCG_transposed(
                        y_pred, y_true,
                        padded_value_indicator=self.padded_value_indicator,
                        temperature=self.temperature,
                        powered_relevancies=self.powered_relevancies,
                        k=self.k,
                        stochastic=self.stochastic,
                        n_samples=self.n_samples,
                        beta=self.beta,
                        log_scores=self.log_scores,
                        max_iter=self.max_iter,
                        tol=self.tol
                    )
                else:
                    return neuralNDCG(
                        y_pred, y_true,
                        padded_value_indicator=self.padded_value_indicator,
                        temperature=self.temperature,
                        powered_relevancies=self.powered_relevancies,
                        k=self.k,
                        stochastic=self.stochastic,
                        n_samples=self.n_samples,
                        beta=self.beta,
                        log_scores=self.log_scores
                    )
        
        loss_func = NeuralNDCGWrapper(
            variant=params['variant'],
            temperature=params['temperature'],
            powered_relevancies=params['powered_relevancies'],
            k=params['k'],
            stochastic=params['stochastic'],
            n_samples=params['n_samples'],
            beta=params['beta'],
            log_scores=params['log_scores'],
            max_iter=params['max_iter'],
            tol=params['tol'],
            padded_value_indicator=params['padded_value_indicator']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'variant': 'standard',  # 'standard' or 'transposed'
            'temperature': 1.0,
            'powered_relevancies': True,
            'k': None,  # None means use full slate length
            'stochastic': False,
            'n_samples': 32,
            'beta': 0.1,
            'log_scores': True,
            'max_iter': 50,
            'tol': 1e-6,
            'padded_value_indicator': -1
        }


class LambdaLossFactory(BaseLossFactory):
    """Factory for LambdaLoss and LambdaRank loss functions"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        class LambdaLossWrapper(nn.Module):
            def __init__(self, eps, padded_value_indicator, weighing_scheme, k, sigma, mu, reduction, reduction_log):
                super().__init__()
                self.eps = eps
                self.padded_value_indicator = padded_value_indicator
                self.weighing_scheme = weighing_scheme
                self.k = k
                self.sigma = sigma
                self.mu = mu
                self.reduction = reduction
                self.reduction_log = reduction_log
            
            def forward(self, y_pred, y_true):
                return lambdaLoss(
                    y_pred, y_true,
                    eps=self.eps,
                    padded_value_indicator=self.padded_value_indicator,
                    weighing_scheme=self.weighing_scheme,
                    k=self.k,
                    sigma=self.sigma,
                    mu=self.mu,
                    reduction=self.reduction,
                    reduction_log=self.reduction_log
                )
        
        loss_func = LambdaLossWrapper(
            eps=params['eps'],
            padded_value_indicator=params['padded_value_indicator'],
            weighing_scheme=params['weighing_scheme'],
            k=params['k'],
            sigma=params['sigma'],
            mu=params['mu'],
            reduction=params['reduction'],
            reduction_log=params['reduction_log']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'eps': 1e-10,
            'padded_value_indicator': -1,
            'weighing_scheme': None,  # None, 'ndcgLoss1_scheme', 'ndcgLoss2_scheme', 'lambdaRank_scheme', etc.
            'k': None,  # None means use full slate length
            'sigma': 1.0,
            'mu': 10.0,
            'reduction': 'sum',
            'reduction_log': 'binary'
        }


class LambdaRankFactory(BaseLossFactory):
    """Factory for LambdaRank loss (LambdaLoss with lambdaRank_scheme)"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        # LambdaRank is LambdaLoss with lambdaRank_scheme weighing
        params = {**self.get_default_params(), **loss_params}
        params['weighing_scheme'] = 'lambdaRank_scheme'  # Force LambdaRank weighing scheme
        
        class LambdaRankWrapper(nn.Module):
            def __init__(self, eps, padded_value_indicator, k, sigma, mu, reduction, reduction_log):
                super().__init__()
                self.eps = eps
                self.padded_value_indicator = padded_value_indicator
                self.k = k
                self.sigma = sigma
                self.mu = mu
                self.reduction = reduction
                self.reduction_log = reduction_log
            
            def forward(self, y_pred, y_true):
                return lambdaLoss(
                    y_pred, y_true,
                    eps=self.eps,
                    padded_value_indicator=self.padded_value_indicator,
                    weighing_scheme='lambdaRank_scheme',
                    k=self.k,
                    sigma=self.sigma,
                    mu=self.mu,
                    reduction=self.reduction,
                    reduction_log=self.reduction_log
                )
        
        loss_func = LambdaRankWrapper(
            eps=params['eps'],
            padded_value_indicator=params['padded_value_indicator'],
            k=params['k'],
            sigma=params['sigma'],
            mu=params['mu'],
            reduction=params['reduction'],
            reduction_log=params['reduction_log']
        )
        loss_func.to(device)
        return loss_func
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'eps': 1e-10,
            'padded_value_indicator': -1,
            'k': None,  # None means use full slate length
            'sigma': 1.0,
            'mu': 10.0,
            'reduction': 'sum',
            'reduction_log': 'binary'
        }


class NeuralLossFactory(BaseLossFactory):
    """Factory for Neural Loss (pretrained surrogate loss)"""
    
    def create_loss(self, loss_params: Dict[str, Any], device: str):
        params = {**self.get_default_params(), **loss_params}
        
        # Create loss model configuration
        loss_model_config = LossModelConfig(
            metric=params['metric'],
            max_seq_len=params['max_seq_len'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            eval_batch_size=params['eval_batch_size'],
            epochs=params['epochs'],
            min_seq_len=params['min_seq_len'],
            num_seqs=params['num_seqs'],
            num_seqs_val=params['num_seqs_val'],
            num_seqs_test=params['num_seqs_test'],
            repeat=params['repeat'],
            random_len=params['random_len'],
            target_type=params['target_type'],
            model_dim=params['model_dim'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            out_active=params['out_active'],
            at_k=params['at_k']
        )
        
        # Train or load the neural loss model
        loss_model = self._get_or_train_loss_model(loss_model_config, device, params.get('logger'))
        
        # Create the appropriate metric loss wrapper
        if loss_model_config.metric == 'dcg':
            metric_loss_model = NeuralRankNDCG(deepcopy(loss_model))
        elif loss_model_config.metric == 'avg_recall':
            metric_loss_model = NeuralRankRecall(deepcopy(loss_model))
        else:
            raise ValueError('metric must be dcg or avg_recall')
        
        return metric_loss_model
    
    def _get_or_train_loss_model(self, config: LossModelConfig, device: str, logger=None):
        """Get existing or train new loss model"""
        model_path = 'init_loss_model.pt'
        
        # Create the neural loss model
        loss_model = Model(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            embedding_num=config.embed_num,
            padd_idx=config.padd_idx,
            out_active=config.out_active
        )
        
        if os.path.exists(model_path):
            if logger:
                logger.info('Loading existing neural loss model...')
            loss_model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            if logger:
                logger.info('Training new neural loss model...')
            
            # Generate training data
            train_data = load_dataset(
                config.metric, config.num_seqs, config.repeat,
                config.min_seq_len, config.max_seq_len, config.random_len,
                config.target_type, stage='train', at_k=config.at_k, logger=logger
            )
            val_data = load_dataset(
                config.metric, config.num_seqs_val, 1,
                config.min_seq_len, config.max_seq_len, config.random_len,
                config.target_type, stage='val', at_k=config.at_k, logger=logger
            )
            test_data = load_dataset(
                config.metric, config.num_seqs_test, 1,
                config.min_seq_len, config.max_seq_len, config.random_len,
                config.target_type, stage='test', at_k=config.at_k, logger=logger
            )
            
            # Create datasets and dataloaders
            train_ds = RankDataset(train_data, config.padd_idx)
            val_ds = RankDataset(val_data, config.padd_idx)
            test_ds = RankDataset(test_data, config.padd_idx)
            train_dl = RankDataLoader(train_ds, config.batch_size, True)
            val_dl = RankDataLoader(val_ds, config.eval_batch_size, False)
            test_dl = RankDataLoader(test_ds, config.eval_batch_size, False)
            
            # Train the model
            loss_model = train_loss_model(
                config, loss_model, train_dl, val_dl, test_dl, device, logger, ''
            )
            
            # Save the trained model
            torch.save(loss_model.state_dict(), model_path)
        
        return loss_model
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'metric': 'dcg',
            'max_seq_len': 50,
            'lr': 0.001,
            'batch_size': 64,
            'eval_batch_size': 128,
            'epochs': 20,
            'min_seq_len': 40,
            'num_seqs': 1000,
            'num_seqs_val': 100,
            'num_seqs_test': 100,
            'repeat': 10,
            'random_len': False,
            'target_type': 3,
            'model_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'out_active': 'relu',
            'at_k': 10
        }


class ModularTrainingFramework:
    """Main training framework that orchestrates rankers and losses"""
    
    def __init__(self, sbert_model=None, alter_model=None):
        # Register available rankers and losses
        self.ranker_factories = {
            'transformer': TransformerRankerFactory(alter_model=alter_model),
        }
        
        self.loss_factories = {
            'ranknet': RankNetLossFactory(),
            'lrcl': LRCLLossFactory(),
            'neural': NeuralLossFactory(),
            'softndcg': SoftNDCGLossFactory(),
            'listmle': ListMLELossFactory(),
            'listnet': ListNetLossFactory(),
            'rmse': RMSELossFactory(),
            'approxndcg': ApproxNDCGLossFactory(),
            'neuralndcg': NeuralNDCGLossFactory(),
            'lambdaloss': LambdaLossFactory(),
            'lambdarank': LambdaRankFactory()
        }
    
    def register_ranker_factory(self, name: str, factory: BaseRankerFactory):
        """Register a new ranker factory"""
        self.ranker_factories[name] = factory
    
    def register_loss_factory(self, name: str, factory: BaseLossFactory):
        """Register a new loss factory"""
        self.loss_factories[name] = factory
    
    def create_ranker(self, ranker_type: str, config, device: str, logger):
        """Create a ranker of the specified type"""
        if ranker_type not in self.ranker_factories:
            raise ValueError(f"Unknown ranker type: {ranker_type}. Available: {list(self.ranker_factories.keys())}")
        
        factory = self.ranker_factories[ranker_type]
        return factory.create_ranker(config, device, logger)
    
    def create_loss(self, loss_type: str, loss_params: Dict[str, Any], device: str, logger=None):
        """Create a loss function of the specified type"""
        if loss_type not in self.loss_factories:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(self.loss_factories.keys())}")
        
        factory = self.loss_factories[loss_type]
        # Pass logger to neural loss factory for training progress
        if loss_type == 'neural' and logger:
            loss_params['logger'] = logger
        return factory.create_loss(loss_params, device)
    
    def load_config(self, ranker_type: str, config_file: str, job_dir: str, run_id: str):
        """Load configuration for the specified ranker type"""
        if ranker_type not in self.ranker_factories:
            raise ValueError(f"Unknown ranker type: {ranker_type}")
        
        factory = self.ranker_factories[ranker_type]
        return factory.load_config(config_file, job_dir, run_id)
    
    def train(self, ranker_type: str, loss_type: str, rank_config, logger, device: str, 
              tag: str, loss_params: Optional[Dict[str, Any]] = None, wandb_logger=None, dataset_name: str = None):
        """Main training function"""
        
        # Create ranker
        logger.phase = 'RANKER-SETUP'
        ranker = self.create_ranker(ranker_type, rank_config, device, logger)
        logger.info(f'Created {ranker_type} ranker')
        print()
        
        # Create loss function
        logger.phase = 'LOSS-SETUP' if loss_type != 'neural' else 'LOSS-MODEL'
        loss_params = loss_params or {}
        
        # For neural loss, set max_seq_len from rank_config if not provided
        if loss_type == 'neural' and 'max_seq_len' not in loss_params:
            loss_params['max_seq_len'] = rank_config.data.slate_length
        
        loss_function = self.create_loss(loss_type, loss_params, device, logger)
        logger.info(f'Using {loss_type} loss function')
        if loss_type != 'neural':  # Neural loss logs its own detailed parameters during training
            logger.info(f'Loss parameters: {loss_params}')
        print()
        
        # Attach wandb_logger to logger for test metrics logging
        if wandb_logger:
            logger.wandb_logger = wandb_logger
        
        # Train ranker
        logger.phase = 'RANK-MODEL'
        ranking_results_on_test = ranker.train(
            loss_function,
            from_scratch=True,
            tag=tag,
            wandb_logger=wandb_logger,
            dataset_name=dataset_name,
            loss_type=loss_type
        )
        
        return ranking_results_on_test


def load_modular_config(config_file: str) -> Dict[str, Any]:
    """Load modular configuration file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['ranker_type', 'loss_type', 'base_config']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    return config


def main():
    """Main entry point"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser(description='Modular LTR Training Framework')
    parser.add_argument('--config', required=True, help='Modular config file path')
    parser.add_argument('--ranker', help='Ranker type override (transformer, sbert)')
    parser.add_argument('--loss', help='Loss type override (ranknet, lrcl, neural, softndcg, neuralndcg)')
    parser.add_argument('--model', help='Model architecture (default uses main model, options: mlp, set_transformer, mil_att, mil_gate_att, attsets)', default=None)
    parser.add_argument('--fold', help='Fold id (1..N)', default=None)
    parser.add_argument('--wandb', help='Enable wandb logging', action='store_true')
    parser.add_argument('--wandb_project', help='WandB project name', default='modular-bert-ms-lclr')
    
    # Loss-specific parameters
    parser.add_argument('--sigma', type=float, help='RankNet sigma parameter')
    
    # LRCL loss specific parameters
    parser.add_argument('--use_gap_weight', action='store_true', help='LRCL: use gap weighting')
    parser.add_argument('--pair_subsample', type=int, help='LRCL: pair subsampling limit')
    parser.add_argument('--lrcl_seed', type=int, default=0, help='LRCL: random seed')
    parser.add_argument('--margin_num_bases', type=int, default=8, help='LRCL: number of bases for learned margin function')
    parser.add_argument('--penalty_num_bases', type=int, default=8, help='LRCL: number of bases for learned penalty function')
    parser.add_argument('--weight_num_bases', type=int, default=6, help='LRCL: number of bases for learned gap weight function')
    parser.add_argument('--weight_floor', type=float, default=1e-3, help='LRCL: minimum floor value for gap weights')
    parser.add_argument('--slope_range', type=float, nargs=2, default=[0.5, 4.0], help='LRCL: slope range for monotone basis functions (min max)')
    parser.add_argument('--bias_range', type=float, nargs=2, default=[-2.0, 2.0], help='LRCL: bias range for monotone basis functions (min max)')
    
    # Neural loss specific parameters
    parser.add_argument('--neural_metric', choices=['dcg', 'avg_recall'], default='dcg', help='Neural loss metric')
    parser.add_argument('--neural_epochs', type=int, default=100, help='Neural loss training epochs')
    parser.add_argument('--neural_lr', type=float, default=0.001, help='Neural loss learning rate')
    parser.add_argument('--neural_batch_size', type=int, default=32, help='Neural loss batch size')
    parser.add_argument('--neural_num_seqs', type=int, default=1000, help='Neural loss training sequences')
    parser.add_argument('--neural_target_type', type=int, choices=[2, 3, 5], default=5, help='Neural loss target type')
    
    # SoftNDCG loss specific parameters
    #parser.add_argument('--softndcg_type', choices=['softndcg', 'learnable'], default='softndcg', help='SoftNDCG loss variant')
    parser.add_argument('--temperature', type=float, default=0.5, help='SoftNDCG temperature parameter')
    parser.add_argument('--gain_base', type=float, default=2.0, help='SoftNDCG gain base')
    parser.add_argument('--discount_base', type=float, default=2.0, help='SoftNDCG discount base')
    
    # NeuralNDCG loss specific parameters
    parser.add_argument('--neuralndcg_variant', type=str, default='standard', choices=['standard', 'transposed'], help='NeuralNDCG variant')
    parser.add_argument('--neuralndcg_temperature', type=float, default=1.0, help='NeuralNDCG temperature for NeuralSort')
    parser.add_argument('--neuralndcg_powered_relevancies', action='store_true', default=True, help='NeuralNDCG: apply 2^x - 1 gain function')
    parser.add_argument('--neuralndcg_k', type=int, default=None, help='NeuralNDCG: rank at which loss is truncated (None = full slate)')
    parser.add_argument('--neuralndcg_stochastic', action='store_true', help='NeuralNDCG: use stochastic variant')
    parser.add_argument('--neuralndcg_n_samples', type=int, default=32, help='NeuralNDCG: number of stochastic samples')
    parser.add_argument('--neuralndcg_beta', type=float, default=0.1, help='NeuralNDCG: beta parameter for stochastic NeuralSort')
    parser.add_argument('--neuralndcg_log_scores', action='store_true', default=True, help='NeuralNDCG: log_scores parameter for NeuralSort')
    parser.add_argument('--neuralndcg_max_iter', type=int, default=50, help='NeuralNDCG: max iterations for Sinkhorn scaling')
    parser.add_argument('--neuralndcg_tol', type=float, default=1e-6, help='NeuralNDCG: tolerance for Sinkhorn scaling')
    parser.add_argument('--neuralndcg_padded_value_indicator', type=int, default=-1, help='NeuralNDCG: padded value indicator')

    
    args = parser.parse_args()
    
    # Load modular configuration
    modular_config = load_modular_config(args.config)
    
    # Override config with command line arguments
    ranker_type = args.ranker or modular_config['ranker_type']
    loss_type = args.loss or modular_config['loss_type']
    base_config_file = modular_config['base_config']
    
    # Get model architecture parameter
    alter_model = args.model
    
    # Determine job directory and dataset
    if 'MQ2007' in base_config_file:
        rank_job_dir = 'MQ2007'
    elif 'MQ2008' in base_config_file:
        rank_job_dir = 'MQ2008'
    elif 'WEB10K' in base_config_file:
        rank_job_dir = 'WEB10K'
    else:
        rank_job_dir = 'default'
    
    # Load base configuration
    base_config_path = path.join('configs', base_config_file)
    with open(base_config_path, 'r') as f:
        base_config_dict = json.load(f)
    
    # Extract cross-validation configuration
    cross_validation_config = base_config_dict.pop('cross_validation', None)
    
    # Initialize WandB logger if enabled
    wandb_logger = None
    if args.wandb:
        try:
            experiment_config = {
                'ranker_type': ranker_type,
                'loss_type': loss_type,
                'dataset': rank_job_dir,
                'base_config': base_config_file,
            }
            
            # Add model architecture to config
            if alter_model:
                experiment_config['model_architecture'] = alter_model
            
            # Add loss-specific config
            if loss_type == 'ranknet':
                experiment_config['sigma'] = args.sigma or 1.0
            elif loss_type == 'lrcl':
                experiment_config.update({
                    'use_gap_weight': bool(args.use_gap_weight),
                    'pair_subsample': args.pair_subsample,
                    'lrcl_seed': args.lrcl_seed
                })
            elif loss_type == 'neural':
                experiment_config.update({
                    'neural_metric': args.neural_metric,
                    'neural_epochs': args.neural_epochs,
                    'neural_lr': args.neural_lr,
                    'neural_batch_size': args.neural_batch_size,
                    'neural_num_seqs': args.neural_num_seqs,
                    'neural_target_type': args.neural_target_type
                })
            elif loss_type == 'softndcg':
                experiment_config.update({
                    #'softndcg_type': args.softndcg_type,
                    'temperature': args.temperature,
                    'gain_base': args.gain_base,
                    'discount_base': args.discount_base
                })
            elif loss_type == 'neuralndcg':
                experiment_config.update({
                    'neuralndcg_variant': args.neuralndcg_variant,
                    'neuralndcg_temperature': args.neuralndcg_temperature,
                    'neuralndcg_powered_relevancies': args.neuralndcg_powered_relevancies,
                    'neuralndcg_k': args.neuralndcg_k,
                    'neuralndcg_stochastic': args.neuralndcg_stochastic,
                    'neuralndcg_n_samples': args.neuralndcg_n_samples,
                    'neuralndcg_beta': args.neuralndcg_beta,
                    'neuralndcg_log_scores': args.neuralndcg_log_scores,
                    'neuralndcg_max_iter': args.neuralndcg_max_iter,
                    'neuralndcg_tol': args.neuralndcg_tol,
                    'neuralndcg_padded_value_indicator': args.neuralndcg_padded_value_indicator
                })
            
            wandb_logger = WandbParameterLogger(
                project_name=args.wandb_project,
                experiment_name=f"{ranker_type}_{loss_type}_{rank_job_dir}",
                config=experiment_config
            )
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            args.wandb = False
    
    # Prepare loss parameters
    def make_loss_params():
        if loss_type == 'ranknet':
            return {'sigma': args.sigma or 1.0}
        elif loss_type == 'lrcl':
            return {
                'use_gap_weight': bool(args.use_gap_weight),
                'pair_subsample': args.pair_subsample,
                'random_state': args.lrcl_seed,
                'margin_num_bases': args.margin_num_bases,
                'penalty_num_bases': args.penalty_num_bases,
                'weight_num_bases': args.weight_num_bases,
                'weight_floor': args.weight_floor,
                'slope_range': tuple(args.slope_range),
                'bias_range': tuple(args.bias_range)
            }
        elif loss_type == 'neural':
            return {
                'metric': args.neural_metric,
                'epochs': args.neural_epochs,
                'lr': args.neural_lr,
                'batch_size': args.neural_batch_size,
                'num_seqs': args.neural_num_seqs,
                'target_type': args.neural_target_type
            }
        elif loss_type == 'softndcg':
            return {
                #'loss_type': args.softndcg_type,
                'temperature': args.temperature,
                'gain_base': args.gain_base,
                'discount_base': args.discount_base
            }
        elif loss_type == 'neuralndcg':
            return {
                'variant': args.neuralndcg_variant,
                'temperature': args.neuralndcg_temperature,
                'powered_relevancies': args.neuralndcg_powered_relevancies,
                'k': args.neuralndcg_k,
                'stochastic': args.neuralndcg_stochastic,
                'n_samples': args.neuralndcg_n_samples,
                'beta': args.neuralndcg_beta,
                'log_scores': args.neuralndcg_log_scores,
                'max_iter': args.neuralndcg_max_iter,
                'tol': args.neuralndcg_tol,
                'padded_value_indicator': args.neuralndcg_padded_value_indicator
            }
        else:
            return modular_config.get('loss_params', {})
    
    # Initialize framework with SBERT model and alter_model parameters
    framework = ModularTrainingFramework(
        alter_model=alter_model if ranker_type == 'transformer' else None
    )
        # Cross-validation mode
    if cross_validation_config and cross_validation_config.get('enabled', False):
        folds = cross_validation_config.get('folds', [1])
        seeds = cross_validation_config.get('seeds', [1])
        
        all_results = {}
        
        for fold in folds:
            print('\n' + '=' * 80)
            print(f'PROCESSING FOLD {fold}')
            print('=' * 80)
            
            # Create temporary config file for this fold
            temp_config_file = base_config_path.replace('.json', f'_temp_fold{fold}.json')
            with open(temp_config_file, 'w') as f:
                json.dump(base_config_dict, f, indent=2)
            
            rank_run_id = 'run_1'
            fold_config = framework.load_config(ranker_type, temp_config_file, rank_job_dir, rank_run_id)
            
            # Clean up temp file
            os.remove(temp_config_file)
            
            # Inject fold path
            fold_config.data.path = fold_config.data.path.format(fold=fold)
            print(f'Dataset path: {fold_config.data.path}')
            
            loss_params = make_loss_params()
            
            fold_results = {}
            for seed in seeds:
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                
                print('\n\n\n')
                print('=' * 80)
                print(f'\n  Fold: {fold}, Seed: {seed}\n')
                print('=' * 80)
                print()
                
                # Start wandb run for this fold/seed
                if wandb_logger:
                    wandb_logger.init_run(fold=fold, seed=seed)
                
                ranking_results = framework.train(
                    ranker_type=ranker_type,
                    loss_type=loss_type,
                    rank_config=fold_config,
                    logger=Logger(),
                    device=device,
                    tag=fold,
                    loss_params=loss_params,
                    wandb_logger=wandb_logger,
                    dataset_name=rank_job_dir
                )
                fold_results[f'seed-{seed}'] = ranking_results
                
                if wandb_logger:
                    wandb_logger.finish_run()
            
            all_results[f'fold-{fold}'] = fold_results
        
        # Print summary
        print('\n' + '=' * 80)
        print('SUMMARY RESULTS')
        print('=' * 80)
        for fold_key, fold_data in all_results.items():
            print(f'\n{fold_key}:')
            for seed_key, result in fold_data.items():
                print(f'  {seed_key}: {result}')
        
        # Calculate averages
        metrics = ["ndcg_1", "ndcg_3", "ndcg_5", "ndcg_10", "recall_1", "recall_3", "recall_5", "recall_10", "mrr_1", "mrr_3", "mrr_5", "mrr_10", "precision_1", "precision_3", "precision_5", "precision_10"]
        if len(folds) > 1:
            print('\nAVERAGE ACROSS FOLDS:')
            avg_results = {}
            vals = {m: [] for m in metrics}
            for fold_data in all_results.values():
                for seed_data in fold_data.values():
                    for m in metrics:
                        if m in seed_data:
                            vals[m].append(seed_data[m])
            for m, arr in vals.items():
                if arr:
                    avg_results[m] = float(np.mean(arr))
            print(f'Average: {avg_results}')
            
            # Log average results to wandb if available
            if wandb_logger:
                wandb_logger.log_metrics(avg_results, 0, "test_avg")
        
    # Single-fold mode
    else:
        seeds = [1]
        
        temp_config_file = base_config_path.replace('.json', '_temp_single.json')
        with open(temp_config_file, 'w') as f:
            json.dump(base_config_dict, f, indent=2)
        
        rank_run_id = 'run_1'
        rank_config = framework.load_config(ranker_type, temp_config_file, rank_job_dir, rank_run_id)
        
        # Clean up temp file
        os.remove(temp_config_file)
        
        # Optional: switch fold index by CLI
        if args.fold is not None:
            rank_config.data.path = rank_config.data.path.replace('Fold1', f'Fold{args.fold}')
            print(rank_config.data.path)
        
        loss_params = make_loss_params()
        
        results = {}
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            print('\n\n\n')
            print('=' * 80)
            print(f'\n  seed: {seed}\n')
            print('=' * 80)
            print()
            
            if wandb_logger:
                wandb_logger.init_run(fold=args.fold, seed=seed)
            
            fold = args.fold or ''
            ranking_results = framework.train(
                ranker_type=ranker_type,
                loss_type=loss_type,
                rank_config=rank_config,
                logger=Logger(),
                device=device,
                tag=fold,
                loss_params=loss_params,
                wandb_logger=wandb_logger,
                dataset_name=rank_job_dir
            )
            results[f'seed-{seed}'] = ranking_results
            
            if wandb_logger:
                wandb_logger.finish_run()
        
        print(f'\nfold{args.fold}: [')
        for seed, result in results.items():
            print(f'{result},')
        print(']')


if __name__ == '__main__':
    main()