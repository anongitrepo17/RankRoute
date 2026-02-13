# from urllib.parse import urlparse
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders, create_test_dataloader, load_libsvm_dataset_role
from allrank.data.load_npz import load_npz_dataset, load_ds
from allrank.models.model import make_model
from allrank.models.model_utils import CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.experiments import unpack_numpy_values  # pylint: disable=unused-import
from allrank.utils.file_utils import create_output_dirs, PathsContainer
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from allrank.do_test import do_test
from allrank.models.alter_models import make_alter_model
from attr import asdict
from pprint import pformat
from torch import optim


def load_config(config_file_name, job_dir, run_id):
    paths = PathsContainer.from_args(job_dir, run_id, config_file_name)
    # create_output_dirs(paths.output_dir)
    config = Config.from_json(paths.config_path)
    config.paths = paths
    return config


class Ranker:
    def __init__(self, config, alter_model=None, device='cpu', logger=None):
        self.config = config
        self.paths = config.paths
        self.alter_model = alter_model
        self.device = device
        self.logger = logger

        if logger is None:
            self.logger = init_logger(self.paths.output_dir)

        self.logger.info(f"created paths container {self.paths}")
        self.logger.info(f"Config:\n {pformat(vars(config), width=1)}")

        # train_ds, val_ds
        if config.data.ds_type != 'npz':
            train_ds, val_ds = load_libsvm_dataset(
                input_path=config.data.path,
                slate_length=config.data.slate_length,
                validation_ds_role=config.data.validation_ds_role,
            )
        else:
            train_ds, val_ds = load_npz_dataset(input_path=config.data.path,
                validation_ds_role=config.data.validation_ds_role)

        n_features = train_ds.shape[-1]
        assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"
        self.n_features = n_features

        # train_dl, val_dl
        self.train_dl, self.val_dl = create_data_loaders(train_ds, val_ds,
                                                         num_workers=config.data.num_workers,
                                                         batch_size=config.data.batch_size)

        if self.config.test_metrics is not None:
            # test_ds
            if self.config.data.ds_type != 'npz':
                test_ds = load_libsvm_dataset_role('test', config.data.path, config.data.slate_length)
            else:
                test_ds = load_ds(role='test', input_path=config.data.path)
            self.test_dl = create_test_dataloader(test_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

        self.model = None

    def create_model(self):
        # instantiate model
        if self.alter_model is None or self.alter_model == 'none':
            model = make_model(n_features=self.n_features, **asdict(self.config.model, recurse=False))
        else:
            model = make_alter_model(self.alter_model, self.n_features, self.config.alter_models, self.config.data.slate_length)
        if torch.cuda.device_count() > 1:
            model = CustomDataParallel(model)
            self.logger.info(f"Model training will be distributed to {torch.cuda.device_count()} GPUs.")
        return model

    def train(self, loss_func, from_scratch=True, tag='', wandb_logger=None, dataset_name=None, loss_type=None):
        if from_scratch or self.model is None:
            self.model = self.create_model()
        self.model.to(self.device)

        # load optimizer, loss and LR scheduler
        #optimizer = getattr(optim, self.config.optimizer.name)(params=self.model.parameters(), **self.config.optimizer.args)
        
        all_params = list(self.model.parameters()) + list(loss_func.parameters())
        optimizer = getattr(optim, self.config.optimizer.name)(params=all_params, **self.config.optimizer.args)
        
        if self.config.lr_scheduler.name:
            scheduler = getattr(optim.lr_scheduler, self.config.lr_scheduler.name)(optimizer, **self.config.lr_scheduler.args)
        else:
            scheduler = None

        with torch.autograd.detect_anomaly() if self.config.detect_anomaly else dummy_context_mgr():  # type: ignore
            train_args = asdict(self.config.training)

            # run training
            result = fit(
                train_args["epochs"],
                model=self.model,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dl=self.train_dl,
                valid_dl=self.val_dl,
                config=self.config,
                gradient_clipping_norm=train_args['gradient_clipping_norm'],
                early_stopping_patience=train_args['early_stopping_patience'],
                device=self.device,
                output_dir=self.paths.output_dir,
                tensorboard_output_path=self.paths.tensorboard_output_path,
                logger=self.logger,
                tag=tag,
                wandb_logger=wandb_logger
            )

        # Save train predictions for reranker training
        if dataset_name is not None and tag and loss_type is not None:
            from allrank.do_test import save_predictions
            self.logger.info("Saving train predictions for reranker...")
            save_predictions(self.model, self.train_dl, self.device, self.logger,
                           dataset_name, tag, loss_type, split='train')

        if self.config.test_metrics is not None:
            self.logger.stage = "Test"
            test_result = do_test(model=self.model, test_dl=self.test_dl,
                                  config=self.config, device=self.device, logger=self.logger,
                                  dataset_name=dataset_name, fold=tag, loss_type=loss_type)
            result.update(test_result)

        self.logger.stage = None

        return unpack_numpy_values(test_result['test_metrics'])
