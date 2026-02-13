from ..model import LTRModel, FCModel, OutputLayer
import torch.nn as nn
from .set_transformer import SetTransformer
from .mil import MIL
from .attsets import AttSets

def make_alter_model(alter_model, n_features, alter_models_config, slate_length):
    assert alter_model in ['mlp', 'set_transformer', 'mil_att', 'mil_gate_att', 'attsets']
    config = alter_models_config[alter_model]
    feat_dim = config['fc_model']['sizes'][0]
    if alter_model == 'mlp':
        encoder = None
    elif alter_model == 'set_transformer':
        encoder = SetTransformer(feat_dim, **config["encoder"])
    elif alter_model == 'mil_att':
        encoder = MIL(feat_dim, slate_length, False)
    elif alter_model == 'mil_gate_att':
        encoder = MIL(feat_dim, slate_length, True)
    elif alter_model == 'attsets':
        encoder = AttSets(feat_dim)
    else:
        encoder = None
    return make_model(config['fc_model'], encoder, config['post_model'], n_features)


def make_model(pre_args, encoder, post_args, n_features):
    fc_model = FCModel(**pre_args, n_features=n_features)  # type: ignore
    d_model = n_features if not fc_model else fc_model.output_size
    model = LTRModel(fc_model, encoder, OutputLayer(d_model, **post_args))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
