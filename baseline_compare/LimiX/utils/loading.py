import torch
import random
import numpy as np
from model.transformer import FeaturesTransformer

def build_model(config:dict):
    model = FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        feature_positional_embedding_type=config.get('feature_positional_embedding_type', "subortho"),
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_prediction=config.get('mask_prediction', False),
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        pre_norm=config.get('pre_norm', True),
        activation=config.get('activation', 'gelu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        device=config.get('device', None),
        dtype=config.get('dtype', None),
        recompute_attn=config['recompute_attn'],
        layer_arch=config.get('layer_arch', 'fmfmsm'),
        self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
        cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', True),
        seq_attn_isolated=config.get('seq_attn_isolated', False),
        seq_attn_serial=config.get('seq_attn_serial', False),
    )
    return model

def load_model(model_path,mask_prediction:bool=False):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config = state_dict['config']
    config['mask_prediction'] = mask_prediction

    model = build_model(config)
    model.load_state_dict(state_dict['state_dict'])

    model.eval()
    return model