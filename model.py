from modules.crrt import RRTMIL_COLMVKD

import torch
import torch.nn as nn


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def rrt_mil_col_kd_directconcat(num_classes: int = 2, has_logits: bool = True):

    model = RRTMIL_COLMVKD(input_dim=4864, n_classes=num_classes, epeg_k=15, crmsa_k=3)
    return model
