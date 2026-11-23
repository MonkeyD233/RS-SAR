"""Main script to train a model"""
import argparse
import json
import torch.nn as nn
import torch
import numpy as np
import random

from rs_sar.utils.functions import count_params
from rs_sar.learners.initializer import Initializer
from rs_sar.learners.model import Model

from rs_sar.models import (SVNet_AdaPKCTheta, SVNet_AdaPKCXi, SVNet_PKC,
                               SVNet_FreqPKC, SVNet_MVA, SVNet_TMVA,
                               TARSSNet_V1, TARSSNet_V2,
                               DiFTNet_S_v2, DiFTNet_D_v2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json')
    parser.add_argument("--finetune", default=False, help="Path to a state_dict checkpoint (.pt) to initialize weights from.")
    args = parser.parse_args()

    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    _set_seeds(cfg)

    device = torch.device(cfg['device'])

    init = Initializer(cfg)
    data = init.get_data()

    if cfg['model'] == 'svnet_adapkctheta':
        net = SVNet_AdaPKCTheta(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'svnet_adapkcxi':
        # threshold: 0 = raw metric selection, > 0 = FiTOS confidence gating
        net = SVNet_AdaPKCXi(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'],
                     threshold=cfg.get('threshold', 0.))
    elif cfg['model'] == 'svnet_pkc':
        net = SVNet_PKC(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'svnet_mva':
        net = SVNet_MVA(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'svnet_tmva':
        net = SVNet_TMVA(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'svnet_freqpkc':
        net = SVNet_FreqPKC(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'tarssnet_v1':
        net = TARSSNet_V1(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'tarssnet_v2':
        net = TARSSNet_V2(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'diftnet_s_v2':
        net = DiFTNet_S_v2(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    elif cfg['model'] == 'diftnet_d_v2':
        net = DiFTNet_D_v2(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'],
                     dataset_type=data['cfg']['dataset'])
    else:
        raise KeyError('Model {} has not been supported yet.'.format(cfg['model']))

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    net.to(device)
    net.apply(_init_weights)

    if args.finetune:
        saved_model = torch.load(args.finetune, map_location=torch.device('cpu'))
        net.load_state_dict(saved_model)

    # All supported models take 5-frame RA sequences: add the temporal dim
    Model(net, data).train(add_temp=True)


def _init_weights(m):
    """Xavier-uniform for Linear/Conv2d weights, zero bias.
    BatchNorm layers keep PyTorch defaults (weight=1, bias=0)."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)


def _set_seeds(cfg):
    np.random.seed(cfg['numpy_seed'])
    random.seed(cfg['numpy_seed'])
    torch.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed(cfg['torch_seed'])
    torch.cuda.manual_seed_all(cfg['torch_seed'])


if __name__ == '__main__':
    main()
