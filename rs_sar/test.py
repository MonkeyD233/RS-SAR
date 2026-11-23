"""Main script to test a pretrained model"""
import argparse
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from rs_sar.utils.functions import count_params
from rs_sar.learners.tester import Tester
from rs_sar.models import (SVNet_AdaPKCTheta, SVNet_AdaPKCXi, SVNet_PKC,
                           SVNet_FreqPKC, SVNet_MVA, SVNet_TMVA,
                           TARSSNet_V1, TARSSNet_V2,
                           DiFTNet_S_v2, DiFTNet_D_v2)
from rs_sar.loaders.dataset import RSSAR
from rs_sar.loaders.dataloaders import SequenceDataset

MODELS = {
    'svnet_adapkctheta': SVNet_AdaPKCTheta,
    'svnet_adapkcxi': SVNet_AdaPKCXi,
    'svnet_pkc': SVNet_PKC,
    'svnet_freqpkc': SVNet_FreqPKC,
    'svnet_mva': SVNet_MVA,
    'svnet_tmva': SVNet_TMVA,
    'tarssnet_v1': TARSSNet_V1,
    'tarssnet_v2': TARSSNet_V2,
    'diftnet_s_v2': DiFTNet_S_v2,
    'diftnet_d_v2': DiFTNet_D_v2,
}


def build_model(cfg):
    model_cls = MODELS.get(cfg['model'])
    if model_cls is None:
        raise ValueError("Unsupported model type: {}".format(cfg['model']))
    kwargs = dict(n_classes=cfg['nb_classes'],
                  n_frames=cfg['nb_input_channels'],
                  dataset_type=cfg['dataset'])
    if cfg['model'] == 'svnet_adapkcxi':
        # threshold: 0 = raw metric selection, > 0 = FiTOS confidence gating
        kwargs['threshold'] = cfg.get('threshold', 0.)
    model = model_cls(**kwargs)
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))
    return model


def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file of the model to test.',
                        default='config.json')
    parser.add_argument('--model-path', default=None, dest="model_path",
                        help='Path to pretrained model weight.')
    args = parser.parse_args()
    cfg_path = args.cfg
    model_path = Path(args.model_path)
    model_parent_dir = model_path.parent
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)

    test_results_path = model_parent_dir / 'test_results.json'

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(cfg['device'])

    tester = Tester(cfg)
    data = RSSAR()
    test = data.get('Test')
    testset = SequenceDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    # All supported models take 5-frame RA sequences: add the temporal dim
    test_results = tester.predict(model, seq_testloader, get_quali=True, add_temp=True)
    tester.write_params(test_results_path)

    print('\n========== Test Results ==========')
    print(f"Pixel Accuracy : {test_results['global_acc']:.4f}")
    print(f"Pixel Precision: {test_results['global_prec']:.4f}")
    print(f"Pixel Recall   : {test_results['global_recall']:.4f}")
    print(f"mIoU           : {test_results['global_miou']:.4f}")
    print(f"Dice           : {test_results['global_dice']:.4f}")
    print(f"Results saved to: {test_results_path}")


if __name__ == '__main__':
    test_model()
