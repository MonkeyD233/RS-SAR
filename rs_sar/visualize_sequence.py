"""Visualize ground-truth masks and model predictions for a specific sequence.

Usage:
  # With model inference
  python visualize_sequence.py --cfg config_files/svnet_adapkctheta.json \
      --model-path /path/to/model.pt --sequence city_1_0

  # Ground truth only (no model)
  python visualize_sequence.py --cfg config_files/svnet_adapkctheta.json \
      --sequence city_1_0 --gt-only
"""
import argparse
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

from rs_sar.utils.functions import normalize, get_transformations
from rs_sar.utils.paths import Paths
from rs_sar.models import (SVNet_AdaPKCTheta, SVNet_AdaPKCXi, SVNet_PKC,
                               SVNet_FreqPKC, SVNet_MVA, SVNet_TMVA,
                               TARSSNet_V1, TARSSNet_V2,
                               DiFTNet_S_v2, DiFTNet_D_v2)
from rs_sar.loaders.dataset import RSSAR
from rs_sar.loaders.dataloaders import SequenceDataset, RSSARDataset

COLORS = np.array([
    [0, 0, 0],          # 0: background
    [255, 0, 0],        # 1: small_vehicle
    [0, 255, 0],        # 2: large_vehicle
    [0, 0, 255],        # 3
    [255, 255, 0],      # 4
    [0, 255, 255],      # 5
    [255, 255, 255],    # 6
    [128, 0, 128],      # 7
], dtype=np.uint8)


def mask_to_rgb(mask, num_classes):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, min(num_classes, len(COLORS))):
        rgb[mask == c] = COLORS[c]
    return rgb


def make_comparison(ra_slice, gt_mask, pred_mask, num_classes):
    """Side-by-side: RA input | GT mask | Predicted mask."""
    ra_viz = ra_slice - ra_slice.min()
    ra_viz = (ra_viz / (ra_viz.max() + 1e-8) * 255).astype(np.uint8)
    ra_viz = np.stack([ra_viz] * 3, axis=-1)
    gt_viz = mask_to_rgb(gt_mask, num_classes)
    pred_viz = mask_to_rgb(pred_mask, num_classes)
    return np.concatenate([ra_viz, gt_viz, pred_viz], axis=1)


def load_model(cfg, model_path):
    model_name = cfg['model']
    kwargs = dict(n_classes=cfg['nb_classes'],
                  n_frames=cfg['nb_input_channels'],
                  dataset_type=cfg['dataset'])

    model_cls = {
        'svnet_adapkctheta': SVNet_AdaPKCTheta,
        'svnet_adapkcxi':   SVNet_AdaPKCXi,
        'svnet_pkc':        SVNet_PKC,
        'svnet_freqpkc':    SVNet_FreqPKC,
        'svnet_mva':        SVNet_MVA,
        'svnet_tmva':       SVNet_TMVA,
        'tarssnet_v1':      TARSSNet_V1,
        'tarssnet_v2':      TARSSNet_V2,
        'diftnet_s_v2':     DiFTNet_S_v2,
        'diftnet_d_v2':     DiFTNet_D_v2,
    }.get(model_name)

    if model_cls is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_cls(**kwargs)
    model.load_state_dict(torch.load(model_path))
    model.cuda().eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize masks for specific sequences')
    parser.add_argument('--cfg', required=True, help='Path to config JSON')
    parser.add_argument('--model-path', default=None, help='Path to model .pt')
    parser.add_argument('--sequence', nargs='+', required=True,
                        help='Sequence name(s), e.g. city_1_0')
    parser.add_argument('--output-dir', default=None,
                        help='Output root (default: {logs}/viz/{seq_name}/)')
    parser.add_argument('--gt-only', action='store_true',
                        help='Save ground-truth masks only, skip model inference')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Limit frames per sequence (0 = all)')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = json.load(f)

    model = None
    if not args.gt_only:
        if args.model_path is None:
            parser.error('--model-path is required unless --gt-only is set')
        model = load_model(cfg, Path(args.model_path))

    paths = Paths().get()
    dataset_root = paths['RS-SAR']
    logs_root = paths['logs']

    data = RSSAR()
    norm_type = cfg.get('norm_type', 'local')
    w_size = cfg.get('w_size', 288)
    h_size = cfg.get('h_size', 288)
    n_frames = cfg.get('nb_input_channels', 5)
    process_signal = cfg.get('process_signal', True)
    num_classes = cfg.get('nb_classes', 3)
    transform_names = cfg.get('transformations', '').split(',')

    # Deterministic transforms — no random flips at test time
    transformations = get_transformations(transform_names, split='test',
                                          sizes=(w_size, h_size))

    for seq_name in args.sequence:
        # ---- look up the sequence across all splits --------------------------
        seq_data = None
        for split in ('Train', 'Validation', 'Test'):
            split_dict = data.get(split)
            if seq_name in split_dict:
                seq_data = split_dict[seq_name]
                break

        if seq_data is None:
            print(f"[SKIP] Sequence '{seq_name}' not found in any split")
            continue

        # ---- use the exact same pipeline as tester.py -----------------------
        # Build a single-sequence dict, wrap in SequenceDataset + DataLoader
        single_seq = {seq_name: seq_data}
        seqset = SequenceDataset(single_seq)
        seq_loader = DataLoader(seqset, batch_size=1, shuffle=False, num_workers=0)

        # ---- output directory -----------------------------------------------
        out_dir = Path(args.output_dir) if args.output_dir \
                  else logs_root / 'viz' / seq_name
        mask_dir = out_dir / 'mask'
        output_dir = out_dir / 'output'
        cmp_dir = out_dir / 'compare'
        for d in (mask_dir, output_dir, cmp_dir):
            d.mkdir(parents=True, exist_ok=True)

        n_saved = 0
        with torch.no_grad():
            for sequence_data in seq_loader:
                seq_name_batch, seq = sequence_data
                path_to_frames = dataset_root / seq_name_batch[0]

                frame_dataset = RSSARDataset(
                    seq, path_to_frames, process_signal, n_frames,
                    transformations, add_temp=True)

                frame_loader = DataLoader(frame_dataset, batch_size=1,
                                          shuffle=False, num_workers=0)

                print(f"Processing '{seq_name_batch[0]}' → {out_dir}  "
                      f"({len(frame_loader)} frames)")

                for j, frame in enumerate(frame_loader):
                    if args.max_frames and n_saved >= args.max_frames:
                        break

                    ra_data = frame['ra_matrix'].cuda().float()
                    ra_mask = frame['ra_mask'].cuda().float()
                    ra_data = normalize(ra_data, cfg['dataset'], 'range_angle',
                                        norm_type=norm_type)

                    gt_mask = torch.argmax(ra_mask, dim=1).squeeze(0).cpu().numpy()
                    ra_np = ra_data[0, 0, -1].cpu().numpy()

                    if model is not None:
                        output = model(ra_data)
                        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                    else:
                        pred_mask = np.zeros_like(gt_mask)

                    # Save individual masks
                    Image.fromarray(mask_to_rgb(gt_mask, num_classes)) \
                         .save(mask_dir / f'mask_{n_saved:04d}.png')
                    if model is not None:
                        Image.fromarray(mask_to_rgb(pred_mask, num_classes)) \
                             .save(output_dir / f'output_{n_saved:04d}.png')

                    # Save side-by-side comparison
                    cmp = make_comparison(ra_np, gt_mask, pred_mask, num_classes)
                    Image.fromarray(cmp).save(cmp_dir / f'compare_{n_saved:04d}.png')

                    n_saved += 1

        print(f"  Saved {n_saved} frames")


if __name__ == '__main__':
    main()
