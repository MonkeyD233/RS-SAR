#!/usr/bin/env python3
"""Count trainable parameters for every model in the model zoo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rs_sar.models import (
    SVNet_PKC,
    SVNet_AdaPKCXi,
    SVNet_AdaPKCTheta,
    SVNet_MVA,
    SVNet_TMVA,
    SVNet_FreqPKC,
    TARSSNet_V1,
    TARSSNet_V2,
    DiFTNet_S_v2,
    DiFTNet_D_v2,
)

MODELS = {
    "SVNet_PKC": SVNet_PKC,
    "SVNet_AdaPKCXi": SVNet_AdaPKCXi,
    "SVNet_AdaPKCTheta": SVNet_AdaPKCTheta,
    "SVNet_MVA": SVNet_MVA,
    "SVNet_TMVA": SVNet_TMVA,
    "SVNet_FreqPKC": SVNet_FreqPKC,
    "TARSSNet_V1": TARSSNet_V1,
    "TARSSNet_V2": TARSSNet_V2,
    "DiFTNet_S_v2": DiFTNet_S_v2,
    "DiFTNet_D_v2": DiFTNet_D_v2,
}


def main():
    n_classes = 3
    n_frames = 5
    dataset_type = "RS-SAR"

    print(f"{'Model':<22} {'Total Params':>14} {'Trainable':>14}")
    print("-" * 52)
    for name, model_cls in MODELS.items():
        model = model_cls(n_classes, n_frames, dataset_type)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<22} {total:>14,} {trainable:>14,}")


if __name__ == "__main__":
    main()
