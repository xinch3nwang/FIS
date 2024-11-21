import os
import numpy as np
from fis.loader import DataLoader


def get_path(args):
    path = f"logs/{args.dataset}/{args.bits}_bits"
    if args.seed is not None:
        path = f"{path}_{args.seed}_seed"

    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


def get_loader(args):
    if args.eval:
        train = None
    else:
        train = DataLoader(
            f"../datasets/{args.dataset}/train/",
            limit=args.limit,
            shuffle=True,
            batch_size=args.batch_size,
            train=True,
            crop_size=args.random_crop)
    validation = DataLoader(
        f"../datasets/{args.dataset}/val/",
        limit=np.inf,
        shuffle=False,
        batch_size=1,
        train=False,
        crop_size=args.random_crop)
    return train, validation
