import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
import argparse
from fis import FIS
from fis.encoders import BasicEncoder
from fis.decoders import BasicDecoder
from fis.utils import calc_psnr, calc_ssim, to_np_img
from tqdm import tqdm
from imageio import imread, imwrite
from utils import get_loader, get_path
from PIL import Image
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)



parser = argparse.ArgumentParser("Frequency-Guided Iterative Network for Image Steganography")

parser.add_argument("--dataset", type=str, default="div2k", choices=["div2k", "mscoco", "celeba"])
parser.add_argument("--bits", type=int, default=1)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--random-crop", type=int, default=360)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--limit", type=int, default=800, help="number of training images")
parser.add_argument("--hidden-size", type=int, default=32)
parser.add_argument("--private-key", type=int, default=11111)
parser.add_argument("--mse-weight", type=float, default=1.0)
parser.add_argument("--step-size", type=float, default=1.0)
parser.add_argument("--iters", type=int, default=12)
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--test-step-size", type=float, default=0.1)
parser.add_argument("--test-iters", type=int, default=150)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--constraint", type=float, default=None)

args = parser.parse_args()


if __name__ == "__main__":
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    train, validation = get_loader(args)
    save_dir = get_path(args)
    print(save_dir)

    if args.eval and args.load is None:
        args.load = os.path.join(save_dir, "checkpoints", "best.steg")
        if not os.path.isfile(args.load):
            print("Using the latest checkpoint instead of the best.")
            args.load = os.path.join(save_dir, "checkpoints", "latest.steg")
    if args.load is not None and os.path.isfile(args.load):
        print(f"Loading pretrained weight from {args.load}.")
        model = FIS.load(path=args.load)
    else:
        print("Creating a new model.")
        model = FIS(
            data_depth=args.bits,
            encoder=BasicEncoder,
            decoder=BasicDecoder,
            hidden_size=args.hidden_size,
            iters=args.iters,
            lr=args.lr,
            opt=args.opt,
            private_key=args.private_key,
        )

    if args.eval:
        model.encoder.iters = args.test_iters
        model.encoder.step_size = args.test_step_size
    else:
        model.encoder.step_size = args.step_size

    model.mse_weight = args.mse_weight
    model.encoder.constraint = args.constraint

    if args.eval:
        out_folder = os.path.join(save_dir, "samples")
        if args.lbfgs:
            out_folder = f"{out_folder}_lbfgs"
        if args.constraint is not None:
            out_folder = f"{out_folder}_{args.constraint}_constraint"
        os.makedirs(out_folder, exist_ok=True)

        img_names = [os.path.basename(x[0]).split(".")[0] for x in validation.dataset.imgs]
        print(f"{len(img_names)} images will be saved to {out_folder}.")

        times, steps, errors, ssims, psnrs = [], [], [], [], []

        for i, (cover, _) in tqdm(enumerate(validation)):
            cover = cover.cuda()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                generated, payload, decoded = model._encode_decode(cover, epoch=-10, quantize=True)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

            original_image = imread(validation.dataset.imgs[i][0], pilmode="RGB").astype(np.float32)
            _psnrs = [calc_psnr(
                original_image,
                to_np_img(x[0], dtype=np.float32)) for x in generated]
            with torch.no_grad():
                _errors = [float(1 - (x >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()) * 100 for x in decoded]

            costs = np.array([-y if x == 0 else x for x, y in zip(_errors, _psnrs)])
            #costs = np.array([_psnrs])
            best_idx = np.argmin(costs)
            #best_idx = np.argmax(costs)

            steps.append(best_idx)
            errors.append(_errors[best_idx])

            generated = to_np_img(generated[best_idx][0])
            imwrite(os.path.join(out_folder, f"{img_names[i]}.png"), generated)

            ssims.append(calc_ssim(original_image, generated.astype(np.float32)))
            psnrs.append(calc_psnr(original_image, generated.astype(np.float32)))

            log_str = f"{img_names[i]}, time: {times[-1]:0.2f}ms, steps: {steps[-1]}, error: {errors[-1]:0.6f}%, ssim: {ssims[-1]:0.3f}, psnr: {psnrs[-1]:0.2f}"
            print(log_str)

        print(f"Error: {np.mean(errors):0.6f}%")
        print(f"SSIM: {np.mean(ssims):0.3f}")
        print(f"PSNR: {np.mean(psnrs):0.2f}")
        with open(os.path.join(out_folder, f"time.txt"), "w") as f:
            f.write("\n".join(map(str, times)))
        with open(os.path.join(out_folder, f"step.txt"), "w") as f:
            f.write("\n".join(map(str, steps)))
        with open(os.path.join(out_folder, f"error.txt"), "w") as f:
            f.write("\n".join(map(str, errors)))
        with open(os.path.join(out_folder, f"ssim.txt"), "w") as f:
            f.write("\n".join(map(str, ssims)))
        with open(os.path.join(out_folder, f"psnr.txt"), "w") as f:
            f.write("\n".join(map(str, psnrs)))

    else:
        print(f"Start training for {args.epochs} epochs.")
        model.fit(train, validation, save_path=os.path.join(save_dir, "checkpoints"), epochs=args.epochs)
