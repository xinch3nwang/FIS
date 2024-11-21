import gc
import inspect
import os

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam, SGD
from tqdm import tqdm

from .utils import calc_psnr, calc_ssim, to_np_img



METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]


def seq_loss(loss_func, generated, target, gamma=0.8, normalize=False):
    weights = [gamma ** x for x in range(len(generated)-1, -1, -1)]
    loss = 0
    for w, x in zip(weights, generated):
        loss += loss_func(x, target) * w
    if normalize:
        loss /= sum(weights)
    return loss


class FIS(object):
    def _get_instance(self, class_or_instance, kwargs):
        if not inspect.isclass(class_or_instance):
            return class_or_instance
        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')
        if not cuda:
            print('Using CPU device')
        elif not self.cuda:
            print('CUDA is not available. Defaulting to CPU device')
        else:
            print('Using CUDA device')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        if not self.no_critic:
            self.critic.to(self.device)

    def __init__(self, data_depth, encoder, decoder, lr=1e-4, opt="adam",
                 cuda=True, verbose=True, extra_verbose=True, **kwargs):

        self.verbose = verbose
        self.extra_verbose = extra_verbose
        self.lr = lr
        self.opt = opt

        self.data_depth = data_depth

        print("data_depth", self.data_depth)
        kwargs['data_depth'] = data_depth
        self.decoder = self._get_instance(decoder, kwargs)
        self.encoder = self._get_instance(encoder, kwargs)
        self.set_device(cuda)

        self.encoder.decoder = self.decoder

        self.decoder_optimizer = None

        self.fit_metrics = None
        self.history = list()

    def _decoder(self, x):
        return self.decoder(x)

    def _random_data(self, cover):
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    def _encode_decode(self, cover, epoch=-10, quantize=False, payload=None, init_noise=False, verbose=False):
        if payload is None:
            payload = self._random_data(cover)
        generated = self.encoder(cover, payload, epoch, verbose=verbose)
        if quantize:
            for i in range(len(generated)):
                generated[i] = (255.0 * (generated[i] + 1.0) / 2.0).long()
                generated[i] = torch.clamp(generated[i], 0, 255)
                generated[i] = 2.0 * generated[i].float() / 255.0 - 1.0

        decoded = [self._decoder(x) for x in generated]

        return generated, payload, decoded

    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        opt_cls = Adam if self.opt == "adam" else SGD
        decoder_optimizer = opt_cls(_dec_list, lr=self.lr)

        return decoder_optimizer

    def _fit_coders(self, train, metrics, epoch, finetune=False):
        print("Training encoder & decoder.")
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, epoch)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)

            self.decoder_optimizer.zero_grad()
            if finetune:
                decoder_loss.backward()
            else:
                (self.mse_weight * encoder_mse + decoder_loss).backward()
            self.decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

    def _coding_scores(self, cover, generated, payload, decoded):
        encoder_mse = seq_loss(mse_loss, generated, cover, gamma=0.8)
        decoder_loss = seq_loss(binary_cross_entropy_with_logits, decoded, payload, gamma=0.8)
        decoder_acc = [(x >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel() for x in decoded]
       
        return encoder_mse, decoder_loss, max(decoder_acc)

    def _validate(self, validate, metrics, epoch):
        print("Validating.")
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            with torch.no_grad():
                generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)

            generated_score = torch.tensor(0)
            cover_score = torch.tensor(0)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(
                np.mean([
                    calc_ssim(
                        to_np_img(cover[i]),
                        to_np_img(generated[-1][i])) for i in range(cover.shape[0])
                ])
            )
            metrics['val.psnr'].append(
                np.mean([
                    calc_psnr(
                        to_np_img(cover[i]),
                        to_np_img(generated[-1][i])) for i in range(cover.shape[0])
                ])
            )
            metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    def fit(self, train, validate, save_path, epochs=5):
        print("Start training.")
        best_acc = 0
        os.makedirs(save_path, exist_ok=True)

        self.decoder_optimizer = self._get_optimizers()
        self.epochs = 0

        # Start training
        total = self.epochs + epochs
        for epoch in range(self.epochs + 1, epochs + 1):

            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            self._fit_coders(train, metrics, epoch)
            self.save(os.path.join(save_path, f"{epoch}.steg"))
            self.save(os.path.join(save_path, f"latest.steg"))

            self._validate(validate, metrics, epoch)

            self.fit_metrics = {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch
            if self.fit_metrics["val.decoder_acc"] > best_acc:
                best_acc = self.fit_metrics["val.decoder_acc"]
                self.save(os.path.join(save_path, f"best.steg"))
            print(self.fit_metrics)
            with open(os.path.join(os.path.dirname(save_path), "log.txt"), "a") as f:
                if epoch == 1:
                    f.write(", ".join(["epoch"] + list(self.fit_metrics.keys())) + "\n")
                f.write(", ".join(map(str, [epoch] + list(self.fit_metrics.values()))) + "\n")

            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path=None, cuda=True, verbose=False):
        model = torch.load(path, map_location='cpu')
        model.verbose = verbose
        model.set_device(cuda)

        return model
