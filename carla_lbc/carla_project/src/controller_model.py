import uuid
import argparse
import pathlib

import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw

from map_model import MapModel
from models import SegmentationModel
from dataset import get_dataset
from converter import Converter
import src.common as common


class RawController(torch.nn.Module):
    def __init__(self, n_input=4, k=32):
        self.layers = torch.nn.Sequential(
                torch.nn.BatchNorm1d(n_input * 2),
                torch.nn.Linear(n_input * 2, k), torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, k), torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, 2),
                )

    def forward(self, x):
        return self.layers(torch.flatten(x, 1))


@torch.no_grad()
def visualize(batch, out, loss):
    import torchvision
    import wandb

    images = list()

    for i in range(out.shape[0]):
        _loss = loss[i]
        _out = out[i]
        rgb, topdown, points, heatmap, heatmap_img, actions, meta = [x[i] for x in batch]

        _rgb = Image.fromarray(np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255))
        _draw_rgb = ImageDraw.Draw(_rgb)
        _draw_rgb.text((5, 10), 'Loss: %.3f' % _loss)
        _draw_rgb.text((5, 30), 'Meta: %s' % meta)

        for x, y in _out:
            x = (x + 1) / 2 * _rgb.width
            y = (y + 1) / 2 * _rgb.height

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 255, 0))

        for x, y in _labels_cam:
            x = (x + 1) / 2 * _rgb.width
            y = (y + 1) / 2 * _rgb.height

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw_map = ImageDraw.Draw(_topdown)

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _labels_map:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _rgb.thumbnail((128, 128))
        _topdown.thumbnail(_rgb.size)

        image = np.hstack((_rgb, _topdown)).transpose(2, 0, 1)
        images.append((_loss, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


class ImageModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.net = SegmentationModel(4, 4)

        self.teacher = MapModel.load_from_checkpoint(pathlib.Path('/home/bradyzhou/code/carla_random/') / hparams.teacher_path)
        # self.teacher.eval()

        self.converter = Converter()

    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    @torch.no_grad()
    def _get_labels(self, batch):
        img, topdown, points, heatmap, heatmap_img, meta = batch
        out = self.teacher.forward(torch.cat([topdown, heatmap], 1))

        return out

    def training_step(self, batch, batch_nb):
        img, topdown, points, heatmap, heatmap_img, meta = batch
        labels_map = self._get_labels(batch)

        labels_cam = self.converter.map_to_cam((labels_map + 1) / 2 * 256)
        labels_cam[..., 0] = (labels_cam[..., 0] / 256) * 2 - 1
        labels_cam[..., 1] = (labels_cam[..., 1] / 144) * 2 - 1

        out = self.forward(torch.cat([img, heatmap_img], 1))

        loss = torch.nn.functional.l1_loss(out, labels_cam, reduction='none').mean((1, 2))
        loss_mean = loss.mean()

        metrics = {'train_loss': loss_mean.item()}

        if batch_nb % 250 == 0:
            metrics['train_image'] = visualize(batch, out, labels_cam, labels_map, loss)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss_mean}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, heatmap, heatmap_img, meta = batch
        labels_map = self._get_labels(batch)

        labels_cam = self.converter.map_to_cam((labels_map + 1) / 2 * 256)
        labels_cam[..., 0] = (labels_cam[..., 0] / 256) * 2 - 1
        labels_cam[..., 1] = (labels_cam[..., 1] / 144) * 2 - 1

        out = self.forward(torch.cat([img, heatmap_img], 1))

        loss = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_mean = loss.mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': visualize(batch, out, labels_cam, labels_map, loss)
                }, self.global_step)

        return {'val_loss': loss_mean.item()}

    def validation_epoch_end(self, outputs):
        results = {'val_loss': list()}

        for output in outputs:
            for key in results:
                results[key].append(output[key])

        summary = {key: np.mean(val) for key, val in results.items()}
        self.logger.log_metrics(summary, self.global_step)

        return summary

    def configure_optimizers(self):
        return torch.optim.Adam(
                self.net.parameters(),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size)

    def val_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size)


def main(hparams):
    model = ImageModel(hparams)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='distillation')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=2)

    try:
        resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
    except:
        resume_from_checkpoint = None

    trainer = pl.Trainer(
            gpus=1, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)

    parser.add_argument('--teacher_path', type=pathlib.Path, required=True)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-6)

    parsed = parser.parse_args()
    parsed.save_dir = parsed.save_dir / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
