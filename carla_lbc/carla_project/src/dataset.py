from pathlib import Path

import numpy as np
import torch
import imgaug.augmenters as iaa
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from numpy import nan

from converter import Converter, PIXELS_PER_WORLD
from dataset_wrapper import Wrap
import src.common as common


# Reproducibility.
np.random.seed(0)
torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 1
STEPS = 4
N_CLASSES = len(common.COLOR)


def get_weights(data, key='speed', bins=4):
    if key == 'none':
        return [1 for _ in range(sum(len(x) for x in data))]
    elif key == 'even':
        values = np.hstack([[i for _ in range(len(x))] for i, x in enumerate(data)])
        bins = len(data)
    else:
        values = np.hstack(tuple(x.measurements[key].values[:len(x)] for x in data))
        values[np.isnan(values)] = np.mean(values[~np.isnan(values)])

    counts, edges = np.histogram(values, bins=bins)
    class_weights = counts.sum() / (counts + 1e-6)
    classes = np.digitize(values, edges[1:-1])

    print(counts)

    return class_weights[classes]


def get_dataset(dataset_dir, is_train=True, batch_size=128, num_workers=4, sample_by='none', **kwargs):
    print('\n'*3, 'dataset_dir', dataset_dir, '\n'*3)
    print('Path(dataset_dir).glob()', Path(dataset_dir).glob('*'))
    data = list()
    transform = transforms.Compose([
        get_augmenter() if is_train else lambda x: x,
        transforms.ToTensor()
        ])

    episodes = list(sorted(Path(dataset_dir).glob('*')))
    print('\n'*3, 'dataset_dir', dataset_dir, '\n'*3)
    print('\n'*3, 'episodes', episodes, '\n'*3)

    for i, _dataset_dir in enumerate(episodes):
        add = False
        # use all data to train
        add |= (is_train and i % 10 <= 9)
        add |= (not is_train and i % 10 == 0)

        if add:
            data.append(CarlaDataset(_dataset_dir, transform, **kwargs))
    print('data', data)
    # print('%d frames.' % sum(map(len, data)))

    weights = torch.DoubleTensor(get_weights(data, key=sample_by))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data = torch.utils.data.ConcatDataset(data)

    return Wrap(data, sampler, batch_size, 200 if is_train else 20, num_workers)


def get_augmenter():
    seq = iaa.Sequential([
        iaa.Sometimes(0.05, iaa.GaussianBlur((0.0, 1.3))),
        iaa.Sometimes(0.05, iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255))),
        iaa.Sometimes(0.05, iaa.Dropout((0.0, 0.1))),
        iaa.Sometimes(0.10, iaa.Add((-0.05 * 255, 0.05 * 255), True)),
        iaa.Sometimes(0.20, iaa.Add((0.25, 2.5), True)),
        iaa.Sometimes(0.05, iaa.contrast.LinearContrast((0.5, 1.5))),
        iaa.Sometimes(0.05, iaa.MultiplySaturation((0.0, 1.0))),
        ])

    return seq.augment_image


# https://github.com/guopei/PoseEstimation-FCN-Pytorch/blob/master/heatmap.py
def make_heatmap(size, pt, sigma=8):
    img = np.zeros(size, dtype=np.float32)
    pt = [
            np.clip(pt[0], sigma // 2, img.shape[1]-sigma // 2),
            np.clip(pt[1], sigma // 2, img.shape[0]-sigma // 2)
            ]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img


def preprocess_semantic(semantic_np):
    topdown = common.CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor()):
        dataset_dir = Path(dataset_dir)
        measurements = list(sorted((dataset_dir / 'measurements').glob('*.json')))

        self.transform = transform
        self.dataset_dir = dataset_dir
        self.frames = list()
        self.measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])
        self.converter = Converter()

        print(dataset_dir)

        for image_path in sorted((dataset_dir / 'rgb').glob('*.png')):
            frame = str(image_path.stem)

            assert (dataset_dir / 'rgb_left' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'rgb_right' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'topdown' / ('%s.png' % frame)).exists()
            assert int(frame) < len(self.measurements)

            self.frames.append(frame)

        assert len(self.frames) > 0, '%s has 0 frames.' % dataset_dir

    def __len__(self):
        return len(self.frames) - GAP * STEPS

    def __getitem__(self, i):
        path = self.dataset_dir
        frame = self.frames[i]
        meta = '%s %s' % (path.stem, frame)

        with Image.open(path / 'rgb' / ('%s.png' % frame)) as rgb_image:
            rgb = transforms.functional.to_tensor(rgb_image)
        with Image.open(path / 'rgb_left' / ('%s.png' % frame)) as rgb_left_image:
            rgb_left = transforms.functional.to_tensor(rgb_left_image)
        with Image.open(path / 'rgb_right' / ('%s.png' % frame)) as rgb_right_image:
            rgb_right = transforms.functional.to_tensor(rgb_right_image)
        with Image.open(path / 'topdown' / ('%s.png' % frame)) as topdown_image:
            topdown = topdown_image.crop((128, 0, 128 + 256, 256))
            topdown = np.array(topdown)
            topdown = preprocess_semantic(topdown)

        u = np.float32(self.measurements.iloc[i][['x', 'y']])
        theta = self.measurements.iloc[i]['theta']
        if np.isnan(theta):
            theta = 0.0
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        points = list()

        for skip in range(1, STEPS+1):
            j = i + GAP * skip
            v = np.array(self.measurements.iloc[j][['x', 'y']])

            target = R.T.dot(v - u)
            target *= PIXELS_PER_WORLD
            target += [128, 256]

            points.append(target)

        points = torch.FloatTensor(points)
        points = torch.clamp(points, 0, 256)
        points = (points / 256) * 2 - 1

        target = np.float32(self.measurements.iloc[i][['x_command', 'y_command']])
        target = R.T.dot(target - u)
        target *= PIXELS_PER_WORLD
        target += [128, 256]
        target = np.clip(target, 0, 256)
        target = torch.FloatTensor(target)

        # heatmap = make_heatmap((256, 256), target)
        # heatmap = torch.FloatTensor(heatmap).unsqueeze(0)

        # command_img = self.converter.map_to_cam(torch.FloatTensor(target))
        # heatmap_img = make_heatmap((144, 256), command_img)
        # heatmap_img = torch.FloatTensor(heatmap_img).unsqueeze(0)

        actions = np.float32(self.measurements.iloc[i][['steer', 'target_speed']])
        actions[np.isnan(actions)] = 0.0
        actions = torch.FloatTensor(actions)

        return torch.cat((rgb, rgb_left, rgb_right)), topdown, points, target, actions, meta


if __name__ == '__main__':
    import sys
    import cv2
    from PIL import ImageDraw
    from .utils.heatmap import ToHeatmap

    # for path in sorted(Path('/home/bradyzhou/data/carla/carla_challenge_curated').glob('*')):
        # data = CarlaDataset(path)

        # for i in range(len(data)):
            # data[i]

    data = CarlaDataset(sys.argv[1])
    converter = Converter()
    to_heatmap = ToHeatmap()

    for i in range(len(data)):
        rgb, topdown, points, target, actions, meta = data[i]
        points_unnormalized = (points + 1) / 2 * 256
        points_cam = converter(points_unnormalized)

        target_cam = converter(target)

        heatmap = to_heatmap(target[None], topdown[None]).squeeze()
        heatmap_cam = to_heatmap(target_cam[None], rgb[None]).squeeze()

        _heatmap = heatmap.cpu().squeeze().numpy() / 10.0 + 0.9
        _heatmap_cam = heatmap_cam.cpu().squeeze().numpy() / 10.0 + 0.9

        _rgb = (rgb.cpu() * 255).byte().numpy().transpose(1, 2, 0)[:, :, :3]
        _rgb[heatmap_cam > 0.1] = 255
        _rgb = Image.fromarray(_rgb)

        _topdown = common.COLOR[topdown.argmax(0).cpu().numpy()]
        _topdown[heatmap > 0.1] = 255
        _topdown = Image.fromarray(_topdown)
        _draw_map = ImageDraw.Draw(_topdown)
        _draw_rgb = ImageDraw.Draw(_rgb)

        for x, y in points_unnormalized:
            _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        for x, y in converter.cam_to_map(points_cam):
            _draw_map.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

        for x, y in points_cam:
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown.thumbnail(_rgb.size)

        cv2.imshow('debug', cv2.cvtColor(np.hstack((_rgb, _topdown)), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
