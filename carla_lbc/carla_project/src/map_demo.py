import sys

import cv2
import torch
import numpy as np

from PIL import Image, ImageDraw

from dataset import CarlaDataset
from converter import Converter
from map_model import MapModel
import src.common as common


net = MapModel.load_from_checkpoint(sys.argv[1])
net.cuda()
net.eval()

data = CarlaDataset(sys.argv[2])
converter = Converter()

for i in range(len(data)):
    rgb, topdown, points, heatmap, heatmap_img, meta = data[i]
    points_unnormalized = (points + 1) / 2 * 256
    points_cam = converter(points_unnormalized)
    heatmap_flipped = torch.FloatTensor(heatmap.numpy()[:, :, ::-1].copy())


    with torch.no_grad():
        points_pred = net(torch.cat((topdown, heatmap), 0).cuda()[None]).cpu().squeeze()
        points_pred_flipped = net(torch.cat((topdown, heatmap_flipped), 0).cuda()[None]).cpu().squeeze()

    _heatmap = np.uint8(heatmap.detach().cpu().squeeze().numpy() * 255)
    _heatmap_flipped = np.uint8(heatmap_flipped.squeeze().numpy() * 255)
    _heatmap_img = np.uint8(heatmap_img.detach().cpu().squeeze().numpy() * 255)
    _rgb = Image.fromarray(np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255))

    _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
    _draw_map = ImageDraw.Draw(_topdown)
    _draw_rgb = ImageDraw.Draw(_rgb)

    for x, y in (points_pred + 1) / 2 * 256:
        _draw_map.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

    for x, y in (points_pred_flipped + 1) / 2 * 256:
        _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

    # for x, y in points_unnormalized:
        # _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

    # for x, y in converter.cam_to_map(points_cam):
        # _draw_map.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

    # for x, y in points_cam:
        # _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

    canvas = np.hstack((np.stack([_heatmap_flipped] * 3, 2), _topdown, np.stack([_heatmap] * 3, 2)))

    Image.fromarray(canvas).save('/tmp/%04d.png' % i)

    cv2.imshow('heat_img', _heatmap_img)
    cv2.imshow('map', cv2.cvtColor(np.array(_rgb), cv2.COLOR_BGR2RGB))
    cv2.imshow('rgb', cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
