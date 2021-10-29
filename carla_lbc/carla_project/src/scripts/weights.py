import sys
# TBD: need to be made more portable
sys.path.insert(0, 'carla_project/src')
from dataset import get_dataset
import src.common as common
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image, ImageDraw


data = get_dataset('/home/bradyzhou/data/carla/challenge_local/CARLA_challenge_autopilot', False)
samples = set()

for rgb, topdown, points, _, action, meta in tqdm.tqdm(data):
    samples.update(meta)
    print(len(samples))

    for i in range(rgb.shape[0]):
        _topdown = Image.fromarray(common.COLOR[topdown[i].argmax(0).cpu().numpy()])
        _draw_map = ImageDraw.Draw(_topdown)

        for x, y in points[i]:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw_map.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        cv2.imshow('map', cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
