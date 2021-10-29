import sys

from pathlib import Path

import numpy as np
import tqdm
import carla
import cv2
import pandas as pd

from PIL import Image

from carla_env import CarlaEnv
from common import COLOR


EPISODE_LENGTH = 1000
EPISODES = 10
FRAME_SKIP = 5
SAVE_PATH = Path('/home/bradyzhou/data/carla/topdown')
DEBUG = True


def collect_episode(env, save_dir):
    save_dir.mkdir()

    (save_dir / 'rgb_left').mkdir()
    (save_dir / 'rgb').mkdir()
    (save_dir / 'rgb_right').mkdir()
    (save_dir / 'map').mkdir()

    env._client.start_recorder(str(save_dir / 'recording.log'))

    spectator = env._world.get_spectator()
    spectator.set_transform(
            carla.Transform(
                env._player.get_location() + carla.Location(z=50),
                carla.Rotation(pitch=-90)))

    measurements = list()

    for step in tqdm.tqdm(range(EPISODE_LENGTH * FRAME_SKIP)):
        observations = env.step()

        if step % FRAME_SKIP != 0:
            continue

        index = step // FRAME_SKIP
        rgb = observations.pop('rgb')
        rgb_left = observations.pop('rgb_left')
        rgb_right = observations.pop('rgb_right')
        topdown = observations.pop('topdown')

        measurements.append(observations)

        if DEBUG:
            cv2.imshow('rgb', cv2.cvtColor(np.hstack((rgb_left, rgb, rgb_right)), cv2.COLOR_BGR2RGB))
            cv2.imshow('topdown', cv2.cvtColor(COLORS[topdown], cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        Image.fromarray(rgb_left).save(save_dir / 'rgb_left' / ('%04d.png' % index))
        Image.fromarray(rgb).save(save_dir / 'rgb' / ('%04d.png' % index))
        Image.fromarray(rgb_right).save(save_dir / 'rgb_right' / ('%04d.png' % index))
        Image.fromarray(topdown).save(save_dir / 'map' / ('%04d.png' % index))

    pd.DataFrame(measurements).to_csv(save_dir / 'measurements.csv', index=False)

    env._client.stop_recorder()


def main():
    np.random.seed(1337)

    for i in range(1, 8):
        with CarlaEnv(town='Town0%s' % i) as env:
            for episode in range(EPISODES):
                env.reset(
                        n_vehicles=np.random.choice([50, 100, 200]),
                        n_pedestrians=np.random.choice([50, 100, 200]),
                        seed=np.random.randint(0, 256))
                env._player.set_autopilot(True)

                collect_episode(env, SAVE_PATH / ('%03d' % len(list(SAVE_PATH.glob('*')))))


if __name__ == '__main__':
    main()
