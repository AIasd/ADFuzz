import sys
import queue
import tqdm
import torchvision
import numpy as np
import open3d as o3d

import carla
import cv2

from carla_env import Camera


class Lidar(object):
    def __init__(self, world, player, height=2.0, channels=32, range=5000, bins=64):
        bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', str(channels))
        bp.set_attribute('range', str(range))

        loc = carla.Location(x=0.0, z=height)
        transform = carla.Transform(loc)

        self.queue = queue.Queue()

        self.lidar = world.spawn_actor(bp, transform, attach_to=player)
        self.lidar.listen(self.queue.put)
        self.bins = bins

        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()
        self.start = True
        self.pcd = o3d.geometry.PointCloud()

    def get(self):
        lidar = None

        while lidar is None or self.queue.qsize() > 0:
            lidar = self.queue.get()

        point_cloud = np.array([[loc.x, loc.y, loc.z] for loc in lidar])
        k = 50
        mask = True
        mask = np.logical_and(mask, point_cloud[..., 0] > -k)
        mask = np.logical_and(mask, point_cloud[..., 0] <  k)
        mask = np.logical_and(mask, point_cloud[..., 1] > -k)
        mask = np.logical_and(mask, point_cloud[..., 1] <  k)
        point_cloud = point_cloud[mask]
        self.pcd.points = o3d.utility.Vector3dVector(point_cloud)

        if self.start:
            self.visualizer.add_geometry(self.pcd)
            self.start = False
        else:
            self.visualizer.update_geometry(self.pcd)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()


        # y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
        # x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
        # z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]

        # result, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        # result[:, :, 0] = np.array(result[:, :, 0]>0, dtype=np.uint8)
        # result[:, :, 1] = np.array(result[:, :, 1]>0, dtype=np.uint8)

        # return result

    def __del__(self):
        self.lidar.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


class Rollout(object):
    def __init__(self, replay_file, port=2000):
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(60.0)

        set_sync_mode(self.client, True)

        self.client.set_replayer_time_factor(1.0)
        self.client.set_replayer_ignore_hero(False)

        print(self.client.replay_file(replay_file, 0.0, 100.0, 0))

        self.world = self.client.get_world()
        self.cameras = dict()
        self.lidars = dict()

    def step(self):
        self.world.tick()

        images = {k: v.get() for k, v in self.cameras.items()}
        lidars = {k: v.get() for k, v in self.lidars.items()}

        return {
                'img': images,
                'lidar': lidars
                }

    def __enter__(self):
        set_sync_mode(self.client, True)

        for _ in range(10):
            self.step()

        vehicles = self.world.get_actors().filter('*vehicle*')

        # for i in range(16):
            # self.cameras[i] = Camera(self.world, vehicles[i], 256, 144, 90, 1.2, 0.0, 1.3, 0.0, 0.0)

        self.cameras[16] = Camera(self.world, vehicles[16], 256, 144, 90, 1.2, 0.0, 1.3, 0.0, 0.0)
        self.lidars[16] = Lidar(self.world, vehicles[16])

        return self

    def __exit__(self, *args):
        self.cameras.clear()
        self.lidars.clear()

        set_sync_mode(self.client, False)


if __name__ == '__main__':
    rollout = Rollout(sys.argv[1])

    with rollout as env:
        for _ in tqdm.tqdm(range(1000)):
            data = env.step()

            # images = [torchvision.transforms.functional.to_tensor(v.copy()) for k, v in sorted(data['img'].items())]
            # images = (torchvision.utils.make_grid(images, nrow=3) * 255).byte().numpy().transpose(1, 2, 0)

            cv2.imshow('grid', cv2.cvtColor(data['img'][16], cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
