import torch
import numpy as np


PIXELS_PER_WORLD = 5.5
HEIGHT = 144
WIDTH = 256
FOV = 90
MAP_SIZE = 256
CAM_HEIGHT = 1.3


class Converter(torch.nn.Module):
    def __init__(
            self, w=WIDTH, h=HEIGHT, fov=FOV,
            map_size=MAP_SIZE, pixels_per_world=PIXELS_PER_WORLD,
            hack=0.4, cam_height=CAM_HEIGHT):
        super().__init__()

        F = w / (2 * np.tan(fov * np.pi / 360))

        self.map_size = map_size
        self.pixels_per_world = pixels_per_world
        self.w = w
        self.h = h
        self.fy = F
        self.fx = 1.1 * F
        self.hack = hack
        self.cam_height = cam_height

        self.register_buffer('position', torch.FloatTensor([map_size // 2, map_size + 1]))

    def forward(self, map_coords):
        return self.map_to_cam(map_coords)

    def map_to_cam(self, map_coords):
        world_coords = self.map_to_world(map_coords)
        cam_coords = self.world_to_cam(world_coords)

        return cam_coords

    def map_to_world(self, pixel_coords):
        relative_pixel = pixel_coords - self.position
        relative_pixel[..., 1] *= -1

        return relative_pixel / self.pixels_per_world

    def cam_to_map(self, points):
        world_coords = self.cam_to_world(points)
        map_coords = self.world_to_map(world_coords)

        return map_coords

    def cam_to_world(self, points):
        z = (self.fy * self.cam_height) / (points[..., 1] - self.h / 2)
        x = (points[..., 0] - self.w / 2) * (z / self.fx)
        y = z - self.hack

        result = torch.stack([x, y], points.ndim-1)
        result = result.reshape(*points.shape)

        return result

    def world_to_cam(self, world):
        z = world[..., 1] + self.hack
        x = (self.fx * world[..., 0]) / z + self.w / 2
        y = (self.fy * self.cam_height) / z + self.h / 2

        result = torch.stack([x, y], world.ndim-1)
        result[..., 0] = torch.clamp(result[..., 0], 0, self.w-1)
        result[..., 1] = torch.clamp(result[..., 1], 0, self.h-1)
        result = result.reshape(*world.shape)

        return result

    def world_to_map(self, world):
        map_coord = world * self.pixels_per_world
        map_coord[..., 1] *= -1
        map_coord += self.position

        return map_coord
