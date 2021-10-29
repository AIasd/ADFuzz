import os
import numpy as np
import cv2
import torch
import torchvision
import carla

from PIL import Image, ImageDraw

from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter

from team_code.base_agent import BaseAgent
from team_code.pid_controller import PIDController
# addition
import datetime
import pathlib

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))

# addition
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights
from carla_project.src.common import CONVERTER, COLOR
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from local_planner import LocalPlanner
import json
def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
    # modification

    # rgb = np.hstack((tick_data['rgb_left'], tick_data['rgb'], tick_data['rgb_right']))


    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 255, 255))

    for x, y in out:
        x = (x + 1) / 2 * 256
        y = (y + 1) / 2 * 144

    _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

    _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))

    _combined = _combined.resize((int(256 / _combined.size[1] * _combined.size[0]), 256))
    _topdown = Image.fromarray(COLOR[CONVERTER[tick_data['topdown']]])
    _topdown.thumbnail((256, 256))
    _combined = Image.fromarray(np.hstack((_combined, _topdown)))


    _draw = ImageDraw.Draw(_combined)
    _draw.text((5, 10), 'Steer: %.3f' % steer)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle)
    _draw.text((5, 50), 'Brake: %s' % brake)
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'])
    _draw.text((5, 90), 'Desired: %.3f' % desired_speed)
    _draw.text((5, 110), 'Far Command: %s' % str(tick_data['far_command']))

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()




    # addition: modified from leaderboard/team_code/auto_pilot.py
    def save(self, steer, throttle, brake, tick_data):
        # frame = self.step // 10
        frame = self.step

        pos = self._get_position(tick_data)
        far_command = tick_data['far_command']
        speed = tick_data['speed']



        center = os.path.join('rgb', ('%04d.png' % frame))
        left = os.path.join('rgb_left', ('%04d.png' % frame))
        right = os.path.join('rgb_right', ('%04d.png' % frame))
        topdown = os.path.join('topdown', ('%04d.png' % frame))
        rgb_with_car = os.path.join('rgb_with_car', ('%04d.png' % frame))

        data_row = ','.join([str(i) for i in [frame, far_command, speed, steer, throttle, brake, str(center), str(left), str(right)]])
        with (self.save_path / 'measurements.csv' ).open("a") as f_out:
            f_out.write(data_row+'\n')


        Image.fromarray(tick_data['rgb']).save(self.save_path / center)
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / left)
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / right)

        # addition
        Image.fromarray(COLOR[CONVERTER[tick_data['topdown']]]).save(self.save_path / topdown)
        Image.fromarray(tick_data['rgb_with_car']).save(self.save_path / rgb_with_car)



        ########################################################################
        # log necessary info for action-based
        if self.args.save_action_based_measurements:
            from affordances import get_driving_affordances

            self._pedestrian_forbidden_distance = 10.0
            self._pedestrian_max_detected_distance = 50.0
            self._vehicle_forbidden_distance = 10.0
            self._vehicle_max_detected_distance = 50.0
            self._tl_forbidden_distance = 10.0
            self._tl_max_detected_distance = 50.0
            self._speed_detected_distance = 10.0

            self._default_target_speed = 10
            self._angle = None

            current_affordances = get_driving_affordances(self, self._pedestrian_forbidden_distance, self._pedestrian_max_detected_distance, self._vehicle_forbidden_distance, self._vehicle_max_detected_distance, self._tl_forbidden_distance, self._tl_max_detected_distance, self._angle_rad, self._default_target_speed, self._target_speed, self._speed_detected_distance, angle=True)

            is_vehicle_hazard = current_affordances['is_vehicle_hazard']
            is_red_tl_hazard = current_affordances['is_red_tl_hazard']
            is_pedestrian_hazard = current_affordances['is_pedestrian_hazard']
            forward_speed = current_affordances['forward_speed']
            relative_angle = current_affordances['relative_angle']
            target_speed = current_affordances['target_speed']
            closest_pedestrian_distance = current_affordances['closest_pedestrian_distance']
            closest_vehicle_distance = current_affordances['closest_vehicle_distance']
            closest_red_tl_distance = current_affordances['closest_red_tl_distance']



            log_folder = str(self.save_path / 'affordance_measurements')
            if not os.path.exists(log_folder):
                os.mkdir(log_folder)
            log_path = os.path.join(log_folder, f'{self.step:06}.json')


            ego_transform = self._vehicle.get_transform()
            ego_location = ego_transform.location
            ego_rotation = ego_transform.rotation
            ego_velocity = self._vehicle.get_velocity()

            brake_noise = 0.0
            throttle_noise = 0.0 # 1.0 -> 0.0
            steer_noise = 0.0 # NaN -> 0.0

            # class RoadOption
            map_roadoption_to_action_based_roadoption = {-1:2, 1:3, 2:4, 3:5, 4:2, 5:2, 6:2}

            # save info for action-based rep
            json_log_data = {
                "brake": float(brake),
                "closest_red_tl_distance": closest_red_tl_distance,
                "throttle": throttle,
                "directions": float(map_roadoption_to_action_based_roadoption[far_command.value]),
                "brake_noise": brake_noise,
                "is_red_tl_hazard": is_red_tl_hazard,
                "opponents": {},
                "closest_pedestrian_distance": closest_pedestrian_distance,
                "is_pedestrian_hazard": is_pedestrian_hazard,
                "lane": {},
                "is_vehicle_hazard": is_vehicle_hazard,
                "throttle_noise": throttle_noise,
                "ego_actor": {
                    "velocity": [
                        ego_velocity.x,
                        ego_velocity.y,
                        ego_velocity.z
                    ],
                    "position": [
                        ego_location.x,
                        ego_location.y,
                        ego_location.z
                    ],
                    "orientation": [
                        ego_rotation.roll,
                        ego_rotation.pitch,
                        ego_rotation.yaw
                    ]
                },
                "hand_brake": False,
                "steer_noise": steer_noise,
                "reverse": False,
                "relative_angle": relative_angle,
                "closest_vehicle_distance": closest_vehicle_distance,
                "walkers": {},
                "forward_speed": forward_speed,
                "steer": steer,
                "target_speed": target_speed
            }

            with open(log_path, 'w') as f_out:
                json.dump(json_log_data, f_out, indent=4)


    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # addition:
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()


        # -------------------------------------------------------
        # add a local_planner in order to estimate relative angle
        # self.target_speed = 10
        # args_lateral_dict = {
        #     'K_P': 1,
        #     'K_D': 0.4,
        #     'K_I': 0,
        #     'dt': 1.0/10.0}
        # self._local_planner = LocalPlanner(
        #     self._vehicle, opt_dict={'target_speed' : self.target_speed,
        #     'lateral_control_dict':args_lateral_dict})
        # self._hop_resolution = 2.0
        # self._path_seperation_hop = 2
        # self._path_seperation_threshold = 0.5
        # self._grp = None
        #
        # self._map = CarlaDataProvider.get_map()
        # route = [(self._map.get_waypoint(x[0].location), x[1]) for x in self._global_plan_world_coord]
        #
        # self._local_planner.set_global_plan(route)


    def tick(self, input_data):



        result = super().tick(input_data)
        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)


        rgb_with_car = cv2.cvtColor(input_data['rgb_with_car'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        result['rgb_with_car'] = rgb_with_car

        result['radar_central'] = input_data['radar_central']

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        gps = self._get_position(result)
        # modification
        far_node, far_command = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target
        # addition:
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        result['far_command'] = far_command
        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        result['topdown'] = topdown


        return result

    @torch.no_grad()
    def run_step_using_learned_controller(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        import random
        torch.manual_seed(2)
        torch.cuda.manual_seed_all(2)
        torch.backends.cudnn.deterministic = True
        np.random.seed(1)
        random.seed(1)
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = False

        points, (target_cam, _) = self.net.forward(img, target)
        control = self.net.controller(points).cpu().squeeze()

        steer = control[0].item()
        desired_speed = control[1].item()
        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step)

        return control

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)
        radar_data = tick_data['radar_central'][1]



        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()





        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, (target_cam, _) = self.net.forward(img, target)
        points_cam = points.clone().cpu()

        if self.step  == 0:
            print('\n'*5)
            print('step :', self.step)
            # print('radar')
            # print(radar_data.shape)
            # print(radar_data)
            # print(np.max(radar_data, axis=0))
            print('image', np.sum(tick_data['image']))
            # print('img', torch.sum(img))
            # print('target', target)
            # print('points', points)
            print('\n'*5)

        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam).numpy()


        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0]))
        self._angle_rad = np.radians(angle)
        angle = angle / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2
        self._target_speed = desired_speed
        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(tick_data, target_cam.squeeze(), points.cpu().squeeze(), steer, throttle, brake, desired_speed, self.step)

        # addition: from leaderboard/team_code/auto_pilot.py
        if self.step == 0:
            title_row = ','.join(['frame_id', 'far_command', 'speed', 'steering', 'throttle', 'brake', 'center', 'left', 'right'])
            with (self.save_path / 'measurements.csv' ).open("a") as f_out:
                f_out.write(title_row+'\n')
        if self.step % 1 == 0:
            self.gather_info((steer, throttle, float(brake), speed))

        record_every_n_steps = self.record_every_n_step
        if self.step % record_every_n_steps == 0:
            self.save(steer, throttle, brake, tick_data)
        return control
