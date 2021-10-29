import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw

from carla_project.src.common import CONVERTER, COLOR
from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController

from carla_specific_utils.carla_specific import norm_2d, get_bbox
import json

from object_types import WEATHERS

HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
DEBUG = int(os.environ.get('HAS_DISPLAY', 0))



def get_entry_point():
    return 'AutoPilot'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


class AutoPilot(MapAgent):
    def _init(self):
        super()._init()

        self._default_target_speed = 7.0
        self._default_slow_target_speed = 4.0
        self._angle = None
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _get_control(self, target, far_target, tick_data, _draw):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90
        self._angle = np.radians(angle_unnorm)
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = self._default_slow_target_speed if should_slow else self._default_target_speed

        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        self._target_speed = target_speed

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        _draw.text((5, 90), 'Speed: %.3f' % speed)
        _draw.text((5, 110), 'Target: %.3f' % target_speed)
        _draw.text((5, 130), 'Angle: %.3f' % angle_unnorm)
        _draw.text((5, 150), 'Angle Far: %.3f' % angle_far_unnorm)

        return steer, throttle, brake, target_speed

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        #     self._world.set_weather(WEATHERS[int(os.environ['WEATHER_INDEX'])])

        # 100 -> 50 since 20Hz -> 10Hz
        if self.step % 50 == 0 and self.args.changing_weather:
            index = np.random.randint(0, len(WEATHERS))
            self._world.set_weather(WEATHERS[index])

        data = self.tick(input_data)
        rgb_with_car = cv2.cvtColor(input_data['rgb_with_car'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        data['rgb_with_car'] = rgb_with_car

        topdown = data['topdown']
        rgb = np.hstack((data['rgb_left'], data['rgb'], data['rgb_right']))

        gps = self._get_position(data)

        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)



        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _rgb = Image.fromarray(rgb)
        _draw = ImageDraw.Draw(_topdown)

        _topdown.thumbnail((256, 256))
        _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))

        _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
        _draw = ImageDraw.Draw(_combined)

        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data, _draw)

        _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
        _draw.text((5, 30), 'Steer: %.3f' % steer)
        _draw.text((5, 50), 'Throttle: %.3f' % throttle)
        _draw.text((5, 70), 'Brake: %s' % brake)

        if HAS_DISPLAY:
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        # we only gether info every 2 frames for faster processing speed
        if self.step % 1 == 0:
            self.gather_info()


        # if this number is very small, we may not have the exact numbers and images for the event happening (e.g. the frame when a collision happen). However, this is usually ok if we only use these for retraining purpose
        record_every_n_steps = self.record_every_n_step
        if self.step % record_every_n_steps == 0:
            self.save(record_every_n_steps, far_command, steer, throttle, brake, target_speed, data)
            self.save_json(record_every_n_steps, far_node, near_command, steer, throttle, brake, target_speed, data)

        return control


    def save_json(self, record_every_n_steps, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = int(self.step // record_every_n_steps)


        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # pos, , far_node, near_command
        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': target_speed,
                'x_command': far_node[0],
                'y_command': far_node[1],
                'command': near_command.value,
                'steer': steer,
                'throttle': throttle,
                'brake': brake,
        }

        pth = self.save_path / 'measurements'
        pth.mkdir(parents=False, exist_ok=True)
        (pth / ('%04d.json' % frame)).write_text(str(data))

    def save(self, record_every_n_steps, far_command, steer, throttle, brake, target_speed, tick_data):
        frame = int(self.step // record_every_n_steps)



        speed = tick_data['speed']

        center = os.path.join('rgb', ('%06d.png' % frame))
        left = os.path.join('rgb_left', ('%06d.png' % frame))
        right = os.path.join('rgb_right', ('%06d.png' % frame))



        topdown = os.path.join('topdown', ('%06d.png' % frame))
        rgb_with_car = os.path.join('rgb_with_car', ('%06d.png' % frame))

        data_row = ','.join([str(i) for i in [frame, far_command, speed, steer, throttle, brake, str(center), str(left), str(right)]])
        with (self.save_path / 'measurements.csv').open("a") as f_out:
            f_out.write(data_row+'\n')


        Image.fromarray(tick_data['rgb']).save(self.save_path / center)
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / left)
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / right)
        # modification
        # Image.fromarray(COLOR[CONVERTER[tick_data['topdown']]]).save(topdown)
        Image.fromarray(tick_data['topdown']).save(self.save_path / topdown)
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

            current_affordances = get_driving_affordances(self, self._pedestrian_forbidden_distance, self._pedestrian_max_detected_distance, self._vehicle_forbidden_distance, self._vehicle_max_detected_distance, self._tl_forbidden_distance, self._tl_max_detected_distance, self._angle, self._default_target_speed, self._target_speed, self._speed_detected_distance, angle=True)

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

            # relative_angle
            # lane_center_waypoint = self._map.get_waypoint(ego_location, lane_type=carla.LaneType.Any)
            # lane_center_transform = lane_center_waypoint.transform
            #
            #
            # relative_angle = np.abs(lane_center_transform.rotation.yaw - ego_rotation.yaw)
            # if lane_center_transform.rotation.yaw - ego_rotation.yaw > 180:
            #     relative_angle = np.abs(lane_center_transform.rotation.yaw - ego_rotation.yaw - 360)
            # elif ego_rotation.yaw - lane_center_transform.rotation.yaw > 180:
            #     relative_angle = np.abs(lane_center_transform.rotation.yaw - ego_rotation.yaw + 360)
            #
            # relative_angle = np.radians(relative_angle)


            # vehicle, pedestrian, traffic light
            # actors = self._world.get_actors()
            # vehicle_list = actors.filter('*vehicle*')
            # pedestrian_list = actors.filter('walker*')
            # tls = actors.filter('*traffic_light*')
            #
            # ego_bbox = get_bbox(self._vehicle)
            #
            # closest_vehicle_distance = 50
            # for i, vehicle in enumerate(vehicle_list):
            #     if vehicle.id == self._vehicle.id:
            #         continue
            #     other_bbox = get_bbox(vehicle)
            #     for other_b in other_bbox:
            #         for ego_b in ego_bbox:
            #             d = norm_2d(other_b, ego_b)
            #             # print('vehicle', i, 'd', d)
            #             closest_vehicle_distance = np.min([closest_vehicle_distance, d])
            #
            # closest_pedestrian_distance = 50
            # for i, pedestrian in enumerate(pedestrian_list):
            #     other_bbox = get_bbox(pedestrian)
            #     for other_b in other_bbox:
            #         for ego_b in ego_bbox:
            #             d = norm_2d(other_b, ego_b)
            #             # print('pedestrian', i, 'd', d)
            #             closest_pedestrian_distance = np.min([closest_pedestrian_distance, d])
            #
            # closest_red_tl_distance = 50
            # is_red_tl_hazard = False
            # for tl in tls:
            #     if tl.state == carla.TrafficLightState.Red:
            #         tl_location = tl.get_transform().location
            #         d = norm_2d(ego_location, tl_location)
            #         closest_red_tl_distance = np.min([closest_red_tl_distance, d])
            #
            #         if d < 10:
            #             affecting = self._vehicle.get_traffic_light()
            #             if affecting and tl.id == affecting.id:
            #                 is_red_tl_hazard = True
            #
            # is_pedestrian_hazard = bool(closest_pedestrian_distance < 10)
            # is_vehicle_hazard = bool(closest_vehicle_distance < 10)

            brake_noise = 0.0
            throttle_noise = 0.0 # 1.0 -> 0.0
            steer_noise = 0.0 # NaN -> 0.0

            # class RoadOption
            # VOID = -1
            # LEFT = 1
            # RIGHT = 2
            # STRAIGHT = 3
            # LANEFOLLOW = 4
            # CHANGELANELEFT = 5
            # CHANGELANERIGHT = 6
            map_roadoption_to_action_based_roadoption = {-1:2, 1:3, 2:4, 3:5, 4:2, 5:2, 6:2}
            # print('\n far_command', far_command)
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


    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))

        return any(x is not None for x in [vehicle, light, walker])

    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        if not DEBUG:
            return

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0]+v[0], p[1]+v[1], z)
        color = carla.Color(*color)

        self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        # self._draw_line(p1, v1, z+2.5, (0, 0, 255))

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            # self._draw_line(p2, v2, z+2.5)

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
        v1_hat = o1
        v1 = s1 * v1_hat

        # self._draw_line(p1, v1, z+2.5, (255, 0, 0))

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            # self._draw_line(p2, v2, z+2.5, (255, 0, 0))

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None
