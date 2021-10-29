import time

import cv2
import carla

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import numpy as np
from leaderboard.utils.route_manipulation import interpolate_trajectory

from carla_specific_utils.carla_specific import get_angle, norm_2d, get_bbox, angle_from_center_view_fov
from carla_specific_utils.carla_specific_tools import visualize_route
import os
import math
import pathlib

class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.record_every_n_step = 2000
        self.wall_start = time.time()
        self.initialized = False

        parent_folder = os.environ['SAVE_FOLDER']
        string = pathlib.Path(os.environ['ROUTES']).stem
        self.save_path = pathlib.Path(parent_folder) / string

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)
        self.initialized = True


        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._map = CarlaDataProvider.get_map()

        self.min_d = 10000
        self.offroad_d = 10000
        self.wronglane_d = 10000
        self.dev_dist = 0
        self.d_angle_norm = 1






    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': -0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': 0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    },
                # addition
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.0, 'y': 0.0, 'z': 100.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'map'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -6, 'y': 0.0, 'z': 3,
                    'roll': 0.0, 'pitch': -20.0, 'yaw': 0.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_with_car'
                    },
                {
                    'type': 'sensor.other.radar',
                    'x': 2, 'y': 0.0, 'z': 1,
                    'roll': 0.0, 'pitch': 5.0, 'yaw': 0.0,
                    'horizontal_fov': 35, 'vertical_fov': 20, 'range': 20,
                    'id': 'radar_central'
                    }
                ]

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        return {
                'rgb': rgb,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'gps': gps,
                'speed': speed,
                'compass': compass
                }



    # def set_trajectory(self, trajectory):
    #     self.trajectory = trajectory

    def set_args(self, args):
        self.deviations_path = os.path.join(args.deviations_folder, 'deviations.txt')
        self.args = args
        # print('\n'*10, 'self.args.record_every_n_step', self.args.record_every_n_step, '\n'*10)
        self.record_every_n_step = self.args.record_every_n_step


    def record_other_actor_info_for_causal_analysis(self, ego_control_and_speed_info):
        def get_loc_and_ori(agent):
            agent_tra = agent.get_transform()
            agent_loc = agent_tra.location
            agent_rot = agent_tra.rotation
            return agent_loc.x, agent_loc.y, agent_rot.yaw

        data_row = []
        if ego_control_and_speed_info:
            data_row += ego_control_and_speed_info

        x, y, yaw = get_loc_and_ori(self._vehicle)
        data_row += [x, y, yaw]

        other_actor_info_path = os.path.join(self.args.deviations_folder, 'other_actor_info.txt')

        actors = self._world.get_actors()
        vehicle_list = actors.filter('*vehicle*')
        pedestrian_list = actors.filter('*walker*')




        for i, pedestrian in enumerate(pedestrian_list):
            d_angle_norm = angle_from_center_view_fov(pedestrian, self._vehicle, fov=90)
            if d_angle_norm == 0:
                within_view = True
            else:
                within_view = False

            x, y, yaw = get_loc_and_ori(pedestrian)
            data_row.extend([x, y, yaw, within_view])

        for i, vehicle in enumerate(vehicle_list):
            if vehicle.id == self._vehicle.id:
                continue

            d_angle_norm = angle_from_center_view_fov(vehicle, self._vehicle, fov=90)
            if d_angle_norm == 0:
                within_view = True
            else:
                within_view = False

            x, y, yaw = get_loc_and_ori(vehicle)
            data_row.extend([x, y, yaw, within_view])

        with open(other_actor_info_path, 'a') as f_out:
            f_out.write(','.join([str(d) for d in data_row])+'\n')



    def gather_info(self, ego_control_and_speed_info=None):
        # if self.step % 1 == 0:
        #     self.record_other_actor_info_for_causal_analysis(ego_control_and_speed_info)


        ego_bbox = get_bbox(self._vehicle)
        ego_front_bbox = ego_bbox[:2]


        actors = self._world.get_actors()
        vehicle_list = actors.filter('*vehicle*')
        pedestrian_list = actors.filter('*walker*')

        min_d = 10000
        d_angle_norm = 1
        for i, vehicle in enumerate(vehicle_list):
            if vehicle.id == self._vehicle.id:
                continue

            d_angle_norm_i = angle_from_center_view_fov(vehicle, self._vehicle, fov=90)
            d_angle_norm = np.min([d_angle_norm, d_angle_norm_i])
            if d_angle_norm_i == 0:
                other_bbox = get_bbox(vehicle)
                for other_b in other_bbox:
                    for ego_b in ego_bbox:
                        d = norm_2d(other_b, ego_b)
                        # print('vehicle', i, 'd', d)
                        min_d = np.min([min_d, d])


        for i, pedestrian in enumerate(pedestrian_list):
            d_angle_norm_i = angle_from_center_view_fov(pedestrian, self._vehicle, fov=90)
            d_angle_norm = np.min([d_angle_norm, d_angle_norm_i])
            if d_angle_norm_i == 0:
                pedestrian_location = pedestrian.get_transform().location
                for ego_b in ego_front_bbox:
                    d = norm_2d(pedestrian_location, ego_b)
                    # print('pedestrian', i, 'd', d)
                    min_d = np.min([min_d, d])


        if min_d < self.min_d:
            self.min_d = min_d
            with open(self.deviations_path, 'a') as f_out:
                f_out.write('min_d,'+str(self.min_d)+'\n')


        if d_angle_norm < self.d_angle_norm:
            self.d_angle_norm = d_angle_norm
            with open(self.deviations_path, 'a') as f_out:
                f_out.write('d_angle_norm,'+str(self.d_angle_norm)+'\n')



        angle_th = 120

        current_location = CarlaDataProvider.get_location(self._vehicle)
        current_transform = CarlaDataProvider.get_transform(self._vehicle)
        current_waypoint = self._map.get_waypoint(current_location, project_to_road=False, lane_type=carla.LaneType.Any)
        ego_forward = current_transform.get_forward_vector()
        ego_forward = np.array([ego_forward.x, ego_forward.y])
        ego_forward /= np.linalg.norm(ego_forward)
        ego_right = current_transform.get_right_vector()
        ego_right = np.array([ego_right.x, ego_right.y])
        ego_right /= np.linalg.norm(ego_right)


        lane_center_waypoint = self._map.get_waypoint(current_location, lane_type=carla.LaneType.Any)
        lane_center_transform = lane_center_waypoint.transform
        lane_center_location = lane_center_transform.location
        lane_forward = lane_center_transform.get_forward_vector()
        lane_forward = np.array([lane_forward.x, lane_forward.y])
        lane_forward /= np.linalg.norm(lane_forward)
        lane_right = current_transform.get_right_vector()
        lane_right = np.array([lane_right.x, lane_right.y])
        lane_right /= np.linalg.norm(lane_right)



        dev_dist = current_location.distance(lane_center_location)
        # normalized to [0, 1]. 0 - same direction, 1 - opposite direction

        # print('ego_forward, lane_forward, np.dot(ego_forward, lane_forward)', ego_forward, lane_forward, np.dot(ego_forward, lane_forward))
        dev_angle = math.acos(np.clip(np.dot(ego_forward, lane_forward), -1, 1)) / np.pi
        # smoothing and integrate
        dev_dist *= (dev_angle + 0.5)

        if dev_dist > self.dev_dist:
            self.dev_dist = dev_dist
            with open(self.deviations_path, 'a') as f_out:
                f_out.write('dev_dist,'+str(self.dev_dist)+'\n')



        # print(current_location, current_waypoint.lane_type, current_waypoint.is_junction)
        # print(lane_center_location, lane_center_waypoint.lane_type, lane_center_waypoint.is_junction)

        def get_d(coeff, dir, dir_label):

            n = 1
            while n*coeff < 7:
                new_loc = carla.Location(current_location.x + n*coeff*dir[0], current_location.y + n*coeff*dir[1], 0)
                # print(coeff, dir, dir_label)
                # print(dir_label, 'current_location, dir, new_loc', current_location, dir, new_loc)
                new_wp = self._map.get_waypoint(new_loc,project_to_road=False, lane_type=carla.LaneType.Any)

                if not (new_wp and new_wp.lane_type in [carla.LaneType.Driving, carla.LaneType.Parking, carla.LaneType.Bidirectional] and np.abs(new_wp.transform.rotation.yaw%360 - lane_center_waypoint.transform.rotation.yaw%360) < angle_th):
                    # if new_wp and new_wp.lane_type in [carla.LaneType.Driving, carla.LaneType.Parking, carla.LaneType.Bidirectional]:
                    #     print('new_wp.transform.rotation.yaw, lane_center_waypoint.transform.rotation.yaw', new_wp.transform.rotation.yaw, lane_center_waypoint.transform.rotation.yaw)
                    break
                else:
                    n += 1
                # if new_wp:
                #     print(n, new_wp.transform.rotation.yaw)

            d = new_loc.distance(current_location)
            # print(d, new_loc, current_location)


            with open(self.deviations_path, 'a') as f_out:
                if new_wp and new_wp.lane_type in [carla.LaneType.Driving, carla.LaneType.Parking, carla.LaneType.Bidirectional]:
                    # print(dir_label, 'wronglane_d', d)
                    if d < self.wronglane_d:
                        self.wronglane_d = d
                        f_out.write('wronglane_d,'+str(self.wronglane_d)+'\n')
                        # print(dir_label, 'current_location, dir, new_loc', current_location, dir, new_loc, 'wronglane_d,'+str(self.wronglane_d)+'\n')
                else:
                    # if not new_wp:
                    #     s = 'None wp'
                    # else:
                    #     s = new_wp.lane_type
                    # print(dir_label, 'offroad_d', d, s, coeff)
                    # if new_wp:
                        # print(dir_label, 'lanetype', new_wp.lane_type)
                    if d < self.offroad_d:
                        self.offroad_d = d
                        f_out.write('offroad_d,'+str(self.offroad_d)+'\n')
                        # print(dir_label, 'current_location, dir, new_loc', current_location, dir, new_loc, 'offroad_d,'+str(self.offroad_d)+'\n')




        if current_waypoint and not current_waypoint.is_junction:
            get_d(-0.1, lane_right, 'left')
            get_d(0.1, lane_right, 'right')
        get_d(-0.1, ego_right, 'ego_left')
        get_d(0.1, ego_right, 'ego_right')
        get_d(0.1, ego_forward, 'ego_forward')
