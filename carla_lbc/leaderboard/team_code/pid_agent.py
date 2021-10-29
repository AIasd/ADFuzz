import os
import time
import datetime
import pathlib
import sys
import math

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw




from carla_project.src.common import CONVERTER, COLOR
from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController


from enum import Enum
from agents.navigation.agent import AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.tools.misc import is_within_distance_ahead, is_within_distance, compute_distance

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
DEBUG = int(os.environ.get('HAS_DISPLAY', 0))





def get_entry_point():
    return "PIDAgent"


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))




class PIDAgent(MapAgent):
    def _init(self):
        super()._init()


        self._proximity_tlight_threshold = 5.0  # meters
        self._proximity_vehicle_threshold = 10.0  # meters
        self._local_planner = None

        try:
            self._map = self._world.get_map()

        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self._last_traffic_light = None



        self._proximity_tlight_threshold = 5.0  # meters
        self._proximity_vehicle_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING

        # TBD: subject to change
        self.target_speed = 7
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.4,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed' : self.target_speed,
            'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._grp = None

        self._map = CarlaDataProvider.get_map()
        route = [(self._map.get_waypoint(x[0].location), x[1]) for x in self._global_plan_world_coord]

        self._local_planner.set_global_plan(route)







    def _get_control(self, tick_data, _draw):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']






        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step(debug=DEBUG)








        _draw.text((5, 90), 'Speed: %.3f' % speed)
        _draw.text((5, 110), 'Target: %.3f' % self.target_speed)

        return control

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        data = self.tick(input_data)

        # visualization
        rgb_with_car = cv2.cvtColor(input_data['rgb_with_car'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        data['rgb_with_car'] = rgb_with_car

        topdown = data['topdown']
        rgb = np.hstack((data['rgb_left'], data['rgb'], data['rgb_right']))

        # gps = self._get_position(data)
        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _rgb = Image.fromarray(rgb)
        _draw = ImageDraw.Draw(_topdown)

        _topdown.thumbnail((256, 256))
        _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))

        _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
        _draw = ImageDraw.Draw(_combined)

        # change
        control = self._get_control(data, _draw)



        steer = control.steer
        throttle = control.throttle
        brake = control.brake

        _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
        _draw.text((5, 30), 'Steer: %.3f' % steer)
        _draw.text((5, 50), 'Throttle: %.3f' % throttle)
        _draw.text((5, 70), 'Brake: %s' % brake)

        if HAS_DISPLAY:
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)


        if self.step % 2 == 0:
            self.gather_info()

        record_every_n_steps = self.record_every_n_step
        if self.step % record_every_n_steps == 0:
            self.save('', steer, throttle, brake, self.target_speed, data)

        return control

    def save(self, far_command, steer, throttle, brake, target_speed, tick_data):
        # frame = self.step // 10
        frame = self.step

        speed = tick_data['speed']


        center = os.path.join('rgb', ('%04d.png' % frame))
        left = os.path.join('rgb_left', ('%04d.png' % frame))
        right = os.path.join('rgb_right', ('%04d.png' % frame))
        topdown = os.path.join('topdown', ('%04d.png' % frame))
        rgb_with_car = os.path.join('rgb_with_car', ('%04d.png' % frame))

        data_row = ','.join([str(i) for i in [frame, far_command, speed, steer, throttle, brake, str(center), str(left), str(right)]])
        with (self.save_path / 'measurements.csv').open("a") as f_out:
            f_out.write(data_row+'\n')


        Image.fromarray(tick_data['rgb']).save(self.save_path / center)
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / left)
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / right)
        # modification
        Image.fromarray(COLOR[CONVERTER[tick_data['topdown']]]).save(self.save_path / topdown)
        Image.fromarray(tick_data['rgb_with_car']).save(self.save_path / rgb_with_car)


        ########################################################################
        # log necessary info for action-based








    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        if not DEBUG:
            return

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0]+v[0], p[1]+v[1], z)
        color = carla.Color(*color)

        self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)








    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform,
                                        self._vehicle.get_transform(),
                                        self._proximity_tlight_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def _bh_is_vehicle_hazard(self, ego_wpt, ego_loc, vehicle_list,
                           proximity_th, up_angle_th, low_angle_th=0, lane_offset=0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle. We also check the next waypoint, just to be
        sure there's not a sudden road id change.

        WARNING: This method is an approximation that could fail for very large
        vehicles, which center is actually on a different lane but their
        extension falls within the ego vehicle lane. Also, make sure to remove
        the ego vehicle from the list. Lane offset is set to +1 for right lanes
        and -1 for left lanes, but this has to be inverted if lane values are
        negative.

            :param ego_wpt: waypoint of ego-vehicle
            :param ego_log: location of ego-vehicle
            :param vehicle_list: list of potential obstacle to check
            :param proximity_th: threshold for the agent to be alerted of
            a possible collision
            :param up_angle_th: upper threshold for angle
            :param low_angle_th: lower threshold for angle
            :param lane_offset: for right and left lane changes
            :return: a tuple given by (bool_flag, vehicle, distance), where:
            - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
            - vehicle is the blocker object itself
            - distance is the meters separating the two vehicles
        """

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        for target_vehicle in vehicle_list:

            target_vehicle_loc = target_vehicle.get_location()
            # If the object is not in our next or current lane it's not an obstacle

            target_wpt = self._map.get_waypoint(target_vehicle_loc)
            if target_wpt.road_id != ego_wpt.road_id or \
                    target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=5)[0]
                if target_wpt.road_id != next_wpt.road_id or \
                        target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                    continue

            if is_within_distance(target_vehicle_loc, ego_loc,
                                  self._vehicle.get_transform().rotation.yaw,
                                  proximity_th, up_angle_th, low_angle_th):

                return (True, target_vehicle, compute_distance(target_vehicle_loc, ego_loc))

        return (False, None, -1)

    def _is_vehicle_hazard(self, vehicle_list):
        """

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_vehicle_threshold):
                return (True, target_vehicle)

        return (False, None)


    @staticmethod
    def emergency_stop():
        """
        Send an emergency stop command to the vehicle

            :return: control for braking
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control
