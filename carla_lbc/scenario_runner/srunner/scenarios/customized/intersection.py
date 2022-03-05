#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right or left turn.
"""
import math
import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter, ActorDestroy, KeepVelocity, HandBrakeVehicle, StopVehicle, WaypointFollower, AccelerateToVelocity)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute, InTriggerDistanceToVehicle, DriveDistance, InTriggerDistanceToLocation)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint, generate_target_waypoint_in_route

from leaderboard.utils.route_manipulation import interpolate_trajectory, downsample_route
import numpy as np
from collections import OrderedDict
import pickle
import os

from carla_specific_utils.carla_specific_tools import perturb_route, add_transform, create_transform, copy_transform
from customized_utils import make_hierarchical_dir

def get_generated_transform(added_dist, waypoint):
    """
    Calculate the transform of the adversary
    """
    if added_dist == 0:
        return waypoint.transform

    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    return _wp.transform



class Intersection(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a turn. This is the version used when the ego vehicle
    is following a given route. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60, customized_data=None):
        """
        Setup all relevant parameters and create scenario
        """
        self.world = world
        self.customized_data = customized_data

        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._num_lane_changes = 0


        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        if 'number_of_attempts_to_request_actor' in customized_data:
            self._number_of_attempts = customized_data['number_of_attempts_to_request_actor']
        else:
            self._number_of_attempts = 10

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()


        self.static_list = []
        self.pedestrian_list = []
        self.vehicle_list = []
        if 'tmp_travel_dist_file' in self.customized_data and self.customized_data['tmp_travel_dist_file'] and  os.path.exists(self.customized_data['tmp_travel_dist_file']):
            os.remove(self.customized_data['tmp_travel_dist_file'])
            print('remove tmp_travel_dist_file')

        super(Intersection, self).__init__("Intersection",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)




    def _request_actor(self, actor_category, actor_model, waypoint_transform, simulation_enabled=False, color=None, bounds=None, is_waypoint_follower=None, center_transform=None):
        def bound_xy(generated_transform, bounds):
            if bounds:
                x_min, x_max, y_min, y_max = bounds
                g_x = generated_transform.location.x
                g_y = generated_transform.location.y
                if center_transform:
                    c_x = center_transform.location.x
                    c_y = center_transform.location.y
                    x_min += c_x
                    x_max += c_x
                    y_min += c_y
                    y_max += c_y
                generated_transform.location.x = np.max([g_x, x_min])
                generated_transform.location.x = np.min([g_x, x_max])
                generated_transform.location.y = np.max([g_y, y_min])
                generated_transform.location.y = np.min([g_y, y_max])
                # print('bounds', x_min, x_max, y_min, y_max, g_x, g_y, center_transform)

        # If we fail too many times, this will break and the session will be assigned the lowest default score. We do this to disencourage samples that result in invalid locations

        # Number of attempts made so far
        status = 'success'
        is_success = False

        generated_transform = copy_transform(waypoint_transform)


        if actor_category != 'vehicle':
            g_x = generated_transform.location.x
            g_y = generated_transform.location.y
            g_yaw = generated_transform.rotation.yaw
            for i in range(self._number_of_attempts):
                for j in range(10):
                    try:
                        added_dist = i*0.25
                        cur_x = g_x + np.random.uniform(0, added_dist)
                        cur_y = g_y + np.random.uniform(0, added_dist)
                        cur_t = create_transform(cur_x, cur_y, 0, 0, g_yaw, 0)
                        # generated_transform.location.y += np.random.uniform(0, 1)
                        # bound_xy(generated_transform, bounds)
                        actor_object = CarlaDataProvider.request_new_actor(
                            model=actor_model, spawn_point=cur_t, color=color, actor_category=actor_category)
                        is_success = True
                        break
                    except (RuntimeError, AttributeError) as r:
                        status = 'fail_1_'+str(i)
                if is_success:
                    break

        if actor_category == 'vehicle' or status == 'fail_1_'+str(self._number_of_attempts):


            for i in range(self._number_of_attempts):
                try:
                    added_dist = i*0.25
                    waypoint = self._wmap.get_waypoint(waypoint_transform.location, project_to_road=True, lane_type=carla.LaneType.Any)
                    generated_transform = get_generated_transform(added_dist, waypoint)
                    # if actor_category == 'vehicle' and is_waypoint_follower:
                    generated_transform.rotation.yaw = waypoint_transform.rotation.yaw

                    bound_xy(generated_transform, bounds)
                    actor_object = CarlaDataProvider.request_new_actor(
                        model=actor_model, spawn_point=generated_transform, color=color, actor_category=actor_category)

                    is_success = True
                    break
                except (RuntimeError, AttributeError) as r:
                    status = 'fail_2_'+str(i)


        if is_success:
            actor_object.set_simulate_physics(enabled=simulation_enabled)
        else:
            actor_object = None
            generated_transform = None
            status = 'fail_all'


        if status != 'success' and is_success:
            print('\n', '{} {} {} ({:.2f},{:.2f},{:.2f})->({:.2f},{:.2f},{:.2f})'.format(status, actor_model, is_waypoint_follower, waypoint_transform.location.x, waypoint_transform.location.y, waypoint_transform.rotation.yaw, generated_transform.location.x, generated_transform.location.y, waypoint_transform.rotation.yaw), '\n')
        else:
            print('\n', status, actor_model, is_waypoint_follower, waypoint_transform, '\n')

        return actor_object, generated_transform

    def _initialize_actors(self, config):
        """
        Custom initialization

        static_center_transforms, static_center_transforms, vehicle_center_transforms:
        {i:(x_i, y_i)}
        """

        def spawning_actors_within_bounds(object_type):
            final_generated_transforms = []
            if object_type == 'static':
                object_list = self.static_list
            elif object_type == 'pedestrian':
                object_list = self.pedestrian_list
            elif object_type == 'vehicle':
                object_list = self.vehicle_list

            for i, object_i in enumerate(self.customized_data[object_type+'_list']):
                if 'add_center' in self.customized_data and self.customized_data['add_center']:
                    key = object_type+'_center_transform'+'_'+str(i)
                    if key in self.customized_data:
                        center_transform = self.customized_data[key]
                    else:
                        center_transform = self.customized_data['center_transform']


                    # hack: only x, y should be added
                    center_transform.location.z = 0
                    center_transform.rotation.pitch = 0
                    center_transform.rotation.yaw = 0
                    center_transform.rotation.roll = 0

                    spawn_transform_i = add_transform(center_transform, object_i.spawn_transform)
                    print(object_type, i, object_i.model, 'add center', object_i.spawn_transform, '->', spawn_transform_i)
                else:
                    spawn_transform_i = object_i.spawn_transform
                    center_transform = None

                if 'parameters_max_bounds' in self.customized_data.keys():
                    bounds = [self.customized_data[k1][object_type+'_'+k2+'_'+str(i)] for k1, k2 in [('parameters_min_bounds', 'x_min'), ('parameters_max_bounds', 'x_max'), ('parameters_min_bounds', 'y_min'), ('parameters_max_bounds', 'y_max')]]
                    # bounds = []
                else:
                    bounds = []

                if object_type == 'vehicle' and hasattr(object_i, 'color') and object_i.model != 'vehicle.tesla.cybertruck':
                    color = object_i.color
                else:
                    color = None

                if object_type == 'vehicle':
                    is_waypoint_follower = object_i.waypoint_follower
                else:
                    is_waypoint_follower = None

                if object_type == 'pedestrian':
                    simulation_enabled = False
                else:
                    simulation_enabled = True
                actor, generated_transform = self._request_actor(object_type, object_i.model, spawn_transform_i, simulation_enabled, color, bounds, is_waypoint_follower, center_transform)

                if actor and generated_transform:
                    object_list.append((actor, generated_transform))

                    gx = generated_transform.location.x
                    gy = generated_transform.location.y
                    gyaw = generated_transform.rotation.yaw

                    if 'add_center' in self.customized_data and self.customized_data['add_center']:
                        cx = center_transform.location.x
                        cy = center_transform.location.y
                    else:
                        cx = 0
                        cy = 0

                    final_generated_transforms.append((gx - cx, gy - cy, gyaw))
                else:
                    final_generated_transforms.append((None, None, None))


            # print(object_type, 'saved final generated transform', final_generated_transforms)
            return final_generated_transforms



        all_final_generated_transforms = OrderedDict()
        for object_type in ['static', 'pedestrian', 'vehicle']:
            final_generated_transforms = spawning_actors_within_bounds(object_type)

            all_final_generated_transforms[object_type] = final_generated_transforms

        # hack:
        tmp_folder = make_hierarchical_dir(['carla_lbc', 'tmp_folder'])
        filename = os.path.join(tmp_folder, str(config.cur_server_port)+'.pickle')

        with open(filename, 'wb') as f_out:
            pickle.dump(all_final_generated_transforms, f_out)






    def _create_behavior(self):
        """
        """
        def record_travel_dist(tmp_travel_dist_file, actor_id, actor_general_type, index):
            with open(tmp_travel_dist_file, 'a') as f_out:
                f_out.write(','.join([str(actor_id), actor_general_type, str(index)])+'\n')

        # building the tree
        scenario_sequence = py_trees.composites.Sequence()
        waypoint_events = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        destroy_actors = py_trees.composites.Sequence()


        reach_destination = InTriggerDistanceToLocation(self.ego_vehicles[0], self.customized_data['destination'], 2)

        scenario_sequence.add_child(waypoint_events)
        scenario_sequence.add_child(reach_destination)
        scenario_sequence.add_child(destroy_actors)



        tmp_travel_dist_file = None
        if 'tmp_travel_dist_file' in self.customized_data:
            tmp_travel_dist_file = self.customized_data['tmp_travel_dist_file']

        for i in range(len(self.pedestrian_list)):
            pedestrian_actor, pedestrian_generated_transform = self.pedestrian_list[i]
            pedestrian_info = self.customized_data['pedestrian_list'][i]

            if tmp_travel_dist_file:
                record_travel_dist(self.customized_data['tmp_travel_dist_file'], pedestrian_actor.id, 'pedestrian', i)
                print('record_travel_dist tmp_travel_dist_file')

            trigger_distance = InTriggerDistanceToVehicle(self.ego_vehicles[0],
            pedestrian_actor, pedestrian_info.trigger_distance)

            movement = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

            actor_velocity = KeepVelocity(pedestrian_actor, pedestrian_info.speed)
            actor_traverse = DriveDistance(pedestrian_actor, pedestrian_info.dist_to_travel, tmp_travel_dist_file=tmp_travel_dist_file)


            movement.add_child(actor_velocity)
            movement.add_child(actor_traverse)

            if pedestrian_info.after_trigger_behavior == 'destroy':
                after_trigger_behavior = ActorDestroy(pedestrian_actor)
            elif pedestrian_info.after_trigger_behavior == 'stop':
                after_trigger_behavior = StopVehicle(pedestrian_actor, brake_value=0.5)
                destroy_actor = ActorDestroy(pedestrian_actor)
                destroy_actors.add_child(destroy_actor)
            else:
                raise

            pedestrian_behaviors = py_trees.composites.Sequence()

            pedestrian_behaviors.add_child(trigger_distance)
            pedestrian_behaviors.add_child(movement)
            pedestrian_behaviors.add_child(after_trigger_behavior)

            waypoint_events.add_child(pedestrian_behaviors)



        for i in range(len(self.vehicle_list)):
            vehicle_actor, generated_transform = self.vehicle_list[i]
            vehicle_info = self.customized_data['vehicle_list'][i]

            if tmp_travel_dist_file:
                record_travel_dist(self.customized_data['tmp_travel_dist_file'], vehicle_actor.id, 'vehicle', i)



            keep_velocity = py_trees.composites.Parallel("Trigger condition for changing behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            keep_velocity.add_child(InTriggerDistanceToVehicle(self.ego_vehicles[0], vehicle_actor, vehicle_info.trigger_distance))
            keep_velocity.add_child(WaypointFollower(vehicle_actor, vehicle_info.initial_speed, avoid_collision=vehicle_info.avoid_collision))




            if vehicle_info.waypoint_follower:
                # interpolate current location and destination to find a path

                start_location = generated_transform.location
                end_location = vehicle_info.targeted_waypoint.location
                _, route = interpolate_trajectory(self.world, [start_location, end_location])
                ds_ids = downsample_route(route, self.customized_data['sample_factor'])
                route = [(route[x][0], route[x][1]) for x in ds_ids]

                # print('route', len(route))
                perturb_route(route, vehicle_info.waypoints_perturbation)
                # visualize_route(route)

                plan = []
                for transform, cmd in route:
                    wp = self._wmap.get_waypoint(transform.location, project_to_road=False, lane_type=carla.LaneType.Any)
                    if not wp:
                        wp = self._wmap.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Any)
                        print('(', transform.location.x, transform.location.y, ')', 'is replaced by', '(', wp.transform.location.x, wp.transform.location.y, ')')
                    plan.append((wp, cmd))


                movement = WaypointFollower(actor=vehicle_actor, target_speed=vehicle_info.targeted_speed, plan=plan, avoid_collision=vehicle_info.avoid_collision)
            else:
                movement = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
                actor_velocity = KeepVelocity(vehicle_actor, vehicle_info.targeted_speed, target_direction=vehicle_info.target_direction)
                actor_traverse = DriveDistance(vehicle_actor, vehicle_info.dist_to_travel, tmp_travel_dist_file=tmp_travel_dist_file)
                movement.add_child(actor_velocity)
                movement.add_child(actor_traverse)







            if vehicle_info.after_trigger_behavior == 'destroy':
                after_trigger_behavior = ActorDestroy(vehicle_actor)
            elif vehicle_info.after_trigger_behavior == 'stop':
                after_trigger_behavior = StopVehicle(vehicle_actor, brake_value=0.5)
                destroy_actor = ActorDestroy(vehicle_actor)
                destroy_actors.add_child(destroy_actor)
            else:
                raise


            vehicle_behaviors = py_trees.composites.Sequence()

            vehicle_behaviors.add_child(keep_velocity)
            vehicle_behaviors.add_child(movement)
            vehicle_behaviors.add_child(after_trigger_behavior)

            waypoint_events.add_child(vehicle_behaviors)



        return scenario_sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
