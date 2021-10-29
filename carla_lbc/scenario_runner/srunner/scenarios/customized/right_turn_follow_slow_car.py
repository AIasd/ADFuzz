#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right or left turn.
"""

from __future__ import print_function

import math
import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter, ActorDestroy, KeepVelocity, HandBrakeVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute, InTriggerDistanceToVehicle, DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint, generate_target_waypoint_in_route



class RightTurnFollowSlowCar(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """

        self._other_actor_target_velocity = 3
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location

        self.other_actors = []
        self._other_actor_transform = config.other_actors[0].transform

        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(RightTurnFollowSlowCar, self).__init__("RightTurnFollowSlowCar",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # Get the waypoint left after the junction
        waypoint = generate_target_waypoint(self._reference_waypoint, 1)


        while True:

            # Try to spawn the actor
            try:
                self._other_actor_transform = waypoint.transform
                first_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.*', self._other_actor_transform)
                first_vehicle.set_simulate_physics(enabled=False)
                break

            # Move the spawning point a bit and try again
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                waypoint = waypoint.next(1)[0]
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)

        first_vehicle.set_transform(actor_transform)

        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionRightTurn")

        self._other_actor_transform
        dist_to_travel = 30
        start_dist = self._other_actor_transform.location.distance(self._reference_waypoint.transform.location)

        if self._ego_route is not None:
            trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0], self._ego_route, self._other_actor_transform.location, start_dist)
        else:
            trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0],
                                                          self.ego_vehicles[0],
                                                          start_dist)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * dist_to_travel)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * dist_to_travel)
        end_condition = TimeOut(5)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()

        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timeout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform, name='TransformSetterTS11'))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)

        return root

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
