from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from team_code.base_agent import BaseAgent
from team_code.planner import RoutePlanner
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights


def get_entry_point():
    return 'MapAgent'


class MapAgent(BaseAgent):
    # remove the sensors since we made changes to base_agent.py

    def set_global_plan(self, global_plan_gps, global_plan_world_coord, sample_factor=1):
        super().set_global_plan(global_plan_gps, global_plan_world_coord, sample_factor)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        super()._init()

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)

        self._traffic_lights = list()

    def tick(self, input_data):
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))

        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)

        result = super().tick(input_data)
        result['topdown'] = topdown

        return result
