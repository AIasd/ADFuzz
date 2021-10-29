import time


import numpy as np
import carla


VEHICLE_NAME = 'vehicle.lincoln.mkz2017'
COLLISION_THRESHOLD = 20000


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 20.0

    world.apply_settings(settings)


class VehiclePool(object):
    def __init__(self, client, n_vehicles):
        self.client = client
        self.world = client.get_world()

        veh_bp = self.world.get_blueprint_library().filter('vehicle.*')
        veh_bp = self.world.get_blueprint_library().filter(VEHICLE_NAME)
        spawn_points = np.random.choice(self.world.get_map().get_spawn_points(), n_vehicles)
        batch = list()

        for i, transform in enumerate(spawn_points):
            bp = np.random.choice(veh_bp)
            bp.set_attribute('role_name', 'autopilot')

            batch.append(
                    carla.command.SpawnActor(bp, transform).then(
                        carla.command.SetAutopilot(carla.command.FutureActor, True)))

        self.vehicles = list()
        errors = set()

        for msg in self.client.apply_batch_sync(batch):
            if msg.error:
                errors.add(msg.error)
            else:
                self.vehicles.append(msg.actor_id)

        import pdb; pdb.set_trace()

        if errors:
            print('\n'.join(errors))

        print('%d / %d vehicles spawned.' % (len(self.vehicles), n_vehicles))

    def __del__(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])


class PedestrianPool(object):
    def __init__(self, client, n_pedestrians):
        self.client = client
        self.world = client.get_world()

        ped_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        con_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        spawn_points = [self._get_spawn_point() for _ in range(n_pedestrians)]
        batch = [carla.command.SpawnActor(np.random.choice(ped_bp), spawn) for spawn in spawn_points]
        walkers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                walkers.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        batch = [carla.command.SpawnActor(con_bp, carla.Transform(), walker_id) for walker_id in walkers]
        controllers = list()
        errors = set()

        for msg in client.apply_batch_sync(batch, True):
            if msg.error:
                errors.add(msg.error)
            else:
                controllers.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        self.walkers = self.world.get_actors(walkers)
        self.controllers = self.world.get_actors(controllers)

        for controller in self.controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.4 + np.clip(np.random.randn(), -1, 1))

        self.timers = [np.random.randint(60, 600) * 20 for _ in self.controllers]

        print('%d / %d pedestrians spawned.' % (len(self.walkers), n_pedestrians))

    def _get_spawn_point(self, n_retry=10):
        for _ in range(n_retry):
            spawn = carla.Transform()
            spawn.location = self.world.get_random_location_from_navigation()

            if spawn.location is not None:
                return spawn

        raise ValueError('No valid spawns.')

    def tick(self):
        for i, controller in enumerate(self.controllers):
            self.timers[i] -= 1

            if self.timers[i] <= 0:
                self.timers[i] = np.random.randint(60, 600) * 20
                controller.go_to_location(self.world.get_random_location_from_navigation())

    def __del__(self):
        for controller in self.controllers:
            controller.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers])


class CarlaEnv(object):
    def __init__(self, town='Town01', port=2000, **kwargs):
        self._client = carla.Client('localhost', port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()

        self._is_recording = False
        self._tick = 0

    def reset(self, weather='random', n_vehicles=10, n_pedestrians=10, seed=0, save_path=None):
        is_ready = False

        while not is_ready:
            np.random.seed(seed)

            self._clean_up()

            self._pedestrian_pool = PedestrianPool(self._client, n_pedestrians)
            self._vehicle_pool = VehiclePool(self._client, n_pedestrians)

            is_ready = self.ready()

        if save_path is not None:
            self._client.start_recorder(save_path)
            self._is_recording = True

    def ready(self, ticks=10):
        for _ in range(ticks):
            self.step()

        self._time_start = time.time()
        self._tick = 0

        return True

    def step(self):
        self._world.tick()
        self._tick += 1
        self._pedestrian_pool.tick()

    def __enter__(self):
        set_sync_mode(self._client, True)

        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._clean_up()

        set_sync_mode(self._client, False)

    def _clean_up(self):
        if self._is_recording:
            self._client.stop_recorder()

        self._is_recording = False
        self._pedestrian_pool = None
        self._vehicle_pool = None

        self._tick = 0
        self._time_start = time.time()


import tqdm


for _ in range(5):
    with CarlaEnv('Town02') as env:
        env.reset(n_vehicles=50, n_pedestrians=50, save_path='/home/bradyzhou/recording.log')

        for _ in tqdm.tqdm(range(100 * 10)):
            env.step()
