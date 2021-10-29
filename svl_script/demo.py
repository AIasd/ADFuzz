import os
import lgsvl
import time
import psutil
import atexit
import logging
import math




def kill_mainboard():
    PROC_NAME = "mainboard"
    for proc in psutil.process_iter():
        # check whether the process to kill name matches
        if proc.name() == PROC_NAME:
            proc.kill()

def on_waypoint(agent, index):
    print("Waypoint {} reached".format(index))

def run_svl_simulation():
    log = logging.getLogger(__name__)
    atexit.register(kill_mainboard)


    SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
    SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", 8181))
    BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", 9090))

    sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)
    if sim.current_scene == "BorregasAve":
        sim.reset()
    else:
        sim.load("BorregasAve")

    spawns = sim.get_spawn()

    state = lgsvl.AgentState()
    state.transform = spawns[0]

    ego = sim.add_agent("9272dd1a-793a-45b2-bff4-3a160b506d75", lgsvl.AgentType.EGO, state)
    ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)

    # Dreamview setup
    dv = lgsvl.dreamview.Connection(sim, ego, BRIDGE_HOST)
    dv.set_hd_map('Borregas Ave')
    dv.set_vehicle('Lincoln2017MKZ_LGSVL')
    modules = [
        'Localization',
        'Perception',
        'Transform',
        'Routing',
        'Prediction',
        'Planning',
        'Camera',
        'Traffic Light',
        'Control'
    ]
    destination = spawns[0].destinations[0]
    dv.setup_apollo(destination.position.x, destination.position.z, modules, default_timeout=60)
    print('finish setup_apollo')


    # x_long_east = destination.position.x
    # z_lat_north = destination.position.z
    # dv.set_destination(x_long_east, z_lat_north, y=0)


    forward = lgsvl.utils.transform_to_forward(spawns[0])
    right = lgsvl.utils.transform_to_right(spawns[0])

    print('spawns[0]', spawns[0])
    print('forward', forward)
    print('right', right)


    import copy

    wp = [
        lgsvl.WalkWaypoint(spawns[0].position + 6 * right + 50 * forward, 0),
        lgsvl.WalkWaypoint(spawns[0].position + -15 * right + 50 * forward, 0) ]
    state = lgsvl.AgentState()
    state.transform = copy.deepcopy(spawns[0])
    state.transform.position = wp[0].position
    p = sim.add_agent("Pamela", lgsvl.AgentType.PEDESTRIAN, state)
    # p.on_waypoint_reached(on_waypoint)
    p.follow(wp, False)


    # import copy
    # ped_start = copy.deepcopy(spawns[0])
    #
    # state = lgsvl.ObjectState()
    # state.transform.position = ped_start.position + 1 * lgsvl.utils.transform_to_forward(ped_start)
    #
    # state.transform.rotation = lgsvl.Vector(0,0,0)
    # state.velocity = lgsvl.Vector(0,0,0)
    # state.angular_velocity = lgsvl.Vector(0,0,0)
    # static_object = sim.controllable_add('TrafficCone', state)

    wp2 = [
        lgsvl.DriveWaypoint(spawns[0].position + 10 * right + 30 * forward, 1, lgsvl.Vector(0, 0, 0), 0, False, 50),
        lgsvl.DriveWaypoint(spawns[0].position + -30 * right + 30 * forward, 1, lgsvl.Vector(0, 0, 0), 0, False, 50) ]
    wp2[0].position.y = -1.5
    wp2[1].position.y = -1.5
    state = lgsvl.AgentState()
    state.transform = copy.deepcopy(spawns[0])
    state.transform.position = wp2[0].position
    p = sim.add_agent('Sedan', lgsvl.AgentType.NPC, state)
    p.follow(wp2, False)




    sim.run()



if __name__ == '__main__':
    run_svl_simulation()
