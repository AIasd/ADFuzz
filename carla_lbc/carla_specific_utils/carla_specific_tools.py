import re
import os
import carla
import numpy as np
import traceback
import logging
import time
import subprocess
import shlex
import json
from customized_utils import is_port_in_use

def perturb_route(route, perturbation):
    num_to_perturb = min([len(route), len(perturbation) + 2])
    for i in range(num_to_perturb):
        if i != 0 and i != num_to_perturb - 1:
            route[i][0].location.x += perturbation[i - 1][0]
            route[i][0].location.y += perturbation[i - 1][1]

def add_transform(transform1, transform2):
    x = transform1.location.x + transform2.location.x
    y = transform1.location.y + transform2.location.y
    z = transform1.location.z + transform2.location.z
    pitch = transform1.rotation.pitch + transform2.rotation.pitch
    yaw = transform1.rotation.yaw + transform2.rotation.yaw
    roll = transform1.rotation.roll + transform2.rotation.roll
    return create_transform(x, y, z, pitch, yaw, roll)

def create_transform(x, y, z, pitch, yaw, roll):
    location = carla.Location(x, y, z)
    rotation = carla.Rotation(pitch, yaw, roll)
    transform = carla.Transform(location, rotation)
    return transform

def copy_transform(t):
    return create_transform(
        t.location.x,
        t.location.y,
        t.location.z,
        t.rotation.pitch,
        t.rotation.yaw,
        t.rotation.roll,
    )


def visualize_route(route):
    n = len(route)

    x_list = []
    y_list = []

    # The following code prints out the planned route
    for i, (transform, command) in enumerate(route):
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z
        pitch = transform.rotation.pitch
        yaw = transform.rotation.yaw
        if i == 0:
            s = "start"
            x_s = [x]
            y_s = [y]
        elif i == n - 1:
            s = "end"
            x_e = [x]
            y_e = [y]
        else:
            s = "point"
            x_list.append(x)
            y_list.append(y)

        # print(s, x, y, z, pitch, yaw, command

    import matplotlib.pyplot as plt

    plt.gca().invert_yaxis()
    plt.scatter(x_list, y_list)
    plt.scatter(x_s, y_s, c="red", linewidths=5)
    plt.scatter(x_e, y_e, c="black", linewidths=5)

    plt.show()


def estimate_objectives(save_path, default_objectives=np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.]), verbose=True):

    events_path = os.path.join(save_path, "events.txt")
    deviations_path = os.path.join(save_path, "deviations.txt")

    # set thresholds to avoid too large influence
    ego_linear_speed = 0
    min_d = 20
    offroad_d = 7
    wronglane_d = 7
    dev_dist = 0
    d_angle_norm = 1

    ego_linear_speed_max = 7
    dev_dist_max = 7

    is_offroad = 0
    is_wrong_lane = 0
    is_run_red_light = 0
    is_collision = 0

    with open(deviations_path, "r") as f_in:
        for line in f_in:
            type, d = line.split(",")
            d = float(d)
            if type == "min_d":
                min_d = np.min([min_d, d])
            elif type == "offroad_d":
                offroad_d = np.min([offroad_d, d])
            elif type == "wronglane_d":
                wronglane_d = np.min([wronglane_d, d])
            elif type == "dev_dist":
                dev_dist = np.max([dev_dist, d])
            elif type == "d_angle_norm":
                d_angle_norm = np.min([d_angle_norm, d])

    x = None
    y = None
    object_type = None

    infraction_types = [
        "collisions_layout",
        "collisions_pedestrian",
        "collisions_vehicle",
        "red_light",
        "on_sidewalk",
        "outside_lane_infraction",
        "wrong_lane",
        "off_road",
    ]

    try:
        with open(events_path, 'r') as json_file:
            events = json.load(json_file)
    except Exception as e:
        print(repr(e))
        # print("events_path", events_path, "is not found")
        return default_objectives, (None, None), None, None
    infractions = events["_checkpoint"]["records"][0]["infractions"]
    status = events["_checkpoint"]["records"][0]["status"]

    route_completion = float(events["values"][1])

    for infraction_type in infraction_types:
        for infraction in infractions[infraction_type]:
            if "collisions" in infraction_type:
                typ = re.search(".*with type=(.*) and id.*", infraction)
                if verbose:
                    print(infraction, typ)
                if typ:
                    object_type = typ.group(1)
                loc = re.search(
                    ".*x=(.*), y=(.*), z=(.*), ego_linear_speed=(.*), other_actor_linear_speed=(.*)\)",
                    infraction,
                )
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    ego_linear_speed = float(loc.group(4))
                    if loc.group(5).isnumeric():
                        other_actor_linear_speed = float(loc.group(5))
                    else:
                        other_actor_linear_speed = -1
                        print('other_actor_linear_speed is not numeric:', loc.group(5))

                    # only record valid collisions to promote valid collision bugs
                    if ego_linear_speed > 0.1:
                        is_collision = 1

            elif infraction_type == "off_road":
                loc = re.search(".*x=(.*), y=(.*), z=(.*)\)", infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))
                    is_offroad = 1
            else:
                if infraction_type == "wrong_lane":
                    is_wrong_lane = 1
                elif infraction_type == "red_light":
                    is_run_red_light = 1
                loc = re.search(".*x=(.*), y=(.*), z=(.*)[\),]", infraction)
                if loc:
                    x = float(loc.group(1))
                    y = float(loc.group(2))

    # limit impact of too large values
    ego_linear_speed = np.min([ego_linear_speed, ego_linear_speed_max])
    dev_dist = np.min([dev_dist, dev_dist_max])

    return (
        [
            ego_linear_speed,
            min_d,
            d_angle_norm,
            offroad_d,
            wronglane_d,
            dev_dist,
            is_collision,
            is_offroad,
            is_wrong_lane,
            is_run_red_light,
        ],
        (x, y),
        object_type,
        route_completion,
    )

def start_server(port):
    # hack: this heavily relies on the relative path of carla
    cmd_list = shlex.split(
        "sh ../carla_0994_no_rss/CarlaUE4.sh -opengl -carla-rpc-port="
        + str(port)
        + " -carla-streaming-port=0 -quality-level=Epic"
    )
    while is_port_in_use(int(port)):
        try:
            # show_ports_cmd = shlex.split('lsof -t -i:'+str(port))
            # result = subprocess.run(show_ports_cmd, stdout=subprocess.PIPE)
            # pids = result.stdout.decode("utf-8").strip().split('\n')
            # own_pid = str(os.getpid())
            #
            # if own_pid in pids:
            #     pids.remove(own_pid)
            #     if len(pids) > 0:
            #         pid_to_kill = pids[0]
            #         print('pid_to_kill', pid_to_kill)
            #         subprocess.run('kill -9 '+pid_to_kill, shell=True)
            # else:
            #     subprocess.run('kill $(lsof -t -i:'+str(port)+')', shell=True)
            subprocess.run("kill $(lsof -t -i:" + str(port) + ")", shell=True)
            print("-" * 20, "kill server at port", port)
            time.sleep(2)
        except:
            import traceback

            traceback.print_exc()
            continue
    subprocess.Popen(cmd_list)
    print("-" * 20, "start server at port", port)
    # 10s is usually enough
    time.sleep(10)

def start_client(obj, host, port):
    print('initialize carla client')

    while True:
        try:
            obj.client = carla.Client(host, port)
            break
        except:
            logging.exception("__init__ error")
            traceback.print_exc()

def try_load_world(obj, town, host, port):
    while True:
        try:
            print('start loading town :', town)
            obj.world = obj.client.load_world(town)
            print('finish loading town :', town)
            break
        except:
            logging.exception("_load_and_wait_for_world error")
            traceback.print_exc()

            start_server(port)
            obj.client = carla.Client(host, port)
