'''
fix modular mode traffic light (by manually control its cycle?)
figure out determinism on different runs and different step_time (for both modular and non-modular)
rerun code
save in parallel (run a separate thread to record environment values, cyber-recorder, cyber-RT python API: https://github.com/ApolloAuto/apollo/blob/master/docs/cyber/CyberRT_Python_API.md, camera proto https://github.com/ApolloAuto/apollo/blob/master/modules/drivers/camera/proto/config.proto)
tune objectives
top down camera
'''


import os
import lgsvl
import time
import psutil
import atexit
import math
from svl_script.object_types import static_types, pedestrian_types, vehicle_types
from customized_utils import emptyobject
import numpy as np

accident_happen = False

# temporary, can be imported from customized_utils
def exit_handler():
    PROC_NAME = "mainboard"
    for proc in psutil.process_iter():
        # check whether the process to kill name matches
        if proc.name() == PROC_NAME:
            proc.kill()
##################################################


def norm_2d(loc_1, loc_2):
    return np.sqrt((loc_1.x - loc_2.x) ** 2 + (loc_1.z - loc_2.z) ** 2)

def on_waypoint(agent, index):
    print("Waypoint {} reached".format(index))



def initialize_simulator(map, sim_specific_arguments):
    SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
    SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", 8181))
    BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", 9090))

    if not sim_specific_arguments.sim:
        sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)
        sim_specific_arguments.sim = sim
    else:
        sim = sim_specific_arguments.sim
    # hack: TBD: make map name consistent for Apollo and carla

    if sim.current_scene == map:
        sim.reset()
    else:
        # seed make sure the weather and NPC behvaiors deterministic
        sim.load(map, seed=0)

    return sim, BRIDGE_HOST, BRIDGE_PORT


def initialize_dv_and_ego(sim, map, model_id, start, destination, BRIDGE_HOST, BRIDGE_PORT, events_path):

    global accident_happen
    accident_happen = False

    def on_collision(agent1, agent2, contact):
        global accident_happen
        accident_happen = True
        name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
        name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
        print("{} collided with {} at {}".format(name1, name2, contact))
        print('v_ego:', agent1.state.velocity)

        loc = agent1.transform.position
        if not agent2:
            other_agent_type = 'static'
        else:
            other_agent_type = agent2.name
        ego_speed = np.linalg.norm([agent1.state.velocity.x, agent1.state.velocity.y, agent1.state.velocity.z])
        # d_angle_norm = angle_from_center_view_fov(agent2, agent1)
        #
        # if d_angle_norm > 0:
        #     ego_speed = -1

        data_row = ['collision', ego_speed, other_agent_type, loc.x, loc.y]
        data_row = ','.join([str(data) for data in data_row])
        with open(events_path, 'a') as f_out:
            f_out.write(data_row+'\n')
        time.sleep(2)
        sim.stop()


    times = 0
    success = False
    while times < 3:
        try:
            state = lgsvl.AgentState()
            state.transform = start

            ego = sim.add_agent(model_id, lgsvl.AgentType.EGO, state)
            ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)
            ego.on_collision(on_collision)

            # Dreamview setup
            dv = lgsvl.dreamview.Connection(sim, ego, BRIDGE_HOST)
            dv.set_hd_map(map)
            dv.set_vehicle('Lincoln2017MKZ_LGSVL')

            if model_id == '9272dd1a-793a-45b2-bff4-3a160b506d75':
                modules = [
                    'Localization',
                    'Perception',
                    'Transform',
                    'Routing',
                    'Prediction',
                    'Planning',
                    'Camera',
                    # 'Traffic Light',
                    'Control'
                ]
            elif model_id in ['2e9095fa-c9b9-4f3f-8d7d-65fa2bb03921', 'f0daed3e-4b1e-46ce-91ec-21149fa31758']:
                modules = [
                    'Localization',
                    # 'Perception',
                    'Transform',
                    'Routing',
                    'Prediction',
                    'Planning',
                    # 'Camera',
                    # 'Traffic Light',
                    'Control'
                ]
            else:
                raise Exception('unknown model_id: '+ model_id)

            start = lgsvl.Transform(position=ego.transform.position, rotation=ego.transform.rotation)

            print('start', start)
            print('destination', destination)
            dv.setup_apollo(destination.position.x, destination.position.z, modules, default_timeout=60)
            print('finish setup_apollo')
            success = True
            break
        except:
            print('fail to spin up apollo, try again!')
            times += 1
    if not success:
        raise Exception('fail to spin up apollo')

    return ego, dv



#######################################################################
# for step-wise simulation
def save_camera(ego, main_camera_folder, counter, i):
    import os
    for sensor in ego.get_sensors():
        if sensor.name == "Main Camera":
            rel_path = '/home/zhongzzy9/Documents/self-driving-cars/2020_CARLA_challenge/run_results_svl'+'/'+"main_camera_"+str(counter)+'_'+str(i)+".png"
            try:
                sensor.save(rel_path, compression=9)
            except:
                print('exception happens when saving camera image')

def save_measurement(ego, measurements_path):
    state = ego.state
    pos = state.position
    rot = state.rotation
    speed = state.speed * 3.6
    with open(measurements_path, 'a') as f_out:
        f_out.write(','.join([str(v) for v in [speed, pos.x, pos.y, pos.z, rot.x, rot.y, rot.z]])+'\n')

def get_bbox(agent):
    x_min = agent.bounding_box.min.x
    x_max = agent.bounding_box.max.x
    z_min = agent.bounding_box.min.z
    z_max = agent.bounding_box.max.z
    bbox = [
        agent.transform.position+lgsvl.Vector(x_min, 0, z_min),
        agent.transform.position+lgsvl.Vector(x_min, 0, z_max),
        agent.transform.position+lgsvl.Vector(x_max, 0, z_min),
        agent.transform.position+lgsvl.Vector(x_max, 0, z_max)
    ]

    return bbox

def gather_info(ego, other_agents, cur_values, deviations_path):
    ego_bbox = get_bbox(ego)
    # TBD: only using the front two vertices
    # ego_front_bbox = ego_bbox[:2]

    # hack: smaller than default values to make sure the files are created to have at least one line
    min_d = 9999
    d_angle_norm = 0.99
    for i, other_agent in enumerate(other_agents):

        # d_angle_norm_i = angle_from_center_view_fov(other_agent, ego, fov=90)
        # d_angle_norm = np.min([d_angle_norm, d_angle_norm_i])
        # if d_angle_norm_i == 0:
        # TBD: temporarily disable d_angle_norm since Apollo uses LiDAR
        d_angle_norm = 0
        other_bbox = get_bbox(other_agent)
        for other_b in other_bbox:
            for ego_b in ego_bbox:
                d = norm_2d(other_b, ego_b)
                min_d = np.min([min_d, d])

    if min_d < cur_values.min_d:
        cur_values.min_d = min_d
        with open(deviations_path, 'a') as f_out:
            f_out.write('min_d,'+str(cur_values.min_d)+'\n')

    if d_angle_norm < cur_values.d_angle_norm:
        cur_values.d_angle_norm = d_angle_norm
        with open(deviations_path, 'a') as f_out:
            f_out.write('d_angle_norm,'+str(cur_values.d_angle_norm)+'\n')

    # TBD: out-of-road violation related data

#######################################################################
# for continuous simulation
def bind_socket(socket, port_num):
    import subprocess
    while True:
        try:
            socket.bind("tcp://*:"+str(port_num))
            break
        except:
            subprocess.run("kill -9 $(lsof -t -i:" + str(port_num) + " -sTCP:LISTEN)", shell=True)

def receive_zmq(q, path_list, record_every_n_step):
    import zmq
    print('receive zmq')

    def get_d(x1, y1, z1, x2, y2, z2):
        return np.sqrt((x1-x2)**2+(y1-y2)**2)



    odometry_path, perception_obstacles_path, main_camera_folder, deviations_path = path_list


    print('start binding sockets')
    context = zmq.Context()
    socket_odometry = context.socket(zmq.PAIR)
    socket_perception_obstacles = context.socket(zmq.PAIR)
    socket_front_camera = context.socket(zmq.PAIR)

    bind_socket(socket_odometry, 5561)
    bind_socket(socket_perception_obstacles, 5562)
    bind_socket(socket_front_camera, 5563)
    print('finish binding sockets')

    min_ego_i_d = 100
    odometry_dict = {}
    perception_obstacles_dict = {}
    with open(odometry_path, 'a') as f_out_odometry:
        with open(perception_obstacles_path, 'a') as f_out_perception_obstacles:
            while True:
                data_str_odometry = None
                data_str_perception_obstacles = None

                try:
                    cmd = q.get(timeout=0.0001)
                    if cmd == 'end':
                        with open(deviations_path, 'a') as f_out:
                            for k in perception_obstacles_dict:
                                if k in odometry_dict:
                                    ego_x, ego_y, ego_z = odometry_dict[k]
                                    npc_num = len(perception_obstacles_dict[k])
                                    for i in range(npc_num):
                                        i_x, i_y, i_z = perception_obstacles_dict[k][i]

                                        ego_i_d = get_d(ego_x, ego_y, ego_z, i_x, i_y, i_z)
                                        if ego_i_d < min_ego_i_d:
                                            min_ego_i_d = ego_i_d
                                            f_out.write('min_d,'+str(min_ego_i_d)+'\n')
                                            print('time step', k, 'min_d', min_ego_i_d)
                        socket_odometry.close()
                        socket_perception_obstacles.close()
                        socket_front_camera.close()
                        context.term()
                        print('free sockets and context')
                        return
                except:
                    pass

                try:
                    data_str_odometry = socket_odometry.recv_string(flags=zmq.NOBLOCK)
                    f_out_odometry.write(data_str_odometry+'\n')

                    odometry_tokens = data_str_odometry.split(':')[1].split(',')
                    ego_x, ego_y, ego_z = [float(x) for x in odometry_tokens[2:]]
                    time_step = int(perception_obstacles_tokens[1])
                    odometry_dict[time_step] = (ego_x, ego_y, ego_z)
                except Exception:
                    pass

                try:
                    data_str_perception_obstacles = socket_perception_obstacles.recv_string(flags=zmq.NOBLOCK)
                    f_out_perception_obstacles.write(data_str_perception_obstacles+'\n')

                    perception_obstacles_tokens = data_str_perception_obstacles.split(':')[1].split(',')
                    npc_num = int(perception_obstacles_tokens[2])
                    if npc_num > 0:
                        time_step = int(perception_obstacles_tokens[1])
                        for i in range(npc_num):
                            i_x, i_y, i_z = [float(x) for x in perception_obstacles_tokens[3*(1+i):3*(2+i)]]
                            if time_step not in perception_obstacles_dict:
                                perception_obstacles_dict[time_step] = [(i_x, i_y, i_z)]
                            else:
                                perception_obstacles_dict[time_step].append((i_x, i_y, i_z))


                except Exception:
                    pass



                try:
                    data_str_front_camera = socket_front_camera.recv(flags=zmq.NOBLOCK)
                    timestamp_sec, sequence_num, front_image = data_str_front_camera.split(b':data_delimiter:')

                    # record image after warm-up stage
                    if int(sequence_num) > 150 and int(sequence_num) % record_every_n_step == 0:
                        img_path = os.path.join(main_camera_folder, sequence_num.decode()+'_'+timestamp_sec.decode()+'.jpg')
                        with open(img_path, 'wb') as f_out_front_camera:
                            f_out_front_camera.write(front_image)
                except Exception:
                    pass




#######################################################################

def rotate(x, y, rot_rad):
    x_rot = x * np.cos(rot_rad) - y * np.sin(rot_rad)
    y_rot = x * np.sin(rot_rad) + y * np.cos(rot_rad)
    return x_rot, y_rot


def initialize_sim(map, sim_specific_arguments, arguments, customized_data, model_id, events_path):


    sim, BRIDGE_HOST, BRIDGE_PORT = initialize_simulator(map, sim_specific_arguments)

    if len(arguments.route_info["location_list"]) == 0:
        spawns = sim.get_spawn()
        start = spawns[0]
        destination = spawns[0].destinations[0]
    else:
        start, destination = arguments.route_info["location_list"]

        start = lgsvl.Transform(position=lgsvl.Vector(start[0], start[1], start[2]), rotation=lgsvl.Vector(start[3], start[4], start[5]))
        destination = lgsvl.Transform(position=lgsvl.Vector(destination[0], destination[1], destination[2]), rotation=lgsvl.Vector(destination[3], destination[4], destination[5]))

    try:
        sim.weather = lgsvl.WeatherState(rain=customized_data['rain'], fog=customized_data['fog'], wetness=customized_data['wetness'], cloudiness=customized_data['cloudiness'], damage=customized_data['damage'])

        from datetime import datetime
        dt = datetime(
              year=2020,
              month=12,
              day=25,
              hour=int(customized_data['hour']),
              minute = 0,
              second = 0
            )
        sim.set_date_time(dt, fixed=True)
    except:
        import traceback
        traceback.print_exc()

    ego, dv = initialize_dv_and_ego(sim, map, model_id, start, destination, BRIDGE_HOST, BRIDGE_PORT, events_path)

    middle_point = lgsvl.Transform(position=(destination.position + start.position) * 0.5, rotation=start.rotation)

    for k, v in customized_data['customized_center_transforms'].items():
        if v[0] == "absolute_location":
            middle_point_i = lgsvl.Transform(position=lgsvl.Vector(v[1], v[2], v[3]), rotation=lgsvl.Vector(v[4], v[5], v[6]))
            customized_data[k] = middle_point_i



    other_agents = []
    for static in customized_data['static_list']:
        state = lgsvl.ObjectState()
        state.transform.position = lgsvl.Vector(static.x,0,static.y)
        state.transform.rotation = lgsvl.Vector(0,0,0)
        state.velocity = lgsvl.Vector(0,0,0)
        state.angular_velocity = lgsvl.Vector(0,0,0)

        static_object = sim.controllable_add(static_types[static.model], state)

    for i, ped in enumerate(customized_data['pedestrians_list']):
        center_key_i = "pedestrian_center_transform_"+str(i)
        if center_key_i in customized_data:
            middle_point_i = customized_data[center_key_i]
        else:
            middle_point_i = middle_point
        rot_rad = np.deg2rad(360 - middle_point_i.rotation.y)

        ped_x, ped_y = rotate(ped.x, ped.y, rot_rad)

        ped_position_offset = lgsvl.Vector(ped_x, 0, ped_y)
        ped_rotation_offset = lgsvl.Vector(0, 0, 0)

        ped_point = lgsvl.Transform(position=middle_point_i.position+ped_position_offset, rotation=middle_point_i.rotation+ped_rotation_offset)

        wps = [lgsvl.WalkWaypoint(position=ped_point.position, idle=ped.waypoints[0].idle, trigger_distance=ped.waypoints[0].trigger_distance, speed=ped.speed)]


        for j, wp in enumerate(ped.waypoints):
            center_key_i_j = "pedestrian_"+str(i)+"_center_transform_"+str(j)
            if center_key_i_j in customized_data:
                middle_point_i = customized_data[center_key_i_j]
                rot_rad = np.deg2rad(360 - middle_point_i.rotation.y)


            j_next = np.min([j+1, len(ped.waypoints)-1])
            wp_next = ped.waypoints[j_next]

            wp_x, wp_y = rotate(wp.x, wp.y, rot_rad)

            loc = middle_point_i.position+lgsvl.Vector(wp_x, 0, wp_y)

            # to avoid pedestrian going off ground
            loc.y -= 0.1

            wps.append(lgsvl.WalkWaypoint(position=loc, idle=wp_next.idle, trigger_distance=wp_next.trigger_distance, speed=ped.speed))

        state = lgsvl.AgentState()
        state.transform = ped_point
        print('\n'*3, 'ped.model', ped.model, '\n'*3)
        p = sim.add_agent(pedestrian_types[ped.model], lgsvl.AgentType.PEDESTRIAN, state)
        p.follow(wps, False)
        other_agents.append(p)

    for vehicle in customized_data['vehicles_list']:
        center_key_i = "vehicle_center_transform_"+str(i)
        if center_key_i in customized_data:
            middle_point_i = customized_data[center_key_i]
        else:
            middle_point_i = middle_point
        rot_rad = np.deg2rad(360 - middle_point_i.rotation.y)

        vehicle_x, vehicle_y = rotate(vehicle.x, vehicle.x, rot_rad)
        vehicle_position_offset = lgsvl.Vector(vehicle_x, 0, vehicle_y)
        vehicle_rotation_offset = lgsvl.Vector(0, 0, 0)

        vehicle_point = lgsvl.Transform(position=middle_point_i.position+vehicle_position_offset, rotation=middle_point_i.rotation+vehicle_rotation_offset)

        wps = [lgsvl.DriveWaypoint(position=vehicle_point.position, speed=vehicle.speed, acceleration=0, angle=vehicle_point.rotation, idle=vehicle.waypoints[0].idle, deactivate=False, trigger_distance=vehicle.waypoints[0].trigger_distance)]

        for j, wp in enumerate(vehicle.waypoints):
            center_key_i_j = "vehicle_"+str(i)+"_center_transform_"+str(j)
            if center_key_i_j in customized_data:
                middle_point_i = customized_data[center_key_i_j]
                rot_rad = np.deg2rad(360 - middle_point_i.rotation.y)

            j_next = np.min([j+1, len(vehicle.waypoints)-1])
            wp_next = vehicle.waypoints[j_next]

            wp_x, wp_y = rotate(wp.x, wp.y, rot_rad)
            pos = middle_point_i.position + lgsvl.Vector(wp_x, 0, wp_y)

            # to avoid vehicle going underground
            pos.y += 0.3

            wps.append(lgsvl.DriveWaypoint(position=pos, speed=vehicle.speed, acceleration=0, angle=middle_point_i.rotation, idle=wp_next.idle, deactivate=False, trigger_distance=wp_next.trigger_distance))

        state = lgsvl.AgentState()
        state.transform = vehicle_point
        print('\n'*3, 'vehicle.model', vehicle.model, '\n'*3)
        p = sim.add_agent(vehicle_types[vehicle.model], lgsvl.AgentType.NPC, state)
        p.follow(wps, False)
        other_agents.append(p)

    controllables = sim.get_controllables()
    for i in range(len(controllables)):
        signal = controllables[i]
        if signal.type == "signal":
            control_policy = signal.control_policy
            control_policy = "trigger=500;green=10;yellow=2;red=5;loop"
            signal.control(control_policy)

    # extra destination request to avoid previous request lost?
    dv.set_destination(destination.position.x, destination.position.z)

    return sim, ego, destination

def run_sim_with_initialization(q, duration, time_scale, map, sim_specific_arguments, arguments, customized_data, model_id, events_path):

    sim, ego, destination = initialize_sim(map, sim_specific_arguments, arguments, customized_data,model_id, events_path)
    print('start run sim')
    sim.run(time_limit=duration, time_scale=time_scale)

    d_to_dest = norm_2d(ego.transform.position, destination.position)
    if d_to_dest > 10:
        with open(events_path, 'a') as f_out:
            f_out.write('fail_to_finish,'+str(d_to_dest))
    print('ego car final transform:', ego.transform, 'destination', destination, 'd_to_dest', d_to_dest)
    time.sleep(1)
    q.put('end')
    return


def start_simulation(customized_data, arguments, sim_specific_arguments, launch_server, episode_max_time):

    events_path = os.path.join(arguments.deviations_folder, "events.txt")
    deviations_path = os.path.join(arguments.deviations_folder, 'deviations.txt')
    main_camera_folder = os.path.join(arguments.deviations_folder, 'main_camera_data')

    if not os.path.exists(main_camera_folder):
        os.mkdir(main_camera_folder)

    model_id = arguments.model_id
    map = arguments.route_info["town_name"]
    counter = arguments.counter


    duration = episode_max_time
    time_scale = 1
    continuous = True
    if continuous == True:
        odometry_path = os.path.join(arguments.deviations_folder, 'odometry.txt')
        perception_obstacles_path = os.path.join(arguments.deviations_folder, 'perception_obstacles.txt')
        path_list = [odometry_path, perception_obstacles_path, main_camera_folder, deviations_path]

        from multiprocessing import Process, Queue
        q = Queue()
        p = Process(target=run_sim_with_initialization, args=(q, duration, time_scale, map, sim_specific_arguments, arguments, customized_data, model_id, events_path))
        p.daemon = True
        p.start()
        receive_zmq(q, path_list, arguments.record_every_n_step)
        p.terminate()


    else:
        measurements_path = os.path.join(arguments.deviations_folder, 'measurements.txt')

        sim = initialize_sim()
        step_time = 30
        step_rate = 1.0 / step_time
        steps = int(duration * step_rate)

        cur_values = emptyobject(min_d=10000, d_angle_norm=1)

        for i in range(steps):
            sim.run(time_limit=step_time, time_scale=1)
            if i % arguments.record_every_n_step == 0:
                save_camera(ego, main_camera_folder, counter, i)

            save_measurement(ego, measurements_path)
            gather_info(ego, other_agents, cur_values, deviations_path)

            d_to_dest = norm_2d(ego.transform.position, destination.position)
            # print('d_to_dest', d_to_dest)
            if d_to_dest < 5:
                print('ego car reachs destination successfully')
                break

            if accident_happen:
                break



if __name__ == '__main__':
    atexit.register(exit_handler)
    map = "Borregas Ave"
    config = [4, 4, 2, 3, 10, 50]
    run_svl_simulation(map, config)
