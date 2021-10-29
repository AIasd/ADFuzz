#!/usr/bin/env python

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time
import argparse
import atexit
sys.path.append('.')



carla_root = '../carla_0994_no_rss'
sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
sys.path.append(carla_root+'/PythonAPI/carla')
sys.path.append(carla_root+'/PythonAPI')

from customized_utils import exit_handler
from carla_specific_utils.carla_specific_tools import start_server

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

_HOST_ = '127.0.0.1'
_PORT_ = 2000
_SLEEP_TIME_ = 1


def main(map):
    start_server(_PORT_)
    atexit.register(exit_handler, [_PORT_])
    client = carla.Client(_HOST_, _PORT_)
    client.set_timeout(2.0)
    print('map', map)
    client.load_world(map)
    world = client.get_world()

	# print(help(t))
	# print("(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z))
    while(True):
        t = world.get_spectator().get_transform()
        # coordinate_str = "(x,y) = ({},{})".format(t.location.x, t.location.y)
        coordinate_str = "(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z)
        print (coordinate_str)
        time.sleep(_SLEEP_TIME_)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Print Spectator Coordinates')
    argparser.add_argument(
        '--map',
        default='Town04',
        help='which map to use')
    args = argparser.parse_args()
    main(args.map)
