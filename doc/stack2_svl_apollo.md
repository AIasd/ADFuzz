### Setup
Install SVL2021.3 and Apollo Master (tested upto Apollo 7.0) following [the documentation of Running latest Apollo with SVL Simulator](https://www.svlsimulator.com/docs/system-under-test/apollo-master-instructions/).


#### Install SVL Python API
```
git clone https://github.com/lgsvl/PythonAPI.git
```
Following the installation procedure at [https://github.com/lgsvl/PythonAPI](https://github.com/lgsvl/PythonAPI)

#### Add channel_extraction
```
git clone https://github.com/AIasd/apollo_channel_extraction.git
```
and put the folder  `channel_extraction` inside `apollo/cyber/python/cyber_py3/`. Note that this step is preferred to be done before building apollo `./apollo.sh build_opt_gpu` to avoid an extra building step.


#### Create an Configuration supporting Apollo 6.0 (or later) with the perception module in Vehicles Library
SVL does not have a default "Apollo 6.0 (or later)" for "Lincoln2017MKZ" under "Vehicles". To create one, on SVL web UI,

1. Under 'Vehicles Library > Lincoln2017MKZ > Sensor Configurations', clone "Apollo 5.0", rename it to "Apollo 6.0 (with Signal Sensor)".

2. Add the sensors "Clock Sensor", "Signal Sensor", and "3D Ground Truth" into "Apollo 6.0 (with Signal Sensor)" following those defined in "Apollo 6.0 (modular testing)".

3. Change the "Topic" and "X" of "3D Ground Truth" to "/apollo/perception/obstacles_gt" and "10" respectively.

Note the camera module for traffic light detection of Apollo 6.0 (and later) seems to still not work properly so ground-truth traffic signal is provided via "Signal Sensor".

#### Other preparation
One needs to change the used value of the variable `model_id` in `simulation_utils.py` and `svl_specific.py` to one's own model_id on SVL web UI. For example, if one set `ego_car_model` to `'apollo_6_with_signal'` when running fuzzing, one can replace `'9272dd1a-793a-45b2-bff4-3a160b506d75'` in `simulation_utils.py` and `svl_specific.py` with one's own vehicle configuration id (this can be found by clicking the id symbol of one's chosen 'Configuration Name' (e.g., "Apollo 6.0 (with Signal Sensor)") under the 'Actions' column of the 'Sensor Configurations' table under 'Vehicles Library > Lincoln2017MKZ > Sensor Configurations' on SVL web UI).

### Run Fuzzing
Start Apollo and SVL API only respectively following [the documentation of Running latest Apollo with SVL Simulator](https://www.svlsimulator.com/docs/system-under-test/apollo-master-instructions/).


Then in a second terminal:
Find apollo docker container id via:
```
docker ps
```
then entering the docker via:
```
docker exec -it <container name> /bin/bash
```
install zmq via pip in the docker:
```
pip install zmq
```
and finally run the channel_extraction
```
./bazel-bin/cyber/python/cyber_py3/channels_data_extraction/channels_extraction
```


Finally, in a third terminal:
If running GA-UN and using apollo with only ground-truth traffic signal:
```
python ga_fuzzing.py --simulator svl --n_gen 2 --pop_size 2 --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 35 --ego_car_model apollo_6_with_signal --route_type 'BorregasAve_left' --scenario_type 'turn_left_one_ped_and_one_vehicle' --record_every_n_step 5 --n_offsprings 50
```
Or if running GA-UN and using apollo with all ground-truth perception:
```
python ga_fuzzing.py --simulator svl --n_gen 2 --pop_size 2 --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 35 --ego_car_model apollo_6_modular_2gt --route_type 'BorregasAve_left' --scenario_type 'turn_left_one_ped_and_one_vehicle' --record_every_n_step 5 --n_offsprings 100
```

Or if running AVFuzzer
```
python ga_fuzzing.py --simulator svl --n_gen 50 --pop_size 4 --algorithm_name avfuzzer --has_run_num 100 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 35 --ego_car_model apollo_6_with_signal --only_run_unique_cases 0 --route_type 'BorregasAve_left' --scenario_type 'turn_left_one_ped_and_one_vehicle' --record_every_n_step 5
```

### Rerun
```
python svl_script/rerun_svl.py
```

### Add a new map
Copy `generate_map.sh` from [Apollo 5.0 (SVL fork)](https://github.com/lgsvl/apollo-5.0/blob/simulator/scripts/generate_map.sh) into the docker with the path `apollo/scripts`.

Inside docker,
```
sudo bash scripts/generate_map.sh YOUR_MAP_FOLDER_NAME
```

Restart Dreamview to refresh the map list
```
bootstrap.sh stop && bootstrap.sh
```


### Additional Tools
[Cyber Recorder](https://github.com/ApolloAuto/apollo/blob/master/docs/cyber/CyberRT_Developer_Tools.md)
