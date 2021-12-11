# ADFuzz

## Introduction
An open-source software package for fuzzing autonomous driving systems in high-fidelity simulators.

### Current support of stacks
- LBC + CARLA 0.9.9
- Apollo(6.0/master) + SVL 2021.3
- No Simulation

### Current support of algorithms
- AutoFuzz (GA-UN-NN-GRAD)
- NSGA2-SM
- NSGA2-DT
- AV-Fuzzer


## Found Traffic Violation Demos
### pid-1 controller collides with a pedestrian:

<img src="gif_demos/autopilot_pid1_35_rgb_with_car.gif" width="40%" height="40%"/>

### pid-2 controller collides with the stopped leading car:

<img src="gif_demos/pid_pid2_39_rgb_with_car.gif" width="40%" height="40%"/>

### lbc controller is going wrong lane:

<img src="gif_demos/lbc_58_rgb_with_car.gif" width="40%" height="40%"/>

### Apollo6.0 collides with a school bus:
<img src="gif_demos/apollo_schoolbus_collision.gif" width="40%" height="40%"/>


## Uniqueness Definition for Traffic Violation Demos
### A Found Traffic Violation
<img src="gif_demos/lbc_left_ped_8.gif" width="40%" height="40%"/>

### A Highly Similar One
<img src="gif_demos/lbc_left_ped_971.gif" width="40%" height="40%"/>

### A Distinct One
<img src="gif_demos/lbc_left_vehicle_982.gif" width="40%" height="40%"/>




## Preparation
### Install pyenv and python3.8

install pyenv
```
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

install python
```
PATH=$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH
pyenv install -s 3.8.5
pyenv global 3.8.5
pyenv rehash
eval "$(pyenv init -)"
```

add the following lines to the end of `~/.bashrc` to make sure pyenv is active when openning a new terminal
```
PATH=$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH
eval "$(pyenv init -)"
```

### Environment Setup
In `~/Docuements/self-driving-cars`,
```
git clone https://github.com/AIasd/ADFuzz.git
```

Install environment
```
pip3 install -r requirements.txt
```

Install pytorch on its official website via pip.

Install pytroch-lightening
```
pip3 install pytorch-lightning==0.8.5
```



## STACK1: CARLA0.9.9+LBC
### Setup
#### Installation of Carla 0.9.9.4
This code uses CARLA 0.9.9.4. You will need to first install CARLA 0.9.9.4, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.

For convenience, the following commands can be used to install carla 0.9.9.4.

Download CARLA_0.9.9.4.tar.gz and AdditionalMaps_0.9.9.4.tar.gz from [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9), put it at the same level of this repo, and run
```
mkdir carla_0994_no_rss
tar -xvzf CARLA_0.9.9.4.tar.gz -C carla_0994_no_rss
```
move `AdditionalMaps_0.9.9.4.tar.gz` to `carla_0994_no_rss/Import/` and in the folder `carla_0994_no_rss/` run:
```
./ImportAssets.sh
```
Then, run
```
cd carla_0994_no_rss/PythonAPI/carla/dist
easy_install carla-0.9.9-py3.7-linux-x86_64.egg
```

#### Download a LBC pretrained model
LBC model is one of the models supported to be tested. A pretrained-model's checkpoint can be found [here](https://app.wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/files)

Go to the "files" tab, and download the model weights, named "epoch=24.ckpt". Move this model's checkpoint to the `models` folder (May need to create `models` folder under this repo's folder).


### Run Fuzzing
```
# GA-UN
python ga_fuzzing.py -p 2015 -s 8791 -d 8792 --n_gen 2 --pop_size 2 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5

# GA-UN-NN-GRAD
python ga_fuzzing.py -p 2021 -s 8795 -d 8796 --n_gen 15 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path <path-to-warm-up-run-folder> --warm_up_len 500 --check_unique_coeff 0 0.1 0.5 --has_display 0 --record_every_n_step 5 --only_run_unique_cases 1

# AVFuzzer
python ga_fuzzing.py -p 2018 -s 8793 -d 8794 --n_gen 200 --pop_size 4 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name avfuzzer --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --has_display 0 --record_every_n_step 5 --only_run_unique_cases 0 --n_offsprings 50
```



## STACK2: SVL2021.3+Apollo Master
### Setup
Install SVL2021.3 and Apollo Master following [the documentation of Running latest Apollo with SVL Simulator](https://www.svlsimulator.com/docs/system-under-test/apollo-master-instructions/).


#### Install SVL Python API
```
git clone https://github.com/lgsvl/PythonAPI.git
```
Following the installation procedure at [https://github.com/lgsvl/PythonAPI](https://github.com/lgsvl/PythonAPI)

#### Add channel_extraction
```
git clone https://github.com/AIasd/apollo_channel_extraction.git
```
and put the folder  `channel_extraction` inside `apollo/cyber/python/cyber_py3/` and put `generate_route.sh` inside `apollo`. Note that this step is preferred to be done before building apollo `./apollo.sh build_opt_gpu` to avoid extra building step.


#### Create Apollo Master in Vehicles
SVL does not have a default "Apollo Master" for "Lincoln2017MKZ" under "Vehicles". To create one, one can duplicate "Apollo 5.0" and then add sensors "Clock Sensor" and "Signal Sensor" from "Apollo 6.0 (modular testing)".


#### Other preparation
Need to change the field `model_id` in svl_specific to one's own model_id on svl web UI.

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
If running GA-UN and using apollo with ground-truth traffic signal:
```
python ga_fuzzing.py --simulator svl --n_gen 2 --pop_size 2 --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 35 --ego_car_model apollo_6_with_signal --route_type 'BorregasAve_left' --scenario_type 'turn_left_one_ped_and_one_vehicle' --record_every_n_step 5 --n_offsprings 50
```
Or if running GA-UN and using apollo with ground-truth perception:
```
python ga_fuzzing.py --simulator svl --n_gen 2 --pop_size 2 --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 35 --ego_car_model apollo_6_modular --route_type 'BorregasAve_left' --scenario_type 'one_ped_crossing' --record_every_n_step 5 --n_offsprings 100
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
Inside docker,
```
sudo bash scripts/generate_map.sh YOUR_MAP_FOLDER_NAME
```
restart Dreamview to refresh the map list
```
bootstrap.sh stop && bootstrap.sh
```

### Additional Tools
[Cyber Recorder](https://github.com/ApolloAuto/apollo/blob/master/docs/cyber/CyberRT_Developer_Tools.md)



## STACK3: No Simulation
### Setup
Need to prepare data in csv format.
### Run Fuzzing
```
# GA-UN
python ga_fuzzing.py --simulator no_simulation --n_gen 10 --pop_size 20 --algorithm_name nsga2-un --has_run_num 200 --no_simulation_data_path no_simulation_script/grid.csv --n_offsprings 50

# AVFuzzer
python ga_fuzzing.py --simulator no_simulation --n_gen 50 --pop_size 4 --algorithm_name avfuzzer --has_run_num 200 --no_simulation_data_path no_simulation_script/grid.csv --only_run_unique_cases 0
```


<!-- ## STACK4: CARLA0.9.11+OpenPilot (TBD)
### Run Fuzzing
```
python ga_fuzzing.py --simulator carla_op --n_gen 10 --pop_size 50 --algorithm_name nsga2 --has_run_num 500 --episode_max_time 200 --only_run_unique_cases 0 --objective_weights 1 0 0 0 -1 -2 0 -m op --route_type 'Town06_Opt_forward'
```

### Rerun previous simulations
In `~/openpilot/tools/sim/op_script`,
```
python rerun_carla_op.py -p <path-to-the-parent-folder-consisting-of-single-simulation-runs-indexed-by-numbers>
```

### Rerun scenarios using the best sensor prediction
Move all the subfolders indexed of previously run simulation results in `~/Docuements/self-driving-cars/2020_CARLA_challenge/run_results_op` to `openpilot/tools/sim/op_script/rerun_folder`, then in `openpilot/tools/sim/op_script`,
```
python rerun_carla_op.py -p rerun_folder -m2 best_sensor -w 2.5
```

### Check unique violations
In `openpilot/tools/sim/op_script`,
```
python trajectory_analysis.py -p <parent folder> -f <fusion folder>
```
python trajectory_analysis.py -p rerun_folder/ -f rerun_op/nsga2/Town06_Opt_forward/Town06_Opt_forward/op/2021_12_10_17_12_58/non_bugs

-->




## Citing
If you use the project in your work, please consider citing it with:
```
@misc{zhong2021neural,
      title={Neural Network Guided Evolutionary Fuzzing for Finding Traffic Violations of Autonomous Vehicles},
      author={Ziyuan Zhong and Gail Kaiser and Baishakhi Ray},
      year={2021},
      eprint={2109.06126},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```


## Reference
This repo is partially built on top of [Carla Challenge (with LBC supported)](https://github.com/bradyz/2020_CARLA_challenge) and [pymoo](https://github.com/anyoptimization/pymoo)
