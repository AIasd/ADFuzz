# ADFuzz

## Introduction
A Software Package for Fuzzing Autonomous Driving Systems in Simulators



## Install pyenv and python3.8

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

## Environment Setup
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

### No Simulation
#### Setup
Need to prepare data in csv format (A small dataset will be provided as an example).
#### Run Fuzzing
```
# GA-UN
python ga_fuzzing.py --simulator no_simulation --n_gen 10 --pop_size 20 --algorithm_name nsga2-un --has_run_num 200 --no_simulation_data_path no_simulation_script/grid.csv --n_offsprings 50

# AVFuzzer
python ga_fuzzing.py --simulator no_simulation --n_gen 100 --pop_size 4 --algorithm_name avfuzzer --has_run_num 200 --no_simulation_data_path no_simulation_script/grid.csv --only_run_unique_cases 0


```

### SVL2021.3+Apollo Master
#### Setup
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
and put the folder  `channel_extraction` inside `apollo/cyber/python/cyber_py3/`. Note that this step is preferred to be done before building apollo `./apollo.sh build_opt_gpu` to avoid extra building step.


#### Create Apollo Master in Vehicles
SVL does not have a default "Apollo Master" for "Lincoln2017MKZ" under "Vehicles". To create one, one can duplicate "Apollo 5.0" and then add sensors "Clock Sensor" and "Signal Sensor" from "Apollo 6.0 (modular testing)".




### Other preparation
Need to change the field `model_id` in svl_specific to one's own model_id on svl web UI.

#### Run Fuzzing
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
If using apollo with ground-truth traffic signal:
```
python ga_fuzzing.py --simulator svl --n_gen 10 --pop_size 50 --algorithm_name nsga2-un --has_run_num 500 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 30 --ego_car_model apollo_6_with_signal --route_type 'BorregasAve_left' --scenario_type 'turn_left_one_ped_and_one_vehicle' --record_every_n_step 5
```
Or if using apollo with ground-truth perception:
```
python ga_fuzzing.py --simulator svl --n_gen 2 --pop_size 2 --algorithm_name nsga2 --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 30 --ego_car_model apollo_6_modular_2gt --route_type 'BorregasAve_left' --scenario_type 'turn_left_one_ped_and_one_vehicle' --record_every_n_step 5
```

Run AVFuzzer
```
python ga_fuzzing.py --simulator svl --n_gen 175 --pop_size 4 --algorithm_name avfuzzer --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 30 --ego_car_model apollo_6_with_signal --only_run_unique_cases 0
```


### CARLA0.9.9+LBC
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


#### Run Fuzzing
```
# NSGA2-UN
python ga_fuzzing.py -p 2015 -s 8791 -d 8792 --n_gen 2 --pop_size 2 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5

# NSGA2-UN-ADV-NN
python ga_fuzzing.py -p 2021 -s 8795 -d 8796 --n_gen 15 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path <path-to-warm-up-run-folder> --warm_up_len 500 --check_unique_coeff 0 0.1 0.5 --has_display 0 --record_every_n_step 5 --only_run_unique_cases 1

# AVFuzzer
python ga_fuzzing.py -p 2018 -s 8793 -d 8794 --n_gen 200 --pop_size 4 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name avfuzzer --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --has_display 0 --record_every_n_step 5 --only_run_unique_cases 0 --n_offsprings 50
```
