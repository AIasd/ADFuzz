# ADFuzz

## Introduction
A Software Package for Fuzzing Autonomous Driving Systems in Simulators


## Installation
In `~/Docuements/self-driving-cars`,
```
git clone https://github.com/ADFuzz/ADFuzz.git
```

Install environment
```
pip3 install -r requirements.txt
```


### No Simulation
#### Setup
Need to prepare data in csv format (A small dataset will be provided as an example).
#### Run Fuzzing
```
python ga_fuzzing.py --simulator no_simulation --n_gen 2 --pop_size 2 --algorithm_name nsga2 --has_run_num 4 --no_simulation_data_path <path-to-csv-data>
```

### SVL2021.2.2+Apollo Master
#### Setup
Install SVL2021.2.2 and Apollo Master following [the documentation of Running latest Apollo with SVL Simulator](https://www.svlsimulator.com/docs/system-under-test/apollo-master-instructions/).

#### Create Apollo Master in Vehicles
SVL does not have a default "Apollo Master" for "Lincoln2017MKZ" under "Vehicles". To create one, one can duplicate "Apollo 5.0" and then add sensors "Clock Sensor" and "Signal Sensor" from "Apollo 6.0 (modular testing)".

#### Run Fuzzing
Need to change the field `model_id` in svl_specific to one's own model_id on svl web UI.

Start Apollo and SVL API only respectively. Then in a separate terminal:
```
python ga_fuzzing.py --simulator svl --n_gen 2 --pop_size 2 --algorithm_name nsga2 --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --episode_max_time 30 --ego_car_model apollo_6_with_signal
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
python ga_fuzzing.py -p 2015 -s 8791 -d 8792 --n_gen 2 --pop_size 2 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.2 0.5
```
