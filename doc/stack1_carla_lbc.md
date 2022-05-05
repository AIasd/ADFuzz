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
<!-- Then, run
```
cd carla_0994_no_rss/PythonAPI/carla/dist
easy_install carla-0.9.9-py3.7-linux-x86_64.egg
``` -->

#### Download a LBC pretrained model
LBC model is one of the models supported to be tested. A pretrained-model's checkpoint can be found [here](https://app.wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/files)

Go to the "files" tab, and download the model weights, named "epoch=24.ckpt". Move this model's checkpoint to the `models` folder (May need to create a folder named `models` under the folder `carla_lbc`).


### Run Fuzzing
```
# GA-UN
python ga_fuzzing.py -p 2015 --n_gen 2 --pop_size 2 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 4 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --record_every_n_step 5 --debug 0

# GA-UN-NN-GRAD
python ga_fuzzing.py -p 2021 --n_gen 15 --pop_size 50 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name nsga2-un --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --rank_mode adv_nn --warm_up_path <path-to-warm-up-run-folder> --warm_up_len 500 --check_unique_coeff 0 0.1 0.5 --has_display 0 --record_every_n_step 5 --only_run_unique_cases 1 --record_every_n_step 5 --debug 0

# AVFuzzer
python ga_fuzzing.py -p 2018 --n_gen 200 --pop_size 4 -r 'town07_front_0' -c 'go_straight_town07' --algorithm_name avfuzzer --has_run_num 700 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.1 0.5 --has_display 0 --record_every_n_step 5 --only_run_unique_cases 0 --n_offsprings 50 --record_every_n_step 5 --debug 0
```

### Rerun
By default, all the scenarios in the bugs folder under the parent folder will be rerun and visualized.
```
python carla_lbc/rerun_and_data_analysis/rerun_scenario.py --parent_folder path_to_the_parent_folder_to_rerun
```
