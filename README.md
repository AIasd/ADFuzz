# ADFuzz

## Introduction
An open-source software package for fuzzing autonomous driving systems in high-fidelity simulators. It is also currently actively maintained and developed.

### Requirements
* OS: Ubuntu 18.04, 20.04
* CPU: >= 8 cores
* GPU: >= 6GB memory (>= 8GB if the perception module of Apollo is used)

### Current support of stacks
- LBC + CARLA 0.9.9
- Apollo(6.0 or later) + SVL 2021.3
- No Simulation

### Current support of algorithms and variations
ADFuzz currently support several algorithms and variations listed below. The relevant algorithm_name and key parameters are also mentioned.

#### Algorithms
- NSGA2-SM (`-a nsga2 --rank_mode regression_nn --use_single_objective 0 --only_run_unique_cases 0 --regression_nn_use_running_data 0 --warm_up_path <path-to-warm-up-folder> --warm_up_len 500`)
- NSGA2-DT (`-a nsga2-dt --use_single_objective 0 --only_run_unique_cases 0 --outer_iterations 3`)
- AV-Fuzzer (`-a avfuzzer --only_run_unique_cases 0`)
- AutoFuzz (GA-UN-NN-GRAD) (`-a nsga2-un --rank_mode adv_nn` )

#### Baselines/Variations
- Random (`-a random --only_run_unique_cases 0`)
- Random-UN (`-a random-un`)
- GA (`-a nsga2 --only_run_unique_cases 0`)
- GA-UN (`-a nsga2-un`)
- NSGA2-UN-SM-A (`-a nsga2 --rank_mode regression_nn --use_single_objective 0`)

#### Additional Explanation
It should be noted that for NSGA2-SM, additional parameters like `warm_up_path` and `warm_up_len` must be specified. For AutoFuzz (GA-UN-NN-GRAD) and NSGA2-UN-SM-A, they can also be specified. `warm_up_path` refers to the result folder of a run of the initial warm-up stage. Algorithms like Random and GA are commonly used. `warm_up_len` refers to the results of how many simulation instances from this warm-up stage are leveraged.

## Found Traffic Violation Demos
### pid-1 controller collides with a pedestrian:

<img src="gif_demos/autopilot_pid1_35_rgb_with_car.gif" width="40%" height="40%"/>

### pid-2 controller collides with the stopped leading car:

<img src="gif_demos/pid_pid2_39_rgb_with_car.gif" width="40%" height="40%"/>

### lbc controller is going wrong lane:

<img src="gif_demos/lbc_58_rgb_with_car.gif" width="40%" height="40%"/>

### Apollo6.0 collides with a school bus:
<img src="gif_demos/apollo_schoolbus_collision.gif" width="40%" height="40%"/>


<!-- ## Uniqueness Definition for Traffic Violation Demos
### A Found Traffic Violation
<img src="gif_demos/lbc_left_ped_8.gif" width="40%" height="40%"/>

### A Highly Similar One
<img src="gif_demos/lbc_left_ped_971.gif" width="40%" height="40%"/>

### A Distinct One
<img src="gif_demos/lbc_left_vehicle_982.gif" width="40%" height="40%"/> -->




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

## Documentations for the setup of each stack
[CARLA0.9.9+LBC](https://github.com/AIasd/ADFuzz/blob/main/doc/stack1_carla_lbc.md)

[SVL2021.3+Apollo Master](https://github.com/AIasd/ADFuzz/blob/main/doc/stack2_svl_apollo.md)

[No Simulation (Dataset)](https://github.com/AIasd/ADFuzz/blob/main/doc/stack3_no_simulation_dataset.md)


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

### Check the number of unique violations in terms of coverage
In `openpilot/tools/sim/op_script`,
```
python trajectory_analysis.py -p <parent folder> -f <fusion folder>
```

### Analyze fusion errors in terms of objectives
Rename the original folder as "original" and the rerun_folder as "rerun_2.5_best_sensor". Put them into the same parent folder.
In `openpilot/tools/sim/op_script`,
```
python analyze_fusion_errors.py -p <parent_folder>
```
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
This repo leverages code from [Carla Challenge (with LBC supported)](https://github.com/bradyz/2020_CARLA_challenge) and [pymoo](https://github.com/anyoptimization/pymoo)
