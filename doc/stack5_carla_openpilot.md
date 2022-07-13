### Setup
#### Requirements
* Monitor (i.e., due to the limitation of OpenPilot, the simulation can only run on a machine with a monitor/virtual monitor)
* OS: Ubuntu 20.04
* CPU: at least 6 cores
* GPU: at least 6GB memory
* Openpilot 0.8.5 (customized)
* Carla 0.9.11

#### Directory Structure
~(home folder)
```
├── openpilot
├── Documents
│   ├── self-driving-cars (created by the user manually)
│   │   ├── ADFuzz
│   │   ├── carla_0911_rss
```
Note: one can create link for these folders at these paths if one cannot put them in these paths.

#### Install OpenPilot 0.8.5 (customized)
In `~`,
```
git clone https://github.com/AIasd/openpilot.git
```
In `~/openpilot`,
```
./tools/ubuntu_setup.sh
```
In `~/openpilot`, compile Openpilot
```
scons -j $(nproc)
```

#### Common Python Path Issue
Make sure the python path is set up correctly through pyenv, in particular, run
```
which python
```
One should see the following:
```
~/.pyenv/shims/python
```
Otherwise, one needs to follow the displayed instructions after running
```
pyenv init
```

#### Common Compilation Issue
clang 10 is needed. To install it, run
```
sudo apt install clang
```

#### Common OpenCL Issue
Your environment needs to support opencl 2.0+ in order to run `scons` successfully (when using `clinfo`, it must show something like  "your OpenCL library only supports OpenCL <2.0+>")


#### Install Carla 0.9.11
In `~/Documents/self-driving-cars`,
```
curl -O https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11_RSS.tar.gz
mkdir carla_0911_rss
tar -xvzf CARLA_0.9.11_RSS.tar.gz -C carla_0911_rss
```

In `~/Documents/self-driving-cars/carla_0911_rss/PythonAPI/carla/dist`,
```
easy_install carla-0.9.11-py3.7-linux-x86_64.egg
```

#### Install additional maps
In `~/Documents/self-driving-cars/carla_0911_rss`,
```
curl -O https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.11.tar.gz
mv AdditionalMaps_0.9.11.tar.gz Import/

```
and then run
```
./ImportAssets.sh
```

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
