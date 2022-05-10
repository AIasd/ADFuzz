### Visualization
#### General Routine
Set folder path and visualization parameters at the end of `analysis_utils/visualize.py` based on the corresponding comments.

Then, run
```
python analysis_utils/visualize.py
```

#### Example - No Simulation (Function)
Run fuzzing for a specified function using an algorithm one wants to use based on the instruction at [No Simulation (Function)](https://github.com/AIasd/ADFuzz/blob/main/doc/stack4_no_simulation_function.md). Here, we use the algorithm `GA` for the synthetic function `four_modes`.
```
python ga_fuzzing.py --simulator no_simulation_function --n_gen 10 --pop_size 50 --algorithm_name nsga2 --has_run_num 500 --n_offsprings 200 --only_run_unique_cases 0 --use_unique_bugs 0 --synthetic_function four_modes
```

In `analysis_utils/visualize.py`, change `folder_path` to the folder containing the fuzzing result. In this case, it should have a path similar to `no_simulation_function_script/run_results_no_simulation/<algorithm_name>/<synthetic_function>/<specific folder name with fuzzing starting time>'`.

Make sure `chosen_labels` in `analysis_utils/visualize.py` is a subset of `scenario_labels` in `ga_fuzzing.py` (under the elif block of 'no_simulation_function').

For details of changing other visualization parameters, check out the comments in at the end of `analysis_utils/visualize.py`.

Then, run
```
python analysis_utils/visualize.py
```

The resulted figure is saved in the `folder_path`. The generated figure is shown below:
![plain_2_500_ga](doc/figures/plain_2_500_ga.jpg)
