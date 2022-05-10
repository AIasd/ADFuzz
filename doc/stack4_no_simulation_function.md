### Run Fuzzing
```
# GA
python ga_fuzzing.py --simulator no_simulation_function --n_gen 10 --pop_size 50 --algorithm_name nsga2 --has_run_num 500 --n_offsprings 200 --only_run_unique_cases 0 --use_unique_bugs 0 --synthetic_function four_modes
```

```
# Random
python ga_fuzzing.py --simulator no_simulation_function --n_gen 10 --pop_size 50 --algorithm_name random --has_run_num 500 --n_offsprings 200 --only_run_unique_cases 0 --use_unique_bugs 0 --synthetic_function four_modes
```

#### Customized function
One can change the function `customized` in `no_simulation_function_script/synthetic_functions.py` to any query function one wants that takes in a query `x` and return the query result/objectives `[f]`. Note one needs to change the variables in `ga_fuzzing.py` after the line `# These fields need to be set to be consistent with the synthetic_function used` according to the function one uses for the fuzzing process to function properly.

One might also want to change `check_bug` in `no_simulation_function_script/run_results_no_simulation` to define what the function determining if a bug happens based on the result/objectives `[f]`.

Finally, run (the following uses the `random` algorithm for sanity check):
```
# Random
python ga_fuzzing.py --simulator no_simulation_function --n_gen 10 --pop_size 50 --algorithm_name random --has_run_num 500 --n_offsprings 200 --only_run_unique_cases 0 --use_unique_bugs 0 --synthetic_function customized
```
