### Run Fuzzing
```
# GA
python ga_fuzzing.py --simulator no_simulation_function --n_gen 10 --pop_size 50 --algorithm_name nsga2 --has_run_num 100 --n_offsprings 200 --only_run_unique_cases 0 --use_unique_bugs 0 --synthetic_function four_modes

# Random
python ga_fuzzing.py --simulator no_simulation_function --n_gen 10 --pop_size 50 --algorithm_name random --has_run_num 100 --n_offsprings 200 --only_run_unique_cases 0 --use_unique_bugs 0 --synthetic_function four_modes
```
