### Setup
Need to prepare data in csv format similar to `no_simulation_dataset_script/grid.csv`.
### Run Fuzzing
```
# GA-UN
python ga_fuzzing.py --simulator no_simulation_dataset --n_gen 10 --pop_size 20 --algorithm_name nsga2-un --has_run_num 200 --no_simulation_data_path no_simulation_dataset_script/grid.csv --n_offsprings 50

# AVFuzzer
python ga_fuzzing.py --simulator no_simulation_dataset --n_gen 50 --pop_size 4 --algorithm_name avfuzzer --has_run_num 200 --no_simulation_data_path no_simulation_dataset_script/grid.csv --only_run_unique_cases 0
```
