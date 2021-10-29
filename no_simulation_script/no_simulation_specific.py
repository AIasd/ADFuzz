import numpy as np
import pandas
from collections import OrderedDict
from customized_utils import emptyobject

def assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds):
    for d in [fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds]:
        for k, v in d.items():
            assert not hasattr(search_space_info, k), k+'should not appear twice.'
            setattr(search_space_info, k, v)

def generate_fuzzing_content():
    field_info = [
    ('ego_pos', 'int', 0, 5),
    ('ego_init_speed', 'int', 0, 7),
    ('other_pos', 'int', 0, 5),
    ('other_init_speed', 'int', 0, 5),
    ('ped_delay', 'int', 0, 4),
    ('ped_init_speed', 'int', 0, 1)
    ]
    labels = [d[0] for d in field_info]
    mask = [d[1] for d in field_info]
    parameters_min_bounds = {d[0]+'_min':d[2] for d in field_info}
    parameters_max_bounds = {d[0]+'_max':d[3] for d in field_info}
    parameters_distributions = OrderedDict()
    for label in labels:
        parameters_distributions[label] = "uniform"
    n_var = len(field_info)

    fixed_hyperparameters = {}

    search_space_info = emptyobject()
    assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds)

    keywords_dict = {}

    fuzzing_content = emptyobject(labels=labels, mask=mask, parameters_min_bounds=parameters_min_bounds, parameters_max_bounds=parameters_max_bounds, parameters_distributions=parameters_distributions, customized_constraints=[], customized_center_transforms=[], n_var=n_var, fixed_hyperparameters=fixed_hyperparameters, search_space_info=search_space_info, keywords_dict={})

    return fuzzing_content

def run_no_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port):
    from no_simulation_script.no_simulation_objectives_and_bugs import check_bug, classify_bug_type

    if sim_specific_arguments.oracle is None:
        sim_specific_arguments.oracle = {}

        oracle_keys = sim_specific_arguments.df[fuzzing_content.labels].to_numpy()
        # [100 105 110 115 120 125] -> [0,...,5]
        # [10 20 30 40 50 60 70 80]
        # [ 0  5 10 15 20 25]
        # [10 20 30 40 50 60]
        # [0 1 2 3 4]
        # [10 20]

        for i in range(oracle_keys.shape[1]):
            cur_x_i = oracle_keys[:, i]
            cur_x_i_uq = np.unique(cur_x_i)
            cur_x_i_uq_sorted = sorted(cur_x_i_uq)
            new_x_i = (cur_x_i - cur_x_i_uq_sorted[0]) / (cur_x_i_uq_sorted[1] - cur_x_i_uq_sorted[0])
            oracle_keys[:, i] = new_x_i


        oracle_values = sim_specific_arguments.df[sim_specific_arguments.objective_labels].to_numpy()

        for k, v in zip(oracle_keys, oracle_values):
            v[v == np.inf] = 9999
            v[v == False] = 0
            v[v == True] = 1
            # print('v', v)
            sim_specific_arguments.oracle[tuple(k)] = v


    objectives = sim_specific_arguments.oracle[tuple(x)]
    is_bug = check_bug(objectives)
    bug_type, _ = classify_bug_type(objectives)


    run_info = {
    'x': x,
    'objectives': objectives,
    'is_bug': is_bug,
    'bug_type': bug_type,
    'mask': fuzzing_content.mask,
    'labels': fuzzing_content.labels
    }

    return objectives, run_info



def initialize_no_simulation_specific(fuzzing_arguments):
    assert fuzzing_arguments.no_simulation_data_path, 'no path is specified for fuzzing_arguments.no_simulation_data_path'

    df = pandas.read_csv(fuzzing_arguments.no_simulation_data_path).fillna(0)
    objective_labels = fuzzing_arguments.objective_labels

    sim_specific_arguments = emptyobject(df=df, objective_labels=objective_labels, oracle=None)

    return sim_specific_arguments
