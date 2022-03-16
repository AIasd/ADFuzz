import numpy as np
import pandas
from collections import OrderedDict
from customized_utils import emptyobject

def assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds):
    for d in [fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds]:
        for k, v in d.items():
            assert not hasattr(search_space_info, k), k+'should not appear twice.'
            setattr(search_space_info, k, v)

def generate_fuzzing_content(fuzzing_arguments, scenario_labels, scenario_label_types):
    objective_labels = fuzzing_arguments.objective_labels

    assert fuzzing_arguments.no_simulation_data_path, 'no path is specified for fuzzing_arguments.no_simulation_data_path'

    df = pandas.read_csv(fuzzing_arguments.no_simulation_data_path).fillna(0)
    objective_labels = fuzzing_arguments.objective_labels

    parameters_min_bounds = {}
    parameters_max_bounds = {}

    for scenario_label in scenario_labels:
        sl_min = df[scenario_label].min()
        sl_max = df[scenario_label].max()

        parameters_min_bounds[scenario_label+'_min'] = sl_min
        parameters_max_bounds[scenario_label+'_max'] = sl_max

    labels = scenario_labels
    mask = scenario_label_types

    parameters_distributions = OrderedDict()
    for label in labels:
        parameters_distributions[label] = "uniform"
    n_var = len(scenario_labels)

    fixed_hyperparameters = {}

    search_space_info = emptyobject()
    assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds)

    keywords_dict = {}

    fuzzing_content = emptyobject(labels=labels, mask=mask, parameters_min_bounds=parameters_min_bounds, parameters_max_bounds=parameters_max_bounds, parameters_distributions=parameters_distributions, customized_constraints=[], customized_center_transforms=[], n_var=n_var, fixed_hyperparameters=fixed_hyperparameters, search_space_info=search_space_info, keywords_dict={})

    return fuzzing_content

def run_no_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port):
    from no_simulation_script.no_simulation_objectives_and_bugs import check_bug, classify_bug_type

    if sim_specific_arguments.oracle is None:
        from sklearn.neighbors import KDTree

        oracle_keys = sim_specific_arguments.df[fuzzing_content.labels].to_numpy()
        oracle = KDTree(oracle_keys, leaf_size=10)
    else:
        oracle = sim_specific_arguments.oracle

    # print('oracle_keys.shape', oracle_keys.shape)
    # print('x.shape', np.expand_dims(x, 0).shape)

    _, inds = oracle.query(np.expand_dims(x, 0), k=1)


    oracle_values = sim_specific_arguments.df[fuzzing_arguments.objective_labels].to_numpy()
    oracle_values[oracle_values==False] = 0
    oracle_values[oracle_values==True] = 1
    oracle_values[oracle_values==np.inf] = 9999

    objectives = oracle_values[inds[0, 0]]


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
