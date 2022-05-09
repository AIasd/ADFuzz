import numpy as np
import pandas
import os
import pickle
from collections import OrderedDict
from customized_utils import emptyobject, make_hierarchical_dir
from no_simulation_function_script.synthetic_functions import synthetic_function_dict

def assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds):
    for d in [fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds]:
        for k, v in d.items():
            assert not hasattr(search_space_info, k), k+'should not appear twice.'
            setattr(search_space_info, k, v)

def generate_fuzzing_content(fuzzing_arguments, scenario_labels, scenario_label_types, min_bounds, max_bounds):
    labels = scenario_labels
    mask = scenario_label_types

    parameters_min_bounds = {scenario_label+'_min':min_bound for min_bound, scenario_label in zip(min_bounds, scenario_labels)}
    parameters_max_bounds = {scenario_label+'_max':max_bound for max_bound, scenario_label in zip(max_bounds, scenario_labels)}
    parameters_distributions = OrderedDict({label:"uniform" for label in labels})
    n_var = len(scenario_labels)

    search_space_info = emptyobject()
    fixed_hyperparameters = {}
    assign_key_value_pairs(search_space_info, fixed_hyperparameters, parameters_min_bounds, parameters_max_bounds)

    fuzzing_content = emptyobject(labels=labels, mask=mask, parameters_min_bounds=parameters_min_bounds, parameters_max_bounds=parameters_max_bounds, parameters_distributions=parameters_distributions, customized_constraints=[], customized_center_transforms=[], n_var=n_var, fixed_hyperparameters=fixed_hyperparameters, search_space_info=search_space_info, keywords_dict={})

    return fuzzing_content

def run_no_simulation(x, fuzzing_content, fuzzing_arguments, sim_specific_arguments, dt_arguments, launch_server, counter, port):

    from no_simulation_function_script.no_simulation_objectives_and_bugs import check_bug, classify_bug_type


    print('synthetic_function', sim_specific_arguments.synthetic_function)
    if sim_specific_arguments.synthetic_function == '':
        synthetic_function = lambda x:[x[0]+x[1]]
    elif sim_specific_arguments.synthetic_function in synthetic_function_dict:
        synthetic_function = synthetic_function_dict[sim_specific_arguments.synthetic_function]
    else:
        raise

    objectives = synthetic_function(x)

    is_bug = check_bug(objectives)
    bug_type, _ = classify_bug_type(objectives)

    run_info = {
    'x': x,
    'objectives': objectives,
    'is_bug': is_bug,
    'bug_type': bug_type,
    'mask': fuzzing_content.mask,
    'labels': fuzzing_content.labels,

    # for analysis
    'fuzzing_content': fuzzing_content,
    'fuzzing_arguments': fuzzing_arguments,
    'sim_specific_arguments': sim_specific_arguments
    }

    make_hierarchical_dir([fuzzing_arguments.parent_folder, str(counter)])

    with open(os.path.join(fuzzing_arguments.parent_folder, str(counter), 'cur_info.pickle'), 'wb') as f_out:
        pickle.dump(run_info, f_out)

    return objectives, run_info

def initialize_no_simulation_specific(fuzzing_arguments):
    objective_labels = fuzzing_arguments.objective_labels
    synthetic_function = fuzzing_arguments.synthetic_function

    sim_specific_arguments = emptyobject(objective_labels=objective_labels, synthetic_function=synthetic_function)

    return sim_specific_arguments
