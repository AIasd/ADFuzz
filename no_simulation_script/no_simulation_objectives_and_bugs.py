'''
Functions here need to be updated according to the dataset used. Some functions are only used by certain methods.
'''
import numpy as np
from customized_utils import is_distinct_vectorized

# ---------------- Bug, Objective -------------------
def check_bug(objectives):
    # speed needs to be larger than 0.1 to avoid false positive
    return (objectives[-3] > 0.1 and objectives[-2]) or objectives[-1]


def process_specific_bug(
    bug_type_ind, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
):
    if len(bugs) == 0:
        return [], [], 0
    verbose = True
    chosen_bugs = np.array(bugs_type_list) == bug_type_ind

    specific_bugs = np.array(bugs)[chosen_bugs]
    specific_bugs_inds_list = np.array(bugs_inds_list)[chosen_bugs]

    specific_distinct_inds = is_distinct_vectorized(specific_bugs, [], mask, xl, xu, p, c, th, verbose=verbose)
    unique_specific_bugs = specific_bugs[specific_distinct_inds]

    unique_specific_bugs_inds_list = specific_bugs_inds_list[specific_distinct_inds]

    return (
        list(unique_specific_bugs),
        list(unique_specific_bugs_inds_list),
        len(unique_specific_bugs),
    )

def classify_bug_type(objectives, object_type=''):
    bug_str = ''
    bug_type = 3
    if objectives[-3] > 0.1 and objectives[-2]:
        bug_str = 'collision'
        bug_type = 1
    elif objectives[-1]:
        bug_str = 'oob'
        bug_type = 2
    return bug_type, bug_str

def get_unique_bugs(
    X, objectives_list, mask, xl, xu, unique_coeff, objective_weights, return_mode='unique_inds_and_interested_and_bugcounts', consider_interested_bugs=1, bugs_type_list=[], bugs=[], bugs_inds_list=[], trajectory_vector_list=[]
):
    p, c, th = unique_coeff
    # hack:
    if len(bugs) == 0:
        for i, (x, objectives) in enumerate(zip(X, objectives_list)):
            if check_bug(objectives):
                bug_type, _ = classify_bug_type(objectives)
                bugs_type_list.append(bug_type)
                bugs.append(x)
                bugs_inds_list.append(i)

    (
        unique_collision_bugs,
        unique_collision_bugs_inds_list,
        unique_collision_num,
    ) = process_specific_bug(
        1, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )
    (
        unique_oob_bugs,
        unique_oob_bugs_inds_list,
        unique_oob_num,
    ) = process_specific_bug(
        2, bugs_type_list, bugs_inds_list, bugs, mask, xl, xu, p, c, th
    )

    unique_bugs = unique_collision_bugs + unique_oob_bugs
    unique_bugs_num = len(unique_bugs)
    unique_bugs_inds_list = unique_collision_bugs_inds_list + unique_oob_bugs_inds_list

    interested_unique_bugs = unique_bugs
    num_of_collisions = np.sum(np.array(bugs_type_list)==1)
    num_of_oob = np.sum(np.array(bugs_type_list)==2)

    if return_mode == 'unique_inds_and_interested_and_bugcounts':
        return unique_bugs, unique_bugs_inds_list, interested_unique_bugs, [num_of_collisions, num_of_oob,
        unique_collision_num, unique_oob_num]
    elif return_mode == 'return_bug_info':
        return unique_bugs, (bugs, bugs_type_list, bugs_inds_list, interested_unique_bugs)


def choose_weight_inds(objective_weights):
    weight_inds = np.arange(0, len(objective_weights))
    return weight_inds

def determine_y_upon_weights(objective_list, objective_weights):
    y = np.zeros(len(objective_list))
    for i, obj in enumerate(objective_list):
        y[i] = check_bug(obj)
    return y

def get_all_y(objective_list, objective_weights):
    y_list = np.zeros((2, len(objective_list)))
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    for i, obj in enumerate(objective_list):
        if collision_activated:
            y_list[0, i] = (obj[-3] > 0.1) & (obj[-2] == 1)
        if offroad_activated:
            y_list[1, i] = (obj[-1] == 1)

    return y_list
# ---------------- Bug, Objective -------------------
