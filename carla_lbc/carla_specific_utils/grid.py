# 4488
grid_dict_one_ped_town07 = {'pedestrian_x_0': [-5+i for i in range(11)],
'pedestrian_y_0': [-8+i for i in range(17)],
'pedestrian_yaw_0': [15*i for i in range(24)]}
# 2904
grid_dict_one_ped_town05 = {'pedestrian_x_0': [-10+i*2 for i in range(11)],
'pedestrian_y_0': [-10+i*2 for i in range(11)],
'pedestrian_yaw_0': [15*i for i in range(24)]}

# 1440
grid_dict_one_npc_change_lane_town05 = {'vehicle_y_0': [-10+0.5*i for i in range(41)], 'vehicle_targeted_speed_0': [1+0.25*i for i in range(37)]}

# grid_dict_one_npc_change_lane_town05 = {'vehicle_y_0': [20 for i in range(1)], 'vehicle_targeted_speed_0': [10 for i in range(1)]}

# # town 07
# grid_dict_one_ped_town07 = {'pedestrian_x_0': [-5, 5],
# 'pedestrian_y_0': [-8, 8]}
#
# # town 05
# grid_dict_one_ped_town05 = {'pedestrian_x_0': [-9, 9],
# 'pedestrian_y_0': [-9, 9]}

grid_dict_dict = {'grid_dict_one_ped_town05': grid_dict_one_ped_town05, 'grid_dict_one_ped_town07': grid_dict_one_ped_town07, 'grid_dict_one_npc_change_lane_town05': grid_dict_one_npc_change_lane_town05}
