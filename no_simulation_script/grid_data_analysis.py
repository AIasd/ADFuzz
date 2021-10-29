import pandas
import numpy as np
df = pandas.read_csv('grid.csv')
print(df.head())
for col in df.columns:
    print(col)
print(np.unique(df['ego_pos'].to_numpy()))
print(np.unique(df['ego_init_speed'].to_numpy()))
print(np.unique(df['other_pos'].to_numpy()))
print(np.unique(df['other_init_speed'].to_numpy()))
print(np.unique(df['ped_delay'].to_numpy()))
print(np.unique(df['ped_init_speed'].to_numpy()))

print(df[['ego_pos', 'ego_init_speed']].to_numpy().shape)
