#  Script to compare two voltage tables to look for differences
# This could be useful when changing stringency parameters in voltage_calc.py
# both tables have to have the same dimensions

import pickle
import numpy as np
import matplotlib.pyplot as plt

table1 = 'v_tables/02Aug2024_MedResolution_Rext250_Rint70.dat'
table2 = 'v_tables/24Jan2025_MedResolution_Rext250_Rint70_volt_diff_300.dat'

with open(table1, 'rb') as combined_data:
    data = pickle.load(combined_data)
combined_data.close()

fp1 = data[0]
v_vals1 = data[1]
act_vals1 = data[2]

with open(table2, 'rb') as combined_data:
    data = pickle.load(combined_data)
combined_data.close()

fp2 = data[0]
v_vals2 = data[1]
act_vals2 = data[2]

ew = 1  # edge width
act_vals_diff = np.subtract(act_vals2, act_vals1)
act_vals_diff_non_edge = np.subtract(act_vals2[ew:-ew, ew:-ew], act_vals1[ew:-ew, ew:-ew])

act_vals_diff_edge = np.subtract(act_vals2, act_vals1)
act_vals_reldiff = np.divide(act_vals_diff, act_vals2)
act_vals_reldiff_non_edge = np.divide(act_vals_diff_non_edge, act_vals2[ew:-ew, ew:-ew])

# scaled activation values by the largest
act_vals_max1 = np.max(np.abs(act_vals1[:]))
act_vals_max2 = np.max(np.abs(act_vals2[:]))
act_vals_diff_scaled = np.subtract(np.abs(act_vals2/act_vals_max2), np.abs(act_vals1/act_vals_max1))
actvals_diff_reldiff = act_vals_diff_scaled/(act_vals1/act_vals_max1)
print('max of act_vals_diff: ', np.nanmax(np.abs((act_vals_diff[:]))))
print('max of act_vals_reldiff: ', np.nanmax(np.abs((act_vals_reldiff[:]))))
print('max of act_vals_diff_non_edge: ', np.nanmax(np.abs(act_vals_diff_non_edge[:])))
print('max of act_vals_reldiff_non_edge: ', np.nanmax(np.abs((act_vals_reldiff_non_edge[:]))))
print('max of act_vals_diff_scaled: ', np.nanmax(np.abs((act_vals_diff_scaled[:]))))
print('max of act_vals_diff_reldiff_scaled: ', np.nanmax(np.abs(actvals_diff_reldiff[:])))

