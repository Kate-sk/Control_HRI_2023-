import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

T1_files_task2 = [['T1_S1_tr1_ta2.csv', 'T1_S1_tr2_ta2.csv', 'T1_S1_tr3_ta2.csv', 'T1_S1_tr4_ta2.csv', 'T1_S1_tr5_ta2.csv', 'T1_S1_tr6_ta2.csv', 'T1_S1_tr7_ta2.csv', 'T1_S1_tr8_ta2.csv', 'T1_S1_tr9_ta2.csv', 'T1_S1_tr10_ta2.csv'],
                  ['T1_S2_tr1_ta2.csv', 'T1_S2_tr2_ta2.csv', 'T1_S2_tr3_ta2.csv', 'T1_S2_tr4_ta2.csv', 'T1_S2_tr5_ta2.csv', 'T1_S2_tr6_ta2.csv', 'T1_S2_tr7_ta2.csv', 'T1_S2_tr8_ta2.csv', 'T1_S2_tr9_ta2.csv', 'T1_S2_tr10_ta2.csv']]

T2_files_task2 = [['T2_S1_tr1_ta2.csv', 'T2_S1_tr2_ta2.csv', 'T2_S1_tr3_ta2.csv', 'T2_S1_tr4_ta2.csv', 'T2_S1_tr5_ta2.csv', 'T2_S1_tr6_ta2.csv', 'T2_S1_tr7_ta2.csv', 'T2_S1_tr8_ta2.csv', 'T2_S1_tr9_ta2.csv', 'T2_S1_tr10_ta2.csv'],
                  ['T2_S2_tr1_ta2.csv', 'T2_S2_tr2_ta2.csv', 'T2_S2_tr3_ta2.csv', 'T2_S2_tr4_ta2.csv', 'T2_S2_tr5_ta2.csv', 'T2_S2_tr6_ta2.csv', 'T2_S2_tr7_ta2.csv', 'T2_S2_tr8_ta2.csv', 'T2_S2_tr9_ta2.csv', 'T2_S2_tr10_ta2.csv']]


# data_T1[a, (b, c), d]:
    # a = test subject; b = trial number; c = mean, std, rms, time; d = task
data_T1_task2 = np.zeros((len(T1_files_task2), len(T1_files_task2[0]), 4, 2))
data_T2_task2 = np.zeros((len(T2_files_task2), len(T2_files_task2[0]), 4, 2))


for k in range(len(T1_files_task2)):
    for l in range(len(T1_files_task2[k])):
        data_T1_task2[k,l] = pd.read_csv(T1_files_task2[k][l], sep=';', decimal=",", usecols = [1,2], nrows = 4)

for q in range(len(T2_files_task2)):
    for r in range(len(T2_files_task2[q])):
        data_T2_task2[q,r] = pd.read_csv(T2_files_task2[q][r], sep=';', decimal=",", usecols = [1,2], nrows = 4)

# Cutting task - time
fig1, ax1 = plt.subplots()
ax1.plot(range(10), data_T1_task2[0,:,3,0], 'go')
ax1.plot(range(10), data_T2_task2[0,:,3,0], 'ro')
ax1.plot(range(10), data_T1_task2[1,:,3,0], 'go')
ax1.plot(range(10), data_T2_task2[1,:,3,0], 'ro')
ax1.legend(['Group T1', 'Group T2'])
ax1.set(xlabel='Trial number', ylabel='Time (s)', title='Time spent on cutting, second time (perturbation)')

# Cutting task - mean performance
fig2, ax2 = plt.subplots()
ax2.plot(range(10), data_T1_task2[0,:,0,0], 'go')
ax2.plot(range(10), data_T2_task2[0,:,0,0], 'ro')
ax2.plot(range(10), data_T1_task2[1,:,0,0], 'go')
ax2.plot(range(10), data_T2_task2[1,:,0,0], 'ro')
ax2.legend(['Group T1', 'Group T2'])
ax2.set(xlabel='Trial number', ylabel='Performance', title='Mean performance in cutting, second time (perturbation)')

# Stitching task - time
fig3, ax3 = plt.subplots()
ax3.plot(range(10), data_T1_task2[0,:,3,1], 'go')
ax3.plot(range(10), data_T2_task2[0,:,3,1], 'ro')
ax3.plot(range(10), data_T1_task2[1,:,3,1], 'go')
ax3.plot(range(10), data_T2_task2[1,:,3,1], 'ro')
ax3.legend(['Group T1', 'Group T2'])
ax3.set(xlabel='Trial number', ylabel='Time (s)', title='Time spent on stitching, second time (perturbation)')

# Stitching task - mean performance
fig4, ax4 = plt.subplots()
ax4.plot(range(10), data_T1_task2[0,:,0,1], 'go')
ax4.plot(range(10), data_T2_task2[0,:,0,1], 'ro')
ax4.plot(range(10), data_T1_task2[1,:,0,1], 'go')
ax4.plot(range(10), data_T2_task2[1,:,0,1], 'ro')
ax4.legend(['Group T1', 'Group T2'])
ax4.set(xlabel='Trial number', ylabel='Performance', title='Mean performance in stitching, second time (perturbation)')


