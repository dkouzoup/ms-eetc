import sys
sys.path.append('..')

import json
import pandas as pd
import matplotlib.pyplot as plt

from train import Train
from track import Track
from ocp import casadiSolver

from utils import latexify

vMin = 1

train = Train(train='Intercity')

etaMax = 0.73
train.powerLosses = lambda f,v: f*v*(f>0)*(1 - etaMax)/etaMax - (1-etaMax)*f*v*(f<0)

train.forceMinPn = 0

track = Track(config={'id':'00_var_speed_limit_100'}, pathJSON='../tracks')
tripTime = 1541

with open('config.json') as file:

    solverOpts = json.load(file)

solverOpts['minimumVelocity'] = vMin

nRuns = 5

solverOpts['numIntervals'] = 300
solver = casadiSolver(train, track, solverOpts)

cpuTimes = []

for _ in range(nRuns):

    df, stats = solver.solve(tripTime, terminalVelocity=vMin, initialVelocity=vMin)
    cpuTimes.append(stats['CPU time [s]'])

energy_dms = round(df['Energy [kWh]'].sum(), 2)
time_dms = round(min(cpuTimes),2)

# load GPOPS-II results

df_gpops1 = pd.read_csv('../gpops/'+ track.title + '_GPOPSI.csv')
df_gpops1.set_index('Time [s]', inplace=True)
df_gpops1.drop_duplicates(subset='Position [m]', inplace=True)
df_gpops2 = pd.read_csv('../gpops/'+ track.title + '_GPOPSII.csv')
df_gpops2.set_index('Time [s]', inplace=True)
df_gpops2.drop_duplicates(subset='Position [m]', inplace=True)

energy_gpops1 = round(df_gpops1['Energy [kWh]'].iloc[0], 2)
time_gpops1 = round(df_gpops1['CPU Time [s]'].iloc[0], 2)

energy_gpops2 = round(df_gpops2['Energy [kWh]'].iloc[0], 2)
time_gpops2 = round(df_gpops2['CPU Time [s]'].iloc[0], 2)

# plot

latexify()

fig, ax = plt.subplots(2, 1)

styles = ['dashdot', 'solid', (5, (10, 3))]
colors = ['tab:green', 'tab:blue', 'tab:red']

errors = []
errorsPercent = []

error = round(energy_dms - energy_gpops2, 2)
errorPercent = round(100*error/energy_gpops2, 2)

ax[0].plot(df['Position [m]']*1e-3, df['Velocity [m/s]']*3.6, color=colors[0], linestyle=styles[0], label='DMS N={} ({} s)'.format(solverOpts['numIntervals'], time_dms))
ax[1].step(df['Position [m]']*1e-3, df['Force (el) [N]']*1e-3, color=colors[0], linestyle=styles[0], where='post')

ax[0].plot(df_gpops1['Position [m]']*1e-3, df_gpops1['Velocity [m/s]']*3.6, linestyle=styles[-2], color=colors[-2], label='GPOPS-II p N={} ({} s)'.format(len(df_gpops1)-1, time_gpops1))
ax[1].plot(df_gpops1['Position [m]']*1e-3, (df_gpops1['Force (acc) [N]']+df_gpops1['Force (rgb) [N]'])*1e-3, linestyle=styles[-2], color=colors[-2])

ax[0].plot(df_gpops2['Position [m]']*1e-3, df_gpops2['Velocity [m/s]']*3.6, linestyle=styles[-1], color=colors[-1], label='GPOPS-II hp-adaptive N={} ({} s)'.format(len(df_gpops2)-1, time_gpops2))
ax[1].plot(df_gpops2['Position [m]']*1e-3, (df_gpops2['Force (acc) [N]']+df_gpops2['Force (rgb) [N]'])*1e-3, linestyle=styles[-1], color=colors[-1])

ax[0].step(df['Position [m]']*1e-3, df['Speed limit [m/s]']*3.6, '-', color='tab:purple', label='Speed limit', where='post')

ax[0].grid(visible=True)
ax[1].grid(visible=True)

ax[0].set_xlabel('Position [km]')
ax[0].set_ylabel('Velocity [km/h]')

ax[1].set_xlabel('Position [km]')
ax[1].set_ylabel('Force [kN]')

ax[0].set_xlim([0, df['Position [m]'].iloc[-1]*1e-3])
ax[1].set_xlim([0, df['Position [m]'].iloc[-1]*1e-3])

ax[0].legend(loc='lower center', ncol=2)

fig.tight_layout()

fig.set_size_inches(8, 5)

plt.savefig('figure10.pdf', bbox_inches='tight')

plt.show()

print("Energy GPOPS-I:  {}".format(energy_gpops1))
print("Energy GPOPS-II: {}".format(energy_gpops2))
print("Energy DMS:      {}".format(energy_dms))
