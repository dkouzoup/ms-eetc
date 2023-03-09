
from figure6 import runSimulation, plotThreeTrajectories

if __name__ == '__main__':

    df0, df0b, df1, df1b, df2, df2b, _, _ = runSimulation(trackID='Swiss')

    actualLosses0 =  df0b['Losses [kWh]'].sum()
    actualLosses1 = df1b['Losses [kWh]'].sum()
    actualLosses2 = df2b['Losses [kWh]'].sum()

    actualEnergy0 =  df0b['Energy [kWh]'].sum()
    actualEnergy1 = df1b['Energy [kWh]'].sum()
    actualEnergy2 = df2b['Energy [kWh]'].sum()

    plotThreeTrajectories(df0, df1, df2, figSize=[8, 5], filename='figure8.pdf', withSpeedLimits=True, withAltitude=True, \
        losses0=actualLosses0, losses1=actualLosses1, losses2=actualLosses2, \
        energy0=actualEnergy0, energy1=actualEnergy1, energy2=actualEnergy2)
