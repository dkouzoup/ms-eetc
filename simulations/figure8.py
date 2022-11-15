
from figure6 import runSimulation, plotThreeTrajectories

if __name__ == '__main__':

    df0, df0b, df1, df1b, df2, df2b, _, _ = runSimulation(trackID='Swiss')

    actualLosses0 =  df0b['Losses [kWh]'].sum()
    actualLosses1 = df1b['Losses [kWh]'].sum()
    actualLosses2 = df2b['Losses [kWh]'].sum()

    plotThreeTrajectories(df0, df1, df2, figSize=[8, 5], filename='figure8.pdf', \
        withSpeedLimits=True, withAltitude=True, \
        losses0=round(actualLosses0,1), losses1=round(actualLosses1,1), losses2=round(actualLosses2,1))
