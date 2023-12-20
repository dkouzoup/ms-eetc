
from figure6 import runSimulation


if __name__ == '__main__':

    df0_rg, df0b_rg, df1_rg, df1b_rg, df2_rg, df2b_rg, stats_rg, _ = runSimulation(trackID='CH_StGallen_Wil', nRuns=5, brakeType='rg')
    df0_pn, df0b_pn, df1_pn, df1b_pn, df2_pn, df2b_pn, stats_pn, _ = runSimulation(trackID='CH_StGallen_Wil', nRuns=5, brakeType='pn')

    expectedLosses0_rg = df0_rg['Losses [kWh]'].sum()
    expectedEnergy0_rg = df0_rg['Energy [kWh]'].sum()

    actualLosses0_rg =  df0b_rg['Losses [kWh]'].sum()
    actualEnergy0_rg =  df0b_rg['Energy [kWh]'].sum()

    expectedLosses1_rg = df1_rg['Losses [kWh]'].sum()
    expectedEnergy1_rg = df1_rg['Energy [kWh]'].sum()

    actualLosses1_rg = df1b_rg['Losses [kWh]'].sum()
    actualEnergy1_rg = df1b_rg['Energy [kWh]'].sum()

    expectedLosses2_rg = df2_rg['Losses [kWh]'].sum()
    expectedEnergy2_rg = df2_rg['Energy [kWh]'].sum()

    actualLosses2_rg = df2b_rg['Losses [kWh]'].sum()
    actualEnergy2_rg = df2b_rg['Energy [kWh]'].sum()

    expectedLosses0_pn = df0_pn['Losses [kWh]'].sum()
    expectedEnergy0_pn = df0_pn['Energy [kWh]'].sum()

    actualLosses0_pn =  df0b_pn['Losses [kWh]'].sum()
    actualEnergy0_pn =  df0b_pn['Energy [kWh]'].sum()

    expectedLosses1_pn = df1_pn['Losses [kWh]'].sum()
    expectedEnergy1_pn = df1_pn['Energy [kWh]'].sum()

    actualLosses1_pn = df1b_pn['Losses [kWh]'].sum()
    actualEnergy1_pn = df1b_pn['Energy [kWh]'].sum()

    expectedLosses2_pn = df2_pn['Losses [kWh]'].sum()
    expectedEnergy2_pn = df2_pn['Energy [kWh]'].sum()

    actualLosses2_pn = df2b_pn['Losses [kWh]'].sum()
    actualEnergy2_pn = df2b_pn['Energy [kWh]'].sum()

    r = lambda x: round(x,1)
    i = lambda new, old: round(100*abs(new-old)/old,1)

    print("")
    print("Modelling of efficiency:   \tPerfect\tStatic\tDynamic")
    print("Expected losses [kWh]:     \t{}/{}\t{}/{}\t{}/{} ".format(r(expectedLosses0_rg), r(expectedLosses0_pn), r(expectedLosses1_rg), r(expectedLosses1_pn), r(expectedLosses2_rg),  r(expectedLosses2_pn)))
    print("Actual losses [kWh]:       \t{}/{}\t{}/{}\t{}/{} ".format(r(actualLosses0_rg), r(actualLosses0_pn), r(actualLosses1_rg), r(actualLosses1_pn), r(actualLosses2_rg), r(actualLosses2_pn)))
    print("Improvement over perfect:  \t-\t{}/{}%\t{}/{}%".format(i(actualLosses1_rg, actualLosses0_rg), i(actualLosses1_pn, actualLosses0_pn), i(actualLosses2_rg, actualLosses0_rg), i(actualLosses2_pn, actualLosses0_pn)))
    print("Improvement over static:   \t-\t-\t{}/{}%".format(i(actualLosses2_rg, actualLosses1_rg), i(actualLosses2_pn, actualLosses1_pn)))

    print("Expected consumption [kWh]:\t{}/{}\t{}/{}\t{}/{} ".format(r(expectedEnergy0_rg), r(expectedEnergy0_pn), r(expectedEnergy1_rg), r(expectedEnergy1_pn), r(expectedEnergy2_rg), r(expectedEnergy2_pn)))
    print("Actual consumption [kWh]:  \t{}/{}\t{}/{}\t{}/{} ".format(r(actualEnergy0_rg), r(actualEnergy0_pn), r(actualEnergy1_rg), r(actualEnergy1_pn), r(actualEnergy2_rg), r(actualEnergy2_pn)))
    print("Improvement over perfect:  \t-\t{}/{}%\t{}/{}%".format(i(actualEnergy1_rg, actualEnergy0_rg), i(actualEnergy1_pn, actualEnergy0_pn), i(actualEnergy2_rg, actualEnergy0_rg), i(actualEnergy2_pn, actualEnergy0_pn)))
    print("Improvement over static:   \t-\t-\t{}/{}%".format(i(actualEnergy2_rg, actualEnergy1_rg), i(actualEnergy2_pn, actualEnergy1_pn)))

    print("")

    print("Cpu time:                  \t{}/{}\t{}/{}\t{}/{}".format(round(stats_rg['cpuTime0'], 1), round(stats_pn['cpuTime0'], 1), round(stats_rg['cpuTime1'], 1), round(stats_pn['cpuTime1'], 1), round(stats_rg['cpuTime2'], 1), round(stats_pn['cpuTime2'], 1)))
    print("Number of iterations:      \t{}/{}\t{}/{}\t{}/{}".format(stats_rg['numIter0'], stats_pn['numIter0'], stats_rg['numIter1'], stats_pn['numIter1'], stats_rg['numIter2'], stats_pn['numIter2']))
    print("Cpu time per iteration:    \t{}/{}\t{}/{}\t{}/{}".format(round(stats_rg['cpuTime0']/stats_rg['numIter0'], 2), round(stats_pn['cpuTime0']/stats_pn['numIter0'], 2), round(stats_rg['cpuTime1']/stats_rg['numIter1'], 2), round(stats_pn['cpuTime1']/stats_pn['numIter1'], 2), round(stats_rg['cpuTime2']/stats_rg['numIter2'], 2), round(stats_pn['cpuTime2']/stats_pn['numIter2'], 2)))
