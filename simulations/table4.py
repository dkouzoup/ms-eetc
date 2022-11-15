
from figure6 import runSimulation


if __name__ == '__main__':

    df0, df0b, df1, df1b, df2, df2b, stats, _ = runSimulation(trackID='Swiss', nRuns=5)

    expectedLosses0 = df0['Losses [kWh]'].sum()
    expectedEnergy0 = df0['Energy [kWh]'].sum()

    actualLosses0 =  df0b['Losses [kWh]'].sum()
    actualEnergy0 =  df0b['Energy [kWh]'].sum()

    expectedLosses1 = df1['Losses [kWh]'].sum()
    expectedEnergy1 = df1['Energy [kWh]'].sum()

    actualLosses1 = df1b['Losses [kWh]'].sum()
    actualEnergy1 = df1b['Energy [kWh]'].sum()

    expectedLosses2 = df2['Losses [kWh]'].sum()
    expectedEnergy2 = df2['Energy [kWh]'].sum()

    actualLosses2 = df2b['Losses [kWh]'].sum()
    actualEnergy2 = df2b['Energy [kWh]'].sum()

    r = lambda x: round(x,1)
    i = lambda new, old: round(100*abs(new-old)/old,1)

    print("")
    print("Modelling of efficiency:   \tPerfect\tStatic\tDynamic")
    print("Expected losses [kWh]:     \t{}\t{}\t{} ".format(r(expectedLosses0), r(expectedLosses1), r(expectedLosses2)))
    print("Actual losses [kWh]:       \t{}\t{}\t{} ".format(r(actualLosses0), r(actualLosses1), r(actualLosses2)))
    print("Improvement over perfect:  \t-\t{}%\t{}%".format(i(actualLosses1, actualLosses0), i(actualLosses2, actualLosses0)))
    print("Improvement over static:   \t-\t-\t{}%".format(i(actualLosses2, actualLosses1)))

    print("Expected consumption [kWh]:\t{}\t{}\t{} ".format(r(expectedEnergy0), r(expectedEnergy1), r(expectedEnergy2)))
    print("Actual consumption [kWh]:  \t{}\t{}\t{} ".format(r(actualEnergy0), r(actualEnergy1), r(actualEnergy2)))
    print("Improvement over perfect:  \t-\t{}%\t{}%".format(i(actualEnergy1, actualEnergy0), i(actualEnergy2, actualEnergy0)))
    print("Improvement over static:   \t-\t-\t{}%".format(i(actualEnergy2, actualEnergy1)))

    print("")

    print("Cpu time:                  \t{}\t{}\t{}".format(round(stats['cpuTime0'], 1) , round(stats['cpuTime1'], 1), round(stats['cpuTime2'], 1)))
    print("Number of iterations:      \t{}\t{}\t{}".format(stats['numIter0'], stats['numIter1'], stats['numIter2']))
    print("Cpu time per iteration:    \t{}\t{}\t{}".format(round(stats['cpuTime0']/stats['numIter0'], 2) , round(stats['cpuTime1']/stats['numIter1'], 2), round(stats['cpuTime2']/stats['numIter2'], 2)))
