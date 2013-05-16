#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from neuronet import neuronet
from neuronetCuda import neuronetCuda
import matplotlib.pyplot as plt
import pylab
import datetime
import json

def measureNeuronetTraining(precision, neuronsCount, net):
    n = net(0.1, 4, 100, neuronsCount, precision, functionText="3 * x + math.log(x)")
    t1 = datetime.datetime.now()
    n.teach()
    t2 = datetime.datetime.now()
    elapsed = t2 - t1
    return elapsed.seconds + float("0.%s"%elapsed.microseconds)

if __name__ == '__main__':
    # precision = 0.001
    # count = 10
    # n = neuronet(0.1, 4, 10, count, precision, functionText="3 * x + math.log(x)", debug=True)
    # nC = neuronetCuda(0.1, 4, 100, count, precision, functionText="3 * x + math.log(x)", debug=True)

    # nC.teach()
    # x = []
    # y = []
    # y1 = []
    # a = 0.1
    # for i in range(1, 40):
    #     x.append(a)
    #     y.append(nC.function(a))
    #     y1.append(nC.getValue(a))
    #     a += 0.1
    # line1 = plt.plot(x, y, color="green")
    # line2 = plt.plot(x, y1, color="red")
    # plt.ylabel(n.getFunctionText())
    # pylab.show()

    resultsTotal = {}
    resultsM = 0
    res = []
    for i in range(50, 100):
        print("CPU, wave %s"%i)
        time = measureNeuronetTraining(0.0001, i*10, neuronet)
        resultsM += time
        res.append(time)
    resultsTotal["CPU average"] = resultsM / 20
    resultsTotal["CPU results"] = res


    resultsM = 0
    res = []
    for i in range(50, 100):
        print("GPU, wave %s"%i)
        time = measureNeuronetTraining(0.0001, i*10, neuronetCuda)
        resultsM += time
        res.append(time)
    resultsTotal["GPU average"] = resultsM / 20
    resultsTotal["GPU results"] = res

    print(resultsTotal)