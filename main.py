#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from neuronet import neuronet
import matplotlib.pyplot as plt
import pylab

if __name__ == '__main__':
    n = neuronet(0.1, 4, 100, 15, 0.003, functionText="3 * x + math.log(x)")
    n.teach()
    x = []
    y = []
    y1 = []
    a = 0.1
    for i in range(1, 40):
        x.append(a)
        y.append(n.function(a))
        y1.append(n.getValue(a))
        a += 0.1
    line1 = plt.plot(x, y, color="green")
    line2 = plt.plot(x, y1, color="red")
    plt.ylabel(n.getFunctionText())
    pylab.show()
