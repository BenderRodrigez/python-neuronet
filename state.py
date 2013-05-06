#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from neuron import neuron
import math


class State(object):
    #    Нейроны первого слоя
    layer1 = []
    #    Нейроны второго слоя
    layer2 = []

    #   Суммарная ошибка этого состояния
    error = 100

    """docstring for State"""
    def __init__(self):
        pass

    def getValue(self, x):
        y1 = []
        for n in self.layer1:
            y1.append(n.activate([x]))
        tmp = self.layer2[0].activate(y1)
        return tmp

    def buildError(self, inputs, values):
        self.error = 0
        for i in range(0, len(inputs)):
            self.error += 0.5 * pow(values[i] - self.getValue(inputs[i]), 2)

    def mutate(self):
        for i in self.layer1:
            i.mutate()
        for i in self.layer2:
            i.mutate()

    def merge(self, other):
        newState = [None] * 2
        newState[0] = State()
        newState[1] = State()
        newState[0].layer1 = [None] * len(self.layer1)
        newState[0].layer2 = [None] * len(self.layer2)
        newState[1].layer1 = [None] * len(self.layer1)
        newState[1].layer2 = [None] * len(self.layer2)
        for i in range(0, len(newState[0].layer1)):
            newState[0].layer1[i] = self.layer1[i].mergeLeft(other.layer1[i])
            newState[1].layer1[i] = self.layer1[i].mergeRight(other.layer1[i])
        for i in range(0, len(newState[0].layer2)):
            newState[0].layer2[i] = self.layer2[i].mergeLeft(other.layer2[i])
            newState[1].layer2[i] = self.layer2[i].mergeRight(other.layer2[i])
        return newState

    def __cmp__(self, other):
        if self.error < other.error:
            return -1  
        elif self.error > other.error:
            return 1
        else: 
            return 0 