#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
import random
import math
import sys
if sys.version_info < (3,):
    range = xrange


class neuron():

    '''
    Класс нейрона с произвольным количеством входов, активируемого
    сигмоидальной функцией
    '''
#   Количество входов
    countInput = 0
#   Скорость обучения
    speedTeach = 0.2
#   Значения синапсов
    weights = []

    def __init__(self, count):
        self.countInput = count
        self.weights = [None] * count
        for i in range(0, len(self.weights)):
            self.weights[i] = random.uniform(-1, 1)

#   Сумма произведений входов на веса синапсов
    def __getSum(self, inputs):
        summ = 0
        for i in range(0, len(self.weights)):
            summ += self.weights[i] * inputs[i]
        return summ

#   Активация синапса по сигмоиду
    def activate(self, inputs):
        try:
            return 1 / (1 + math.exp(-self.__getSum(inputs)))
        except:
            return self.__getSum(inputs)

#   Мутация
    def mutate(self):
        for i in range(0, len(self.weights)):
            self.weights[i] += random.uniform(-0.01, 0.01)

#   Кроссовер
    def mergeLeft(self, other):
        newNeuron = neuron(1)
        newNeuron.weights = [None] * len(self.weights)
        for i in range(0,len(self.weights),2):
            newNeuron.weights[i] = self.weights[i]
        for i in range(1,len(self.weights),2):
            newNeuron.weights[i] = other.weights[i]
        return newNeuron

    def mergeRight(self, other):
        newNeuron = neuron(1)
        newNeuron.weights = [None] * len(self.weights)
        for i in range(1,len(self.weights),2):
            newNeuron.weights[i] = self.weights[i]
        for i in range(0,len(self.weights),2):
            newNeuron.weights[i] = other.weights[i]
        return newNeuron