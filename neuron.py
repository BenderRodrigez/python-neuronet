#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
import random
import math


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
        self.weights = [random.uniform(-2.5, 2.5)] * count

#   Сумма произведений входов на веса синапсов
    def __getSum(self, inputs):
        summ = 0
        for i in range(0, len(self.weights)):
            summ += self.weights[i] * inputs[i]
        return summ

#   Активация синапса по сигмоиду
    def activate(self, inputs):
        return 1 / (1 + math.exp(-self.__getSum(inputs)))
