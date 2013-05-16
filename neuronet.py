#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from neuron import neuron
from state import State
from random import randint
from copy import copy
import math


class neuronet():

    '''
    Класс нейронной сети, используемой для аппроксимирования непериодичной
    функции на интервале [a, b]
    '''
#    Количество нейронов первого слоя
    countNeurons = None
#    Количество элементов в обучающей выборке
    countTeachElement = None
#    Минимальная ошибка
    maxError = None
#    Минимальное изменение ошибки
    minDeltaError = 0.000001
#    Максимальное количество эпох
    maxEpoch = 1000000
#	 Входные данные с образцами обучения
    __teachCollection = []
#	 Входные данные со значениями
    __inputCollection = []
#	 Маштабированные данные
    __scaleCollection = []
#	 Де-масштабированные данные
    __reverseScaleCollection = []
#    Текущее состояние нейронов
    state = State()

    errors = []

    def __init__(self, a, b, countTeachElement,
                 countNeurons, sumError, functionText="math.sin(x)", debug=False):
        self.debug = debug
        self.countTeachElement = countTeachElement
        self.countNeurons = countNeurons
        self.maxError = sumError
        for i in range(0, countNeurons):
            self.state.layer1.append(neuron(1))
        self.state.layer2.append(neuron(countNeurons))
        self.__functionText = functionText
        self.__inputCollection = self.__buildInputCollection(a, b, countTeachElement)
        self.__teachCollection = self.__buildTeachCollection(self.__inputCollection)
        self.__scaleCollection = self.__buildScaleCollection(self.__teachCollection)

#   Аппроксимируемая функция
    def function(self, x):
        try:
            return eval(self.__functionText)
        except:
            return x

#   Построение набора эталонных результатов
    def __buildTeachCollection(self, x):
        result = []
        for element in x:
            result.append(self.function(element))
        return result

#   Построение набора входных данных
    def __buildInputCollection(self, a, b, count):
        result = []
        step = (b - a) / (count - 1)
        while round(a, 5) <= b:
            result.append(round(a, 5))
            a += step
        return result

#   Масштабирования данных
    def __buildScaleCollection(self, a):
        nRes = []
        for i in a:
            nRes.append((i + 11) / 100)
        return nRes

#   Де-масштабирование данных
    def __buildRevScaleCollection(self, a):
        return (a * 100) - 11

#   Обучение сети
    def teach(self):
        y = []
        self.errors = []
        e = 0
        sumErrorEp = 0
        while True:
            e += 1
            sumErrorEp = 0
            y[:] = []
            for k in range(0, self.countTeachElement):
                y1 = []
                for n in self.state.layer1:
                    y1.append(n.activate([self.__inputCollection[k]]))
                y.append(self.state.layer2[0].activate(y1))

                self.backPropagate(k, y1, y[k])

                sumErrorEp += 0.5 * pow(self.__scaleCollection[k] - y[k], 2)

            if self.debug:
                print(sumErrorEp)
            self.errors.append(sumErrorEp)

            if sumErrorEp <= self.maxError:
                return [True, "Обучена"]
            elif e >= self.maxEpoch:
                return [False, "Не обучена, \
                Достугнуто максимальное количество эпох"]
            elif e > 1 and self.errors[e-2] - self.errors[e-1] <= self.minDeltaError:
                return [False, "Не обучена, Эффективность обучения ниже необходимой: %s"%self.errors[e - 1]]

#   Генетический алгоритм
    def teachGenetic(self):
        s = self.state
        oldGen = []
        for i in range(0, 10):
            oldGen.append(copy(s))
        for i in range(0, 10):
            oldGen[i].mutate()

        newGen = []

        while True:
            for i in oldGen:
                c = copy(i)
                c.mutate()
                c.buildError(self.__inputCollection, self.__scaleCollection)
                newGen.append(c)

            for i in range(0, 10):
                c = oldGen[randint(0, len(oldGen) - 1)].merge(oldGen[randint(0, len(oldGen) - 1)])
                c[0].buildError(self.__inputCollection, self.__scaleCollection)
                c[1].buildError(self.__inputCollection, self.__scaleCollection)
                newGen += c

            newGen += oldGen
            newGen.sort()
            for i in range(0, 10):
                oldGen[i] = newGen[i]

            newGen = []
            print(oldGen[0].error)
            if oldGen[0].error <= self.maxError:
                self.state = oldGen[0]
                return [True, "Обучена"]

#   Обратное распространение ошибки
    def backPropagate(self, k, out1, out2):
        delta = out2 * (1 - out2) * (self.__scaleCollection[k] - out2)
        for i in range(0, self.state.layer2[0].countInput):
            self.state.layer2[0].weights[i] += self.state.layer2[0].speedTeach *\
                delta * out1[i]

        for j in range(0, len(self.state.layer1) - 1):
            for i in range(0, self.state.layer1[j].countInput - 1):
                self.state.layer1[j].weights[i] += self.state.layer1[j].speedTeach *\
                    delta * self.state.layer2[0].weights[j] *\
                    out1[j] * (1 - out1[j]) * \
                    self.__inputCollection[k]

#   Получение аппроксимированного результата для значения
    def getValue(self, x):
        return self.__buildRevScaleCollection(self.state.getValue(x))

#   Получение текста функции
    def getFunctionText(self):
        return self.__functionText
