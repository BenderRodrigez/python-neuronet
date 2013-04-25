#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from neuron import neuron
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
    minDeltaError = 0.0000
#    Максимальное количество эпох
    maxEpoch = 1000000
#	 Нейроны первого слоя
    __layer1 = []
#	 Нейроны второго слоя
    __layer2 = []
#	 Входные данные с образцами обучения
    __teachCollection = []
#	 Входные данные со значениями
    __inputCollection = []
#	 Маштабированные данные
    __scaleCollection = []
#	 Де-масштабированные данные
    __reverseScaleCollection = []

    def __init__(self, a, b, countTeachElement,
                 countNeurons, sumError, functionText="math.sin(x)"):
        self.countTeachElement = countTeachElement
        self.countNeurons = countNeurons
        self.maxError = sumError
        for i in range(0, countNeurons):
            self.__layer1.append(neuron(1))
        self.__layer2.append(neuron(countNeurons))

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
        errors = []
        e = 0
        sumErrorEp = 0
        while True:
            e += 1
            sumErrorEp = 0
            y[:] = []
            for k in range(0, self.countTeachElement):
                y1 = []
                for n in self.__layer1:
                    y1.append(n.activate([self.__inputCollection[k]]))
                y.append(self.__layer2[0].activate(y1))

                self.backPropagate(k, y1, y[k])

                sumErrorEp += 0.5 * pow(self.__scaleCollection[k] - y[k], 2)

            print(sumErrorEp)
            errors.append(sumErrorEp)

            if sumErrorEp <= self.maxError:
                return [True, "Обучена"]
            elif e >= self.maxEpoch:
                return [False, "Не обучена, \
                Достугнуто максимальное количество эпох"]
#            elif e > 1 and errors[e-2] - errors[e-1] <= self.minDeltaError:
#                return ResultTeach(False, "Не обучена, Эффективность\
#                                           обучения ниже необходимой")

#   Обратное распространение ошибки
    def backPropagate(self, k, out1, out2):
        delta = out2 * (1 - out2) * (self.__scaleCollection[k] - out2)
        for i in range(0, self.__layer2[0].countInput):
            self.__layer2[0].weights[i] += self.__layer2[0].speedTeach *\
                delta * out1[i]

        for j in range(0, len(self.__layer1)):
            for i in range(0, self.__layer1[j].countInput):
                self.__layer1[j].weights[i] += self.__layer1[j].speedTeach *\
                    delta * self.__layer2[0].weights[j] *\
                    out1[j] * (1 - out1[j]) * \
                    self.__inputCollection[k]

#   Получение аппроксимированного результата для значения
    def getValue(self, x):
        y1 = []
        for n in self.__layer1:
            y1.append(n.activate([x]))
        tmp = self.__layer2[0].activate(y1)
        tmp = self.__buildRevScaleCollection(tmp)
        return tmp

#   Получение текста функции
    def getFunctionText(self):
        return self.__functionText
