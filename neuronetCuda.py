#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import pycuda.gpuarray as gpuarray
import math


class neuronetCuda():

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
#    Входные данные с образцами обучения
    __teachCollection = []
#    Входные данные со значениями
    __inputCollection = []
#    Маштабированные данные
    __scaleCollection = []
#    Де-масштабированные данные
    __reverseScaleCollection = []

    def __init__(self, a, b, countTeachElement,
                 countNeurons, sumError, functionText="math.sin(x)"):
        self.countTeachElement = countTeachElement
        self.countNeurons = countNeurons
        self.maxError = sumError

        sample = gpuarray.np.random.random(countNeurons)
        self.__weights = gpuarray.to_gpu(gpuarray.np.random.random(countNeurons))
        self.__values = gpuarray.to_gpu(gpuarray.np.empty_like(sample))
        self.__weights2 = gpuarray.to_gpu(gpuarray.np.random.random(countNeurons))
        self.__mid_mult = gpuarray.to_gpu(gpuarray.np.empty_like(sample))
        self.__value2 = 0

        self.__functionText = functionText
        self.__inputCollection = self.__buildInputCollection(a, b, countTeachElement)
        self.__teachCollection = self.__buildTeachCollection(self.__inputCollection)
        self.__scaleCollection = self.__buildScaleCollection(self.__teachCollection)

        self.__function = ElementwiseKernel(
            str("double *x, double y, double *z"),
            str("z[i] = 1 / (1 + exp(-(y * x[i])))"),
            str("linear_multiply"))

        self.__functionMultiply = ElementwiseKernel(
            str("double *x, double *y, double *z"),
            str("z[i] = x[i] * y[i]"),
            str("linear_multiply"))

        self.__backPropagateTop = ElementwiseKernel(
            str("double *x, double *y, double z"),
            str("x[i] += 0.5 * z * y[i]"),
            str("linear_backPropagateTop"))

        self.__backPropagateBot = ElementwiseKernel(
            str("double *x, double *y, double *z, double a, double b"),
            str("x[i] += 0.5 * a * y[i] * z[i] * (1 - z[i]) * b"),
            str("linear_backPropagateBot"))

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
                self.evaluateInput(self.__inputCollection[k])

                self.backPropagate(k)
                sumErrorEp += 0.5 * pow(self.__scaleCollection[k] - self.__value2, 2)

            print(sumErrorEp)
            errors.append(sumErrorEp)

            if sumErrorEp <= self.maxError:
                return [True, "Обучена"]
            elif e >= self.maxEpoch:
                return [False, "Не обучена, \
                Достугнуто максимальное количество эпох"]
            # elif e > 1 and errors[e-2] < errors[e-1]:
            #     return [False, "Не обучена, Эффективность\
            #                               обучения ниже необходимой"]

#   Обратное распространение ошибки
    def backPropagate(self, k):
        delta = self.__value2 * (1 - self.__value2) * (self.__scaleCollection[k] - self.__value2)
        self.__backPropagateTop(self.__weights2, self.__values, delta)
        self.__backPropagateBot(self.__weights, self.__weights2, self.__values, delta, self.__inputCollection[k])

#   Получение аппроксимированного результата для значения
    def getValue(self, x):
        self.evaluateInput(x)
        return self.__buildRevScaleCollection(self.__value2)

#   Получение текста функции
    def getFunctionText(self):
        return self.__functionText

    def evaluateInput(self, currentInput):
        self.__function(self.__weights, currentInput, self.__values)
        self.__functionMultiply(self.__values, self.__weights2, self.__mid_mult)

        bareValue = self.__getArraySum(self.__mid_mult)
        try:
            self.__value2 = 1 / (1 + math.exp(-bareValue))
        except:
            self.__value2 = bareValue

    def __getArraySum(self, arr):
        return pycuda.gpuarray.sum(arr).get()
