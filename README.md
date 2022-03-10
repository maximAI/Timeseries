# Фич инжиниринг
Задача состоит в том, чтобы посмотреть, получится или нет, добавив дополнительные фичи/признаки, избавиться от автокрреляции.
Импортируем нужные библиотеки.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv1D, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import gdown
```
Определим фукции для работы.
```
def getPred(currModel, xVal, yVal, yScaler):
    '''
    Функция рассчитываем результаты прогнозирования сети
    Args:
        В аргументы принимает сеть (currModel) и проверочную выборку
    Return:
        Выдаёт результаты предсказания predVal
        И правильные ответы в исходной размерности yValUnscaled (какими они были до нормирования)
    '''
    # Предсказываем ответ сети по проверочной выборке
    # И возвращаем исходны масштаб данных, до нормализации
    predVal = yScaler.inverse_transform(currModel.predict(xVal))
    yValUnscaled = yScaler.inverse_transform(yVal)
    
    return predVal, yValUnscaled
```
```
def showPredict(start, step, channel, predVal, yValUnscaled):
    '''
    Функция визуализирует графики, что предсказала сеть и какие были правильные ответы
    Args:
        start - точка с которой начинаем отрисовку графика
        step - длина графика, которую отрисовываем
        channel - какой канал отрисовываем
    '''
    plt.plot(predVal[start:start + step, 0],
            label = 'Прогноз')
    plt.plot(yValUnscaled[start:start + step, channel], 
            label = 'Базовый ряд')
    plt.xlabel('Время')
    plt.ylabel('Значение Close')
    plt.legend()
    plt.show()
```
```
def correlate(a, b):
    '''
    Функция расёта корреляции дух одномерных векторов
    '''
    # Рассчитываем основные показатели
    ma = a.mean() # Среднее значение первого вектора
    mb = b.mean() # Среднее значение второго вектора
    mab = (a * b).mean() # Среднее значение произведения векторов
    sa = a.std() # Среднеквадратичное отклонение первого вектора
    sb = b.std() # Среднеквадратичное отклонение второго вектора
    
    # Рассчитываем корреляцию
    val = 1
    if ((sa > 0) & (sb > 0)):
        val = (mab - ma * mb)/(sa * sb)
    return val
```
```
def showCorr(channels, corrSteps, predVal, yValUnscaled):
    '''
    Функция рисуем корреляцию прогнозированного сигнала с правильным,
    cмещая на различное количество шагов назад, для проверки появления эффекта автокорреляции
    Args:
        channels - по каким каналам отображать корреляцию
        corrSteps - на какое количество шагов смещать сигнал назад для рассчёта корреляции
    '''
    # Проходим по всем каналам
    for ch in channels:
        corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно
        yLen = yValUnscaled.shape[0] # Запоминаем размер проверочной выборки

    # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
        # Получаем сигнал, смещённый на i шагов назад
        # predVal[i:, ch]
        # Сравниваем его с верными ответами, без смещения назад
        # yValUnscaled[:yLen-i,ch]
        # Рассчитываем их корреляцию и добавляем в лист
        corr.append(correlate(yValUnscaled[:yLen - i, ch], predVal[i:, 0]))

    own_corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно

    # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
        # Получаем сигнал, смещённый на i шагов назад
        # predVal[i:, ch]
        # Сравниваем его с верными ответами, без смещения назад
        # yValUnscaled[:yLen-i,ch]
        # Рассчитываем их корреляцию и добавляем в лист
        own_corr.append(correlate(yValUnscaled[:yLen - i, ch], yValUnscaled[i:, ch]))

    # Отображаем график коррелций для данного шага
    plt.plot(corr, label = 'Предсказание на ' + str(ch + 1) + ' шаг')
    plt.plot(own_corr, label = 'Эталон')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()
```
[Ноутбук колаба](https://colab.research.google.com/drive/1-YSwg3N_snmKxM7pGICV0ph6ffhhrSGq#scrollTo=i0mHUpXZk7Qj&uniqifier=2)
