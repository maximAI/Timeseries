Задача соостоит в том, чтобы посмотреть, получится или нет, добавив дополнительные фичи/признаки избавиться от автокрреляции.

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
