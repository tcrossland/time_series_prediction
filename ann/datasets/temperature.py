import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

from .base import TimeSeries


class Temperature(TimeSeries):
    def __init__(self, filepath='data/station.csv'):
        TimeSeries.__init__(self)
        dataframe = pd.read_csv(filepath, usecols=range(1, 13), engine='python', skiprows=0)
        imp = Imputer(missing_values=999.90)
        imp.fit(dataframe)
        dataset = imp.transform(dataframe)
        dataset = dataset.reshape((np.prod(dataset.shape), 1))
        self.dataset = dataset.astype('float64')

    def data(self):
        return self.dataset

    def window(self):
        plt.xlim(self.dataset.size * 0.6, self.dataset.size * 0.8)
        plt.ylim(0, 30)

    def __str__(self):
        return "temp"
