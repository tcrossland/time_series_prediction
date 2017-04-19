import matplotlib.pyplot as plt
import numpy as np
import pandas

from .base import TimeSeries


class Bbva(TimeSeries):
    def __init__(self, filepath='data/bbva.csv', feature_range=(0.05, 0.95)):
        TimeSeries.__init__(self, feature_range=feature_range)
        dataframe = pandas.read_csv(filepath, usecols=[3], engine='python')
        dataframe = dataframe.iloc[::-1]
        dataset = dataframe.values.reshape((np.prod(dataframe.shape), 1))
        self.dataset = dataset.astype('float64')

    def data(self):
        return self.dataset

    def window(self):
        plt.xlim(self.dataset.size * 0.6, self.dataset.size * 0.8)
        plt.ylim(5, 11)

    def __str__(self):
        return "bbva"
