import matplotlib.pyplot as plt
import numpy as np
import pandas

from .base import TimeSeries


class CO2(TimeSeries):
    def __init__(self, filepath='data/co2.csv', feature_range=(0.05, 0.95)):
        TimeSeries.__init__(self, feature_range=feature_range)
        dataframe = pandas.read_csv(filepath, usecols=[4], engine='python', skiprows=0)
        dataset = dataframe.values.reshape((np.prod(dataframe.shape), 1))
        self.dataset = dataset.astype('float64')

    def data(self):
        return self.dataset

    def window(self):
        plt.xlim(self.dataset.size * 0.6, self.dataset.size * 0.8)
        plt.ylim(350, 390)

    def __str__(self):
        return "co2"
