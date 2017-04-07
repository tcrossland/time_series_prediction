import pandas
import numpy as np
from .base import TimeSeries


class CO2(TimeSeries):
    def __init__(self, filepath='../data/co2.csv'):
        TimeSeries.__init__(self)
        dataframe = pandas.read_csv(filepath, usecols=[4], engine='python', skiprows=0)
        dataset = dataframe.values.reshape((np.prod(dataframe.shape), 1))
        self.dataset = dataset.astype('float64')

    def data(self):
        return self.dataset

    def __str__(self):
        return "co2"

