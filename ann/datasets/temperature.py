import pandas as pd
import numpy as np
from .base import TimeSeries
from sklearn.preprocessing import Imputer

class Temperature(TimeSeries):
    def __init__(self, filepath='../data/station.csv'):
        TimeSeries.__init__(self)
        dataframe = pd.read_csv(filepath, usecols=range(1,13), engine='python', skiprows=0)
        imp = Imputer(missing_values=999.90)
        imp.fit(dataframe)
        dataset = imp.transform(dataframe)
        dataset = dataset.reshape((np.prod(dataset.shape), 1))
        self.dataset = dataset.astype('float64')

    def data(self):
        return self.dataset

    def __str__(self):
        return "temp"