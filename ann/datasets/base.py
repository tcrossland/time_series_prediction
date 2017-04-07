from sklearn.preprocessing import MinMaxScaler
import numpy as np

class TimeSeries:
    def __init__(self, feature_range=(0.05, 0.95)):
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.dataset = []

    def scaled(self):
        return self.scaler.fit_transform(self.dataset)

    def unscaled(self, dataset):
        return self.scaler.inverse_transform(dataset)

    def create_dataset(self, look_back=1):
        dataset = self.scaled()
        dataX = [dataset[i:i + look_back] for i in range(0, len(dataset) - look_back)]
        dataY = [dataset[i] for i in range(look_back, len(dataset))]
        return np.array(dataX), np.array(dataY)
