import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TimeSeries:
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.dataset = []

    def scaled(self):
        return self.scaler.fit_transform(self.dataset)

    def unscaled(self, dataset):
        return self.scaler.inverse_transform(dataset)

    def create_indexed_dataset(self, look_back=1, index_feature_range=(0.05, 0.95)):
        scaler = MinMaxScaler(feature_range=index_feature_range)
        dataset = self.scaled()
        x = [dataset[i:i + look_back] for i in range(0, len(dataset) - look_back)]
        y = [dataset[i] for i in range(look_back, len(dataset))]
        n = len(y)
        t = np.fromiter(range(n), np.float64).reshape((n, 1))
        t = scaler.fit_transform(t)
        x = np.array(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        return np.hstack((t, x)).reshape((n, look_back + 1, 1)), np.array(y)

    def create_dataset(self, look_back=1):
        dataset = self.scaled()
        x = [dataset[i:i + look_back] for i in range(0, len(dataset) - look_back)]
        y = [dataset[i] for i in range(look_back, len(dataset))]
        return np.array(x), np.array(y)

    def plot(self, filepath=None, title=None, x_min: int = 0, x_max: int = None):
        if x_max is None:
            x_max = len(self.dataset)
        # plt.rcParams['figure.figsize'] = (20.0, 12.0)
        plt.plot(self.dataset)
        plt.xlim(x_min, x_max)
        if title is not None:
            plt.title(title)
        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath)
            plt.close()
