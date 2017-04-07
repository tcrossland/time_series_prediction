import matplotlib.pyplot as plt
import numpy as np
import time


class Config:
    def __init__(self, time_series, look_back=6, batch_size=1, epochs=10, topology=None, validation_split=0.3,
                 activation='tanh', optimizer='adam'):
        self.time_series = time_series
        self.look_back = look_back
        self.batch_size = batch_size
        self.epochs = epochs
        if topology is None:
            topology = [5]
        self.topology = topology
        self.validation_split = validation_split
        self.activation = activation
        self.optimizer = optimizer

    def __str__(self):
        return "w{}-e{}-b{}".format(self.look_back, self.epochs, self.batch_size)


class Scenario:
    def __init__(self, model, config):
        self.model = model
        self.time_series = config.time_series
        self.dataset = config.time_series.dataset
        self.config = config

    def execute(self):
        self.model.summary()
        start = time.clock()
        self.model.train(epochs=self.config.epochs, batch_size=self.config.batch_size)
        self.training_time = time.clock() - start
        print("Training time: %.3f" % self.training_time)
        train_predict, test_predict = self.model.evaluate()
        self.plot(train_predict.prediction, test_predict.prediction)

    def create_empty_plot(self):
        plot = np.empty_like(self.dataset)
        plot[:, :] = np.nan
        return plot

    def create_left_plot(self, data):
        offset = self.config.look_back
        plot = self.create_empty_plot()
        plot[offset:len(data) + offset, :] = data
        return plot

    def create_right_plot(self, data):
        plot = self.create_empty_plot()
        plot[-len(data):, :] = data
        return plot

    def plot(self, train_predict, test_predict):
        plt.figure(figsize=(16, 12))
        plt.xlim(self.dataset.size * 0.6, self.dataset.size * 0.8)
        for plot in [self.dataset, self.create_left_plot(train_predict), self.create_right_plot(test_predict)]:
            plt.plot(plot)

        filename = "out/{}/{}-{}.png".format(self.time_series, self.model, self.config)

        plt.savefig(filename)
        plt.close()
