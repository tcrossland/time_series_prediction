import matplotlib.pyplot as plt
import numpy as np
import time


class Config:
    def __init__(self, time_series, look_back=6, batch_size=1, topology=None, validation_split=0.3,
                 activation='tanh', optimizer='adam'):
        self.time_series = time_series
        self.look_back = look_back
        self.batch_size = batch_size
        if topology is None:
            topology = [5]
        self.topology = topology
        self.validation_split = validation_split
        self.activation = activation
        self.optimizer = optimizer

    def __str__(self):
        return "w{}-b{}".format(self.look_back, self.batch_size)


class Scenario:
    def __init__(self, model, config):
        self.model = model
        self.time_series = config.time_series
        self.dataset = config.time_series.dataset
        self.epochs = 0
        self.config = config

    def execute(self, epochs):
        print()
        print()
        self.epochs = self.epochs + epochs
        print(">>>> {} + {} (epochs={}, topology={})".format(self.model, self.time_series, self.epochs, self.config.topology))
        self.model.summary()
        start = time.clock()
        self.model.train(epochs=epochs, batch_size=self.config.batch_size)
        self.training_time = time.clock() - start
        print("Training time: %.3f" % self.training_time)
        prediction = self.model.evaluate()
        # self.plot(predictions)
        return prediction

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

    def plot(self, predictions):
        plt.figure(figsize=(16, 12))
        plt.xlim(self.dataset.size * 0.6, self.dataset.size * 0.8)
        plt.plot(self.dataset)
        for pred in predictions:
            plt.plot(self.create_right_plot(pred))

        filename = "out/{}/{}-{}.png".format(self.time_series, self.model, self.config)

        plt.savefig(filename)
        plt.close()
