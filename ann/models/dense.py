from .base import BaseConfiguration
from keras.layers import Dense
from ann.scenario import Config
import numpy as np


class FeedForward(BaseConfiguration):
    def __init__(self, config: Config):
        super().__init__(config)

        # First layer - input shape should be specified
        self.model.add(Dense(config.topology[0], input_dim=config.look_back, activation=config.activation))

        # No need to specify shape on intermediate layers
        for num_cells in config.topology[1:]:
            self.model.add(Dense(num_cells, activation=config.activation))

        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer=config.optimizer)
        self.model.summary()

    def __str__(self):
        return "dense-{}".format(self.topology)

    def reshape(self, x):
        return np.reshape(x, (x.shape[0], x.shape[1]))
