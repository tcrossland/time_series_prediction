import numpy as np
from keras.layers import Dense

from ann.scenario import Config
from .base import BaseConfiguration


class FeedForward(BaseConfiguration):
    def __init__(self, config: Config):
        super().__init__(config)
        activation = config.activation
        input_dim = config.look_back
        if config.include_index:
            input_dim += 1

        # First layer - input shape should be specified
        self.model.add(Dense(config.topology[0], input_dim=input_dim, activation=activation))

        # No need to specify shape on intermediate layers
        for num_cells in config.topology[1:]:
            self.model.add(Dense(num_cells, activation=activation))

        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer=config.optimizer)

    def __str__(self):
        return "dense-{}".format(self.topology)

    def reshape(self, x):
        return np.reshape(x, (x.shape[0], x.shape[1]))
