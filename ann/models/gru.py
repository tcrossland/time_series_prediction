from keras.layers import Dense
from keras.layers import GRU

from .base import BaseConfiguration


class Gru(BaseConfiguration):
    def __init__(self, config):
        super().__init__(config)

        # First layer - input shape should be specified
        self.model.add(
            GRU(config.topology[0], input_shape=(None, 1), activation=config.activation,
                return_sequences=(len(config.topology) > 1)))

        # Intermediate recurrent layers should return sequences
        for num_cells in config.topology[1:-1]:
            self.model.add(
                GRU(num_cells, activation=config.activation, return_sequences=True))

        # Final layer
        if len(config.topology) > 1:
            self.model.add(GRU(config.topology[-1], activation=config.activation))

        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer=config.optimizer)

    def __str__(self):
        return "gru-{}".format(self.topology)
