import numpy as np
from keras.callbacks import Callback
from keras.layers import Dense
from keras.layers import LSTM

from .base import BaseConfiguration
from .prediction import Prediction


class ResetStatesCallback(Callback):
    def __init__(self, max_len=0):
        super().__init__()
        self.counter = 0
        self.max_len = max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
        self.counter += 1


class StatefulLstm(BaseConfiguration):
    def __init__(self, config):
        super().__init__(config)
        self.x, self.y = self.ts.create_dataset(look_back=1)

        # First layer - input shape should be specified
        self.model.add(
            LSTM(config.topology[0], batch_input_shape=(1, 1, 1), input_shape=(None, 1), activation=config.activation,
                 stateful=True,
                 return_sequences=(len(config.topology) > 1)))

        # Intermediate recurrent layers should return sequences
        for num_cells in config.topology[1:-1]:
            self.model.add(
                LSTM(num_cells, activation=config.activation, stateful=True, return_sequences=True))

        # Final layer
        if len(config.topology) > 1:
            self.model.add(LSTM(config.topology[-1], stateful=True, activation=config.activation))

        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer=config.optimizer)

    def train(self, epochs=10, batch_size=1):
        x = self.reshape(self.x)
        y = np.expand_dims(np.array([[v] for v in self.y.flatten()]).flatten(), axis=1)
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                              callbacks=[ResetStatesCallback(len(self.x) * self.validation_split)],
                              validation_split=self.validation_split, shuffle=False, verbose=2)

    def evaluate(self):
        x = self.reshape(self.x)
        y = np.expand_dims(np.array([[v] for v in self.y.flatten()]).flatten(), axis=1)
        pred = Prediction(self, x, y)
        print('Score: %.6f RMSE' % pred.score)
        return pred.plot

    def reshape(self, x):
        return np.expand_dims(np.expand_dims(x.flatten(), axis=1), axis=1)

    def __str__(self):
        return "slstm-{}".format(self.topology)
