import math
import numpy as np
from .prediction import Prediction
from collections import deque
from keras.models import Sequential
from sklearn.metrics import mean_squared_error


class BaseConfiguration:
    def __init__(self, config, validation_split=0.3, activation='tanh', optimizer='adam'):
        self.ts = config.time_series
        self.x, self.y = self.ts.create_dataset(look_back=config.look_back)
        self.activation = activation
        self.optimizer = optimizer
        self.look_back = config.look_back
        self.validation_split = validation_split
        self.train_size = int(len(self.x) * (1.0 - validation_split))
        self.test_size = len(self.x) - self.train_size
        self.model = Sequential()
        self.topology = '-'.join([str(i) for i in config.topology])

    def summary(self):
        self.model.summary()

    def train(self, epochs=10, batch_size=1):
        x = self.reshape(self.x)
        return self.model.fit(x, self.y, epochs=epochs, batch_size=batch_size,
                              validation_split=self.validation_split, shuffle=False, verbose=2)

    def predict(self, inputs, batch_size=1):
        return self.model.predict(inputs, batch_size=batch_size)

    def reshape(self, x):
        return x

    def evaluate(self):
        pred = Prediction(self, self.reshape(self.x), self.y)
        print('Score: %.6f RMSE' % pred.score)
        return pred.plot

    def predict_future(self):
        [_train_x, test_x] = np.array_split(self.reshape(self.x), [self.train_size])
        [_train_y, test_y] = np.array_split(self.y, [self.train_size])
        test_predict = self.predict(test_x)
        latest_prediction = test_predict[0]
        fut_predict = [latest_prediction]

        dq = deque(test_x[0].copy())
        future_xs = []

        for i in range(self.test_size - 1):
            dq.popleft()
            dq.append(latest_prediction)
            future_x = np.reshape(dq, (1, self.look_back, 1))
            future_xs = np.reshape(np.append(future_xs, future_x), (i + 1, self.look_back, 1))
            latest_prediction = self.predict(future_xs)[-1][0]
            fut_predict.append(latest_prediction)

        fut_predict = np.reshape(np.array([fut_predict]), test_predict.shape)
        fut_score = math.sqrt(mean_squared_error(self.ts.unscaled(test_y[0]), self.ts.unscaled(fut_predict[0])))
        print('Future Score: %.6f RMSE' % fut_score)
        return fut_score
