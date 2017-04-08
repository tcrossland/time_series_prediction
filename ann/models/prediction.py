import math
import numpy as np

from sklearn.metrics import mean_squared_error


class Prediction:
    def __init__(self, model, x, y):
        shape = x.shape
        xx = np.reshape(x, (shape[0], shape[1]))
        next_shape = (shape[0] - 1,) + shape[1:]
        #print(y)
        # print(x.shape)
        #print(y)
        # print(np.hstack((x[:,1:],y)).shape)
        #print(np.hstack((x,y))[:,1:-1])
        self.prediction = model.ts.unscaled(model.predict(x))
        self.score = math.sqrt(mean_squared_error(model.ts.unscaled(y), self.prediction[:, 0]))
        self.plot = self.prediction[1:]
        # print(x.shape, self.prediction.shape, self.plot.shape)
        self.next_x = np.reshape(np.hstack((xx[:,1:],y))[1:], next_shape)
        self.next_y = y[1:]
