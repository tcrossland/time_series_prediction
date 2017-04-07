import math

from sklearn.metrics import mean_squared_error


class Prediction:
    def __init__(self, model, x, y):
        self.prediction = model.ts.unscaled(model.predict(x))
        self.score = math.sqrt(mean_squared_error(model.ts.unscaled(y), self.prediction[:, 0]))
