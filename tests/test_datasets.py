from unittest import TestCase

import numpy as np
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from ann.datasets.temperature import Temperature


class TestTemperature(TestCase):
    def test_size(self):
        temp = Temperature()
        self.assertEqual(len(temp.dataset), 768)

    def test_scaled(self):
        for i in range(10):
            feature_range = np.sort(np.random.rand(2))
            temp = Temperature(feature_range=feature_range)
            scaled = temp.scaled()
            self.assertAlmostEqual(np.min(scaled), feature_range[0])
            self.assertAlmostEqual(np.max(scaled), feature_range[1])

    def test_create_dataset(self):
        temp = Temperature()
        scaled = temp.scaled()
        n = len(temp.dataset)
        for look_back in range(1, 10):
            x, y = temp.create_dataset(look_back=look_back)
            self.assertEqual(x.shape, (n - look_back, look_back, 1))
            self.assertEqual(y.shape, (n - look_back, 1))
            self.assertTrue(np.array_equal(scaled[0:look_back], x[0]))
            self.assertTrue(np.array_equal(scaled[-look_back - 1:-1], x[-1]))

    def test_create_dataset_with_index(self):
        temp = Temperature()
        scaled = temp.scaled()
        n = len(temp.dataset)
        for look_back in range(1, 10):
            x, y = temp.create_indexed_dataset(look_back=look_back)
            self.assertEqual(x.shape, (n - look_back, look_back + 1, 1))
            self.assertEqual(y.shape, (n - look_back, 1))
            self.assertTrue(np.array_equal(scaled[0:look_back], x[0][1:]))
            self.assertTrue(np.array_equal(scaled[-look_back - 1:-1], x[-1][1:]))

    def test_roll(self):
        temp = Temperature()
        x, y = temp.create_indexed_dataset(look_back=1)
        print(x[0:10])
        print(np.roll(x[0:10], -2))

    def test_convolve(self):
        from keras.layers.convolutional import Conv1D
        m = Sequential()
        m.add(Conv1D(1, 3))
        m.compile(loss='mean_squared_error', optimizer='adam')

