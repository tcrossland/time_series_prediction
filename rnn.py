import tensorflow as tf

from ann.datasets.bbva import Bbva
from ann.datasets.co2 import CO2
from ann.datasets.temperature import Temperature
from ann.models.simple_rnn import SimpleRnn
from ann.scenario import Config
from ann.scenario import Scenario

### CONFIGURATION ###
window_sizes = [4, 6, 8, 12, 16]
topologies = [[4], [8], [16], [32], [64], [128],
              [4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128],
              [4, 4, 4], [8, 8, 8], [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]]
time_series = [CO2(), Temperature(), Bbva()]
epochs = 300


def main():
    with tf.device('/gpu:1'):
        for ts in time_series:
            for window_size in window_sizes:
                for topology in topologies:
                    config = Config(time_series=ts, look_back=window_size, topology=topology, batch_size=1)
                    # models = [StatefulLstm(config), FeedForward(config), SimpleRnn(config), Gru(config), Lstm(config)]
                    model = SimpleRnn(config)
                    scenario = Scenario(model=model, config=config)
                    scenario.execute(epochs)


if __name__ == "__main__":
    main()
