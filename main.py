from ann.datasets.co2 import CO2
from ann.scenario import Config
from ann.scenario import Scenario
from ann.models.simple_rnn import SimpleRnn
from ann.models.lstm import Lstm
from ann.models.dense import FeedForward
from ann.models.gru import Gru
from ann.datasets.temperature import Temperature
import matplotlib.pyplot as plt
import numpy as np


def create_empty_plot(dataset):
    plot = np.empty_like(dataset)
    plot[:, :] = np.nan
    return plot


def create_left_plot(data, offset):
    plot = create_empty_plot(data)
    plot[offset:len(data) + offset, :] = data
    return plot


def create_right_plot(dataset, data):
    plot = create_empty_plot(dataset)
    plot[-len(data):, :] = data
    return plot


def plot(filename, look_back, dataset, predictions):
    # plt.xlim(dataset.size * 0.6, dataset.size * 0.8)
    plt.plot(dataset, color='black', label='Time Series')
    for (pred, label) in predictions:
        plt.plot(create_right_plot(dataset, pred), linestyle='solid', label=label)

    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(filename)
    plt.close()


def main():
    window_size = 6
    topologies = [[5], [16], [32], [64], [128], [5, 5], [16, 16], [32, 32], [64, 64], [128, 128], [5, 5, 5],
                  [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]]

    for topology in topologies:
        time_series = [Temperature(), CO2()]
        for ts in time_series:
            config = Config(time_series=ts, look_back=window_size, topology=topology, batch_size=1)
            topo = '-'.join([str(i) for i in config.topology])
            models = [FeedForward(config), SimpleRnn(config), Gru(config), Lstm(config)]
            scenarios = list(map(lambda m: Scenario(model=m, config=config), models))
            for epochs in range(1, 20):
                predictions = list(map(lambda s: (s.execute(1), str(s.model)), scenarios))
                filename = "out/{}/{}-{}-e{}.png".format(ts, topo, config, epochs)
                plt.figure(figsize=(16, 12))
                plt.title('Topology {} epoch {}'.format(topo, epochs))
                ts.window()
                plot(filename, window_size, ts.dataset, predictions)


if __name__ == "__main__":
    main()
