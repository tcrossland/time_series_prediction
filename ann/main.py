from ann.datasets.co2 import CO2
from ann.scenario import Config
from ann.scenario import Scenario
from ann.models.simple_rnn import SimpleRnn
from ann.models.lstm import Lstm
from ann.models.dense import FeedForward
from ann.models.gru import Gru
from ann.datasets.temperature import Temperature


def main():
    topologies = [[5], [16], [32], [64], [128], [5, 5], [16, 16], [32, 32], [64, 64], [128, 128], [5, 5, 5],
                  [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]]
    for topology in topologies:
        for epochs in [1, 2, 3, 4, 5, 10, 20, 50]:
            for ts in [Temperature(), CO2()]:
                config = Config(time_series=ts, look_back=6, topology=topology, epochs=epochs, batch_size=1)
                for model in [FeedForward(config), SimpleRnn(config), Lstm(config), Gru(config=config)]:
                    print()
                    print()
                    print(">>>> {} + {} (epochs={}, topology={})".format(model, ts, epochs, topology))
                    scenario = Scenario(model=model, config=config)
                    scenario.execute()


if __name__ == "__main__":
    main()
