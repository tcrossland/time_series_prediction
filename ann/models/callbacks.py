import csv
import os
import time
from collections import OrderedDict

from keras.callbacks import Callback


class CustomCallback(Callback):
    def __init__(self, scenario, filename=None, separator=',', append=True):
        super().__init__()
        self.scenario = scenario
        config = scenario.config
        self.filename = filename
        self.sep = separator
        self.append = append
        self.append_header = True
        self.epoch_history = []
        self.epoch_begin = 0
        self.train_begin = 0
        self.meta = {
            'series': str(config.time_series),
            'window': config.look_back,
            'type': str(scenario),
            'layers': len(config.topology),
            'topology': 'T' + '-'.join([str(i) for i in config.topology])
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        now = time.perf_counter()
        dur_epoch = (now - self.epoch_begin) * 1000
        dur_accum = (now - self.train_begin) * 1000
        hist = {'dur_epoch': dur_epoch, 'dur_accum': dur_accum}
        if logs is not None:
            hist.update(logs)
        self.epoch_history.append(hist)

    def on_train_begin(self, logs=None):
        self.train_begin = time.perf_counter()

    def on_train_end(self, logs=None):
        class CustomDialect(csv.excel):
            delimiter = self.sep

        now = time.perf_counter()
        train_duration = (now - self.train_begin) * 1000
        print('Training completed (%d epochs) in %d ms' % (len(self.epoch_history), train_duration))
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    self.append_header = not bool(len(f.readline()))
            csv_file = open(self.filename, 'a')
        else:
            csv_file = open(self.filename, 'w')

        hist_keys = self.epoch_history[0].keys()
        writer = csv.DictWriter(csv_file,
                                fieldnames=['epoch'] + sorted(self.meta.keys()) + sorted(hist_keys),
                                dialect=CustomDialect)
        if self.append_header:
            writer.writeheader()

        for epoch, hist in enumerate(self.epoch_history):
            row = OrderedDict({'epoch': epoch})
            row.update((key, self.meta[key]) for key in sorted(self.meta.keys()))
            row.update((key, hist[key]) for key in sorted(hist_keys))
            writer.writerow(row)
        csv_file.flush()
        csv_file.close()
