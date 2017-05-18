"""Data util script for managing and processing data."""

import os
import numpy as np

class Data:

    def __init__(self):
        self.offset = 0
        self.data = None
        self.count = 0

    def __init__(self, filepath):
        self.offset = 0
        self.count = 0
        self.data = self.load_data(filepath)

    def load_data(self, filepath):
        self.data = np.load(filepath)
        self.count = self.data.shape[0]

        return np.load(filepath)

    def save_data(self, filepath):
        np.save(filepath, self.data)

    def get_data(self, shape=None):
        """Returns input data. Can be returned in desired shape."""
        data = np.array([row[0] for row in self.data])
        if shape != None:
            return np.array([np.reshape(data_point, shape)for data_point in data])
        else:
            return data

    def get_labels(self):
        """Returns data labels."""
        return np.array([row[1] for row in self.data])

    def shuffle_data(self):
        """Shuffles the data along axis=0."""
        np.random.shuffle(self.data)

    def next_batch(self, batch_size):
        """Returns the next data batch of size batch_size."""
        data_points = []
        labels = []
        for i in range(batch_size):
            idx = i + self.offset
            if idx >= self.data.shape[0]:
                self.offset = i - batch_size
                idx = i + self.offset

            data_points.append(self.data[idx][0])
            labels.append(self.data[idx][1])

        self.offset += batch_size

        return data_points, labels

if __name__ == "__main__":
    filepath = 'sample_data.npy'
    if not os.path.isfile(filepath):
        data = []
        for i in range(1000):
            data_pts = np.random.random(28*28)
            labels = np.random.random(10)

            data.append([data_pts, labels])

        data = np.array(data)
        np.save(filepath, data)

    my_data = Data()
    my_data.load_data(filepath)

    my_data.shuffle_data()
    print(my_data.get_data().shape)
    print(my_data.get_labels().shape)

    my_data.save_data(filepath)
