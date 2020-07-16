"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 12/11/2019
  Time: 22:18
 """
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data_manager.base_manager import BaseManager
from src.util.path_helper import get_path


class RspoDataManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.path = '/data/rspo/'
        self.sc = MinMaxScaler(feature_range=(0, 1))

    def load(self):
        """
        Load the dataset
        :return:
        """
        self.raw_data = self.load_raw(self.path)
        self.normalize_data = self.normalize_data(self.raw_data)
        self.result_data = self.normalize_data
        self._X = self.result_data[0]
        self._y = self.result_data[1]
        return self.result_data

    def load_raw(self, path):
        """
        Loaded raw data
        :param str path:
        :return:
        """
        rock_dataset = pd.read_csv(get_path(path + "0.csv"), header=None)  # class = 0
        scissors_dataset = pd.read_csv(get_path(path + "1.csv"), header=None)  # class = 1
        paper_dataset = pd.read_csv(get_path(path + "2.csv"), header=None)  # class = 2
        ok_dataset = pd.read_csv(get_path(path + "3.csv"), header=None)  # class = 3
        frames = [rock_dataset, scissors_dataset, paper_dataset, ok_dataset]

        dataset = pd.concat(frames)

        dataset_train = dataset.iloc[np.random.permutation(len(dataset))]
        dataset_train.reset_index(drop=True)

        X_train = []
        y_train = []

        for i in range(0, dataset_train.shape[0]):
            row = np.array(dataset_train.iloc[i:1 + i, 0:64].values)
            X_train.append(np.reshape(row, (64, 1)))
            y_train.append(np.array(dataset_train.iloc[i:1 + i, -1:])[0][0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return [X_train, y_train]

    def normalize_data(self, raw_data):
        """
        Normalize data
        :param list raw_data:
        :return:
        """
        X_train = raw_data[0]
        y_train = raw_data[1]

        # Reshape to one flatten vector
        X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], 1)
        X_train = self.sc.fit_transform(X_train)

        # Reshape again after normalization to (-1, 8, 8)
        X_train = X_train.reshape((-1, 8, 8))

        # Convert to one hot
        y_train = np.eye(np.max(y_train) + 1)[y_train]
        return [X_train, y_train]
