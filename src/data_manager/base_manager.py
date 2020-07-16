"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 12/11/2019
  Time: 22:53
 """

import numpy as np


class BaseManager:

    def __init__(self):
        self._X = np.array([])
        self._y = np.array([])
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_test = np.array([])
        self.y_test = np.array([])

    def split_data(self, proportion=0.5):
        """
        :param float proportion: Proportion of train data.
        :return:
        """
        if (proportion <= 0) or (proportion >= 1):
            proportion = 0.5
        full_size = self._y.shape[0]
        test_size = full_size - int(full_size * proportion)
        if test_size <= 0:
            test_size = int(full_size / 2)
        train_size = full_size - test_size
        # Splitting Train/Test
        self.X_train = self._X[0:train_size]
        self.y_train = self._y[0:train_size]

        self.X_test = self._X[train_size:]
        self.y_test = self._y[train_size:]

    def print_stat(self):
        print("All Data size X and y")
        print(self._X.shape)
        print(self._y.shape)
        print("Train Data size X and y")
        print(self.X_train.shape)
        print(self.y_train.shape)
        print("Test Data size X and y")
        print(self.X_test.shape)
        print(self.y_test.shape)

    def get_train(self):
        return [self.X_train, self.X_train]

    def get_test(self):
        return [self.X_test, self.y_test]
