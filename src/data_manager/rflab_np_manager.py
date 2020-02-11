"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 10/01/2020
  Time: 00:39
 """
import os
import re
import urllib.request as urllib
import rarfile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from src.data_manager.base_manager import BaseManager


class RflabNpDataManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.dataset_remote_url = 'https://github.com/RF-Lab/emg_platform/raw/master/data/nine_movs_six_sub_split.rar'
        self.path = './data/rf-lab/nine_movs_six_sub_split'
        self.sc = MinMaxScaler(feature_range=(0, 1))

    def load(self):
        """
        Load the dataset
        :return:
        """

        if not os.path.isfile(self.path):
            self.download()
        self.raw_data = self.load_raw(self.path)
        self.normalize_data = self.normalize_data(self.raw_data)
        self.result_data = self.normalize_data
        # self._X = self.result_data
        # self._y = np.zeros(self._X.shape)
        return self.result_data

    def normalize_data(self, data):
        # TODO
        return data

    def get_hand_gesture_class(self, file_name):
        index = int(re.sub(r'^\d_', "", file_name).replace('.txt', ''))
        if index > 0:
            index = index - 1
        return index

    def prepare_signal(self, mat, gesture_class):
        N = mat.shape[0]
        m = 400
        i = 0
        k = 0
        result = np.zeros((1, 401))
        while i < N:
            signal = mat[i:i + m]
            if signal.shape[0] < m:
                # print('added zeros')
                signal = np.append(signal, np.zeros((1, m - signal.shape[0])))
            signal = np.append(signal, [gesture_class])
            i = i + m
            result = np.vstack([result, signal])
            k = k + 1
        return result[1:]

    def read_signal(self, file_path):
        return np.fromfile(file_path)

    def load_raw(self, path):
        files = os.listdir(os.path.abspath(path))
        files.sort()

        dataset = np.zeros((1, 401))

        k = 0
        for file in files:
            if not file.endswith(".txt"):
                continue

            data = self.read_signal(path + '/' + file)
            gesture_class = self.get_hand_gesture_class(file)
            signal = self.prepare_signal(data, gesture_class)
            for x in signal:
                dataset = np.vstack([dataset, x])
            k = k + 1

        return np.array(dataset)

    def download(self):
        temp_path = os.path.abspath("./data/rf-lab/temp.rar")
        urllib.urlretrieve(self.dataset_remote_url, temp_path)

        with rarfile.RarFile(temp_path, "r") as rf:
            rf.extractall(os.path.abspath("./data/rf-lab"))

        if os.path.isfile(temp_path):
            os.unlink(temp_path)
