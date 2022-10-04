import os
import re
import urllib.request as urllib
import rarfile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

from src.data_manager.base_manager import BaseManager
from scipy import signal
import pickle
import sys
import os
from random import shuffle


class RflabDataManager(BaseManager):
    def __init__(self, persons = None, moves = None):
        super().__init__()
        # TODO актуализировать
        self.dataset_remote_url = 'https://github.com/RF-Lab/emg_platform/raw/master/data/nine_movs_six_sub_split.rar'
        self.path = './data/rf-lab/nine_movs_six_sub_split/'
        self.sc = MinMaxScaler(feature_range=(0, 1))
        if (persons is None):
            self.persons = [0, 1, 2, 3, 4, 5, 6]
        else:
            self.persons = persons
        if (moves is None):
            self.moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            self.moves = moves

    def load(self):
        """
        Load the dataset
        :return:
        """

        if not os.path.isdir(self.path):
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
        path = path + "{}_{}.pickle"
        sgn = []
        lbl = []
        for i in self.persons:
            for j in self.moves:
                with open(path.format(i, j + 1), "rb") as fp:  # Unpickling
                    data = pickle.load(fp)

                for k in range(np.shape(data)[0]):
                    sgn.append(data[k])
                    lbl.append(j)

        sgn = np.asarray(sgn, dtype=np.float32)
        lbl = np.asarray(lbl, dtype=np.int32)

        c = list(zip(sgn, lbl))
        shuffle(c)
        sgn, lbl = zip(*c)

        sgn = np.asarray(sgn, dtype=np.float64)
        lbl = np.asarray(lbl, dtype=np.int64)

        print(sgn.shape)

        train_signals = sgn[0:int(0.8 * len(sgn))]
        train_labels = lbl[0:int(0.8 * len(lbl))]
        val_signals = sgn[int(0.8 * len(sgn)):]
        val_labels = lbl[int(0.8 * len(lbl)):]
        # test_signals = sgn[int(0.8*len(sgn)):]
        # test_labels = lbl[int(0.8*len(lbl)):]

        train_labels = to_categorical(train_labels)
        val_labels = to_categorical(val_labels)
        # test_labels = to_categorical(test_labels)

        return train_signals, train_labels, val_signals, val_labels

    def download(self):
        print('rflab dataset dowloading...')
        temp_path = os.path.abspath("./data/rf-lab/temp.rar")
        urllib.urlretrieve(self.dataset_remote_url, temp_path)

        with rarfile.RarFile(temp_path, "r") as rf:
            rf.extractall(os.path.abspath("./data/rf-lab"))

        if os.path.isfile(temp_path):
            os.unlink(temp_path)
