from keras.optimizers import Adam
from keras.layers import TimeDistributed, Flatten
from keras.layers import Conv1D
from keras.optimizers import Adam
from keras.layers import TimeDistributed, Flatten, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import TimeDistributed, Flatten
from src.data_manager.rflab_np_manager import RflabNpDataManager
import numpy as np
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, BatchNormalization, LSTM, Conv1D, Reshape, Dropout
from keras.datasets import imdb
from keras.utils import pad_sequences
import matplotlib.pyplot as plt


class Factory:
    """
    Factory for building neural network model
    """

    def __init__(self):
        pass

    def build(self, model_name=None, input_size=400, num_sensors=1, num_classes=9):
        """
        Build one model
        :param model_name:
        :return:
        """

        if model_name == 'lstm':
            return self._build_lstm(input_size, num_sensors, num_classes)
        elif model_name == 'conv1d':
            return self._build_conv1d(input_size, num_sensors, num_classes)
        elif model_name == 'dense':
            return self._build_dense(input_size, num_sensors, num_classes)

        raise Exception('Wrong value of parameter: model_name')

    def build_all(self, input_size=400, num_sensors=1, num_classes=9):
        """
        Build the list with all models
        :return: list
        """
        models = list()

        models.append(self._build_dense(input_size, num_sensors, num_classes))
        models.append(self._build_conv1d(input_size, num_sensors, num_classes))
        models.append(self._build_lstm(input_size, num_sensors, num_classes))

        return models

    def _build_lstm(self, input_size, num_sensors, num_classes):
        """
            LSTM model
            """

        model = Sequential()
        # RNN
        model.add(Reshape((40, 10), input_shape=(input_size,)))
        model.add(LSTM(units=64, return_sequences=True, input_shape=(None, 40, 10)))
        model.add(Dropout(0.2))
        # model.add(LSTM(units=16,return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=50,return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(units=50))
        # model.add(Dropout(0.2))

        # FFNN
        # model.add(Dense(256, input_dim=400, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(BatchNormalization())
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Flatten())
        model.add((Dense(32, activation='relu')))
        model.add(Dropout(0.2))
        model.add((Dense(32, activation='relu')))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))

        opt = Adam(learning_rate=0.001)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

        return model

    def _build_dense(self, input_size, num_sensors, num_classes):
        """
            Simple Dense model
            """
        model = Sequential()
        # FFNN
        model.add(Dense(256, input_dim=input_size, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add((Dense(32, activation='relu')))
        model.add(Dropout(0.2))
        model.add((Dense(32, activation='relu')))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(num_classes, activation='softmax'))

        opt = Adam(learning_rate=0.01)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

        return model

    def _build_conv1d(self, input_size, num_sensors, num_classes):
        """
            Conv 1D model
            """

        model = Sequential()
        # Conv1d
        model.add(Reshape((input_size, num_sensors), input_shape=(input_size,)))
        model.add(Conv1D(25, 80, activation='relu', input_shape=(input_size, num_sensors)))
        model.add(Conv1D(25, 6, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(16, 6, activation='relu'))
        model.add(Conv1D(12, 4, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        opt = Adam(learning_rate=0.01)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

        return model
