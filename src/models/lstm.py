"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 12/11/2019
  Time: 21:42
 """

from keras import Sequential
from keras.layers import LSTM, Dropout, Dense


class LstmModel:

    def __init__(self):
        self.model = None
        self.layers = 10
        self.shape = [400, 25000]
        pass

    def set_layers(self, layers):
        """
        Set number of layers.
        :param int layers:
        :return:
        """
        self.layers = layers

    def get_layers(self):
        """
        Return layers of built model.
        :return:
        """
        return self.model.layers

    def build(self):
        """
        Build Sequential model
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.shape[1], 8)))
        self.model.add(Dropout(0.2))
        for i in range(0, self.layers):
            self.model.add(LSTM(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))

        self.model.add(Dense(units=64))
        self.model.add(Dense(units=128))

        self.model.add(Dense(units=4, activation="softmax"))

        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.summary()

    def fit(self, X, Y):
        pass
