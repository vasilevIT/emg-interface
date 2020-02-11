"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 11/02/2020
  Time: 19:47
 """

import numpy as np
from keras.optimizers import SGD

from src.data_manager.rflab_np_manager import RflabNpDataManager
import pandas as pd

from keras.utils import to_categorical, normalize
from keras.layers import Dropout, Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=400))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

rflab_manager = RflabNpDataManager()
dataset = rflab_manager.load()
X_train = dataset[1:, 0:-1]
row_sums = X_train.sum(axis=1)
X_train = X_train / row_sums[:, np.newaxis]
# X_train = np.random.random((1480, 400))
Y_train = dataset[1:, -1]

# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(Y_train, num_classes=9)

# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, one_hot_labels, epochs=10, batch_size=32, validation_split=0.75)
print(X_train[1:10, :])