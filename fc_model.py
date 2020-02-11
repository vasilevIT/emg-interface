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
model.add(Dense(512, activation='relu', input_dim=400))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.1, nesterov=True)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

rflab_manager = RflabNpDataManager()
dataset = rflab_manager.load()
X_train = dataset[2:, 0:-1]
# i = 0
# for x in X_train:
#     print("i = %s max = %s" % (i, x.max()))
#     if x.max() == np.nan:
#         print(x.max())
#         print(x.min())
#         print(x.mean())
#         break
#     i = i + 1
# print("i = %s" % i)
# exit()
X_train[np.isnan(X_train)] = 0.
row_sums = X_train.sum(axis=1)
X_train = X_train / row_sums[:, np.newaxis]

X_train[np.isnan(X_train)] = 0.
#
# print(X_train.max())
# print(X_train.min())
# print(X_train.mean())
# exit()
# X_train = np.random.random((1480, 400))
Y_train = dataset[2:, -1]
print(Y_train)
exit()

# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(Y_train, num_classes=9)

# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, one_hot_labels, epochs=100, batch_size=32, validation_split=0.5)
y_pred = model.predict(X_train[0:1])
# print(X_train[0:1])
print(y_pred)
# print(X_train.max())
# print(X_train.min())
# print(X_train.mean())
# df = pd.DataFrame(X_train)
# print(df.info(verbose=True))
