
from src.data_manager.rflab_np_manager import RflabNpDataManager
import numpy as np
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, BatchNormalization
from keras.datasets import imdb
from keras.utils import pad_sequences
import matplotlib.pyplot as plt

rflab_manager = RflabNpDataManager()
dataset = rflab_manager.load()
X_train = dataset[2:, 0:-1]
X_train[np.isnan(X_train)] = 0.
print('Shape of one row data:')
print(X_train.shape)
print(X_train[0].shape)
print(X_train[0])
X_train = np.reshape(X_train, (X_train.shape[0],1,400))
print('ReShape of one row data:')
print(X_train.shape)
print(X_train[0].shape)
print(X_train[0])

# row_sums = X_train.sum(axis=1)
# X_train = X_train / row_sums[:, np.newaxis]

Y_train = dataset[2:, -1]
print('Class labels:')
print(np.unique(Y_train))

# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(Y_train, num_classes=9)

model = Sequential()
input_dim = 400
# model.add(Embedding(input_dim=input_dim, output_dim=64))
model.add(SimpleRNN(64, input_shape=(1,400)))
model.add(BatchNormalization())
model.add(Dense(9, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, one_hot_labels, epochs=10, batch_size=32, validation_split=0.5)
y_pred = model.predict(X_train[0:1])
# print(X_train[0:1])
print('X_pred')
print(X_train[0:1])
print('y_pred')
print(y_pred)