import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for training on gpu

from scipy import signal
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from random import shuffle
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Reshape

# path to data files
path = "./nine_movs_six_sub_split/"

# path where you want to save trained model and some other files
sec_path = "./"


def create_dataset(file_path, persons):
    path = file_path + "{}_{}.txt"
    sgn = []
    lbl = []
    for i in persons:
        for j in range(9):
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


def create_dataset2(file_path, persons):
    path = file_path + "{}_{}.txt"
    sgn = []
    lbl = []
    i = persons
    for j in range(9):
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

    train_signals = sgn[0:int(0.6 * len(sgn))]
    train_labels = lbl[0:int(0.6 * len(lbl))]
    val_signals = sgn[int(0.6 * len(sgn)):int(0.8 * len(sgn))]
    val_labels = lbl[int(0.6 * len(lbl)):int(0.8 * len(lbl))]
    test_signals = sgn[int(0.8 * len(sgn)):]
    test_labels = lbl[int(0.8 * len(lbl)):]

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
    test_labels = to_categorical(test_labels)

    return train_signals, train_labels, val_signals, val_labels, test_signals, test_labels


def evaluate_model(model, expected_person_index=2):
    print("evaluate_model, expected_person_index:", expected_person_index)
    persons = [1, 2, 3, 4, 5, 6]
    persons.remove(expected_person_index)
    train_signals, train_labels, val_signals, val_labels = create_dataset(path, persons)
    model.evaluate(train_signals, train_labels)

    train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset2(path,
                                                                                                      expected_person_index)
    model.evaluate(train_signals, train_labels)


def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Эпоха')
    plt.ylabel('Вероятность корректного распознавания')
    # plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.grid(True)
    print(history.history['val_accuracy'])


# training model on 5 form 6 persons
a = [1, 3, 4, 5, 6]
train_signals, train_labels, val_signals, val_labels = create_dataset(path, a)

num_classes = 9
num_sensors = 1
input_size = train_signals.shape[1]

model = Sequential()
model.add(Reshape((input_size, num_sensors), input_shape=(input_size,)))
model.add(Conv1D(50, 10, activation='relu', input_shape=(input_size, num_sensors)))
model.add(Conv1D(25, 10, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Conv1D(100, 10, activation='relu'))
model.add(Conv1D(50, 10, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Dropout(0.5))
# next layers will be retrained
model.add(Conv1D(100, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# elapsed_time = time.time() - start_time # training time

# loss, accuracy = model.evaluate(val_signals, val_labels) # evaluating model on test data

# loss = float("{0:.3f}".format(loss))
# accuracy = float("{0:.3f}".format(accuracy))
# elapsed_time = float("{0:.3f}".format(elapsed_time))

# saving some data
# f = open(sec_path + "info.txt", 'w')
# f.writelines(["loss: ", str(loss), '\n', "accuracy: ", str(accuracy), '\n', "elapsed_time: ", str(elapsed_time), '\n'])

# saving model
# model.save(sec_path + "pretrained_model.h5")

# saving test data just in case
# cc = list(zip(test_signals, test_labels))
# with open(sec_path + "pretrained_model_test_data.txt", "wb") as fp:
#   pickle.dump(cc, fp)

# saving history
# with open(sec_path + "pretrained_model_history.h5", "wb") as fp:
#    pickle.dump(history.history, fp)

start_time = time.time()

history = model.fit(train_signals, train_labels,
                    steps_per_epoch=25,
                    epochs=100,
                    batch_size=None,
                    validation_data=(val_signals, val_labels),
                    # validation_steps=25
                    )

train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset2(path, 2)

plot_history(history)

evaluate_model(model, 2)
# evaluate_model, expected_person_index: 2
# (2329, 400)
# 59/59 [==============================] - 0s 7ms/step - loss: 0.1076 - accuracy: 0.9721
# (491, 400)
# 10/10 [==============================] - 0s 6ms/step - loss: 4.5096 - accuracy: 0.3810


checkpoin_weights = []

for l in model.layers:
    checkpoin_weights.append(l.get_weights())

model2 = Sequential()
model2.add(Reshape((input_size, num_sensors), input_shape=(input_size,)))
# Новый слой (КИХ фильтра)
# 1 свертка из 11 чисел
length_of_conv_filter = 11
model2.add(Conv1D(1, length_of_conv_filter, activation='linear', input_shape=(input_size, num_sensors), padding='same',
                  name="Filter"))
model2.add(Dropout(0.5))
model2.add(Conv1D(50, 10, activation='relu', input_shape=(input_size, num_sensors), trainable='False'))
model2.add(Conv1D(25, 10, activation='relu', trainable='False'))
model2.add(MaxPooling1D(4))
model2.add(Conv1D(100, 10, activation='relu', trainable='False'))
model2.add(Conv1D(50, 10, activation='relu', trainable='False'))
model2.add(MaxPooling1D(4))
model2.add(Dropout(0.5))
# next layers will be retrained
model2.add(Conv1D(100, 10, activation='relu', trainable='False'))
model2.add(GlobalAveragePooling1D())
model2.add(Dense(num_classes, activation='softmax', trainable='False'))

# for i in range(1, 11):
#  model2.layers[i+1].set_weights(checkpoin_weights[i])

w = model2.layers[1].get_weights()
print(w[0].shape)

# Подбираем параметры для первого слоя КИХ фильтра
w[0] = w[0] * 0
w[0][int(length_of_conv_filter / 2), 0, 0] = 1
w[1] = w[1] * 0
plt.plot(w[0].flatten())

w = model2.layers[1].set_weights(w)

# Устанавливаем веса для уже обученных слоев
n_layers = 11
for i in range(1, n_layers):
    model2.layers[i + 2].set_weights(checkpoin_weights[i])

model2.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])

model2.evaluate(train_signals, train_labels)
# 10/10 [==============================] - 0s 6ms/step - loss: 4.8465 - accuracy: 0.3707

keras.utils.plot_model(model2, 'dense_image_classifier2.png', show_shapes=True)

history2 = model2.fit(train_signals, train_labels, epochs=25,
                      validation_data=(test_signals, test_labels))
# Epoch 25/25
# 10/10 [==============================] - 0s 44ms/step - loss: 0.6439 - accuracy: 0.7755 - val_loss: 0.8151 # - val_accuracy: 0.7172


plot_history(history2)


# функция вывода коэффициентов свёрточного слоя
def check_coef_conv_layer(model_name, num_layer, num_filter):
    # сохраняем в переменную коэффициенты наблюдаемого слоя
    layer = model_name.layers[num_layer].get_weights()

    # коэффициенты 'а' наблюдаемого слоя первой сети
    weights = layer[0]
    # коэффициенты 'b' наблюдаемого слоя первой сети
    biases = layer[1]
    # вывод данных на экран
    for i in range(10):
        print("k{} = {:7.4f}".format(i, weights[i][0][num_filter]))
    print("\nb = {:7.4f}".format(biases[num_filter]))


# функция вывода коеффициентов полносвязного слоя
def check_coef_dense_layer(model_name, num_layer, num_filter):
    # сохраняем в переменную веса наблюдаемого слоя сети
    layer = model_name.layers[num_layer].get_weights()

    # коэффициенты 'а' наблюдаемого слоя сети
    weights = layer[0]
    # коэффициенты 'b' наблюдаемого слоя сети
    biases = layer[1]
    # вывод данных на экран
    for i in range(10):
        print("k{} = {:7.4f}".format(i, weights[i][num_filter]))
    print("\nb = {:7.4f}".format(biases[num_filter]))


l = model.layers[10].get_weights()

evaluate_model(model2, 2)
# (2329, 400)
# 59/59 [==============================] - 0s 8ms/step - loss: 3.1865 - accuracy: 0.3768
# (491, 400)
# 10/10 [==============================] - 0s 6ms/step - loss: 0.6444 - accuracy: 0.7755

b = model2.layers[1].get_weights()
# b[0] - neurons weights
w, h = signal.freqz(b[0].flatten())

plt.figure(figsize=(7, 5))
plt.plot(w, 20 * np.log10(abs(h)), 'b', label='амплитудно-частотная характеристика 1 человек')
plt.grid(True)
plt.xlabel('Нормированная частота')
plt.ylabel('Амплитуда, dB')
plt.legend(loc='lower right')
# print(b[0])

# plt.set_xlabel('Frequency [rad/sample]')
# plt.set_ylabel('Amplitude [dB]', color='b')

plt.figure(figsize=(8, 5))
print(b[0].flatten())
print(abs(b[0].flatten().min()))
# plt.plot(np.log10(b[0].flatten()+0.1), label='импульсная характеристика')
plt.plot(b[0].flatten(), label='импульсная характеристика')
plt.grid(True)
plt.xlabel('коэффициент')
plt.ylabel('значение')
plt.legend(loc='upper right')
plt.title('импульсная характеристика')

# Обучаем теперь на третьем субъекте
train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = create_dataset2(path, 3)
model.evaluate(train_signals, train_labels)
history2 = model2.fit(train_signals, train_labels, epochs=25,
                      validation_data=(test_signals, test_labels))

evaluate_model(model2, 3)
# (2473, 400)
# 62/62 [==============================] - 0s 7ms/step - loss: 5.1025 - accuracy: 0.3064
# (347, 400)
# 7/7 [==============================] - 0s 7ms/step - loss: 0.3206 - accuracy: 0.8846