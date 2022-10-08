from keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from scipy import signal
from random import shuffle


def _filter_signal(x):
    N = 10
    Fc = 40
    Fs = 1600
    h = signal.firwin(numtaps=N, cutoff=Fc, nyq=Fs / 2)
    y = signal.lfilter(h, 1.0, x)
    return y


def _filter_hz(X):
    """
    Filter the 50 hz noize
    :param X: numpy dataset
    :return:
    """
    X_train_filtered = []
    for X in X:
        X_train_filtered.append(_filter_signal(X))
    X_train_filtered = np.array(X_train_filtered)

    return X_train_filtered


def _filter_avg(X):
    # Moving Average Filter (SMA)
    X_train_filtered_sma0 = []
    for i0 in range(0, X.shape[0]):
        X_train_filtered_sma = []
        last_index = X[i0].shape[0]

        N = 20
        N_half = int(N / 2)

        for i in range(0, last_index):
            X_average = 0
            X_i = X[i0]
            if i >= N_half and i <= last_index - N_half:
                for j in range(i - N_half, i + N_half):
                    X_average += X_i[j]
                X_average /= N
            elif i < N_half:
                for j in range(0, i + N_half):
                    #                 print('+=')
                    #                 print(X_train_scaled[i0][j])
                    X_average += X_i[j]
                X_average /= (i + N_half)
            else:
                for j in range(i - N_half, last_index):
                    X_average += X_i[j]
                X_average /= (last_index - i + N_half)
            #     print('scaled')
            #     print(X_train_scaled[i])
            #     print('average')
            #     print(X_average)
            X_train_filtered_sma.append(X_average)

        #     plt.plot(X_train_scaled[i0],
        #              label='Сигнал scaled ' + str(np.argmax(train_labels[i]) + 1))
        #     plt.xlabel('Время')
        #     plt.ylabel('Импульс')
        #     plt.legend()
        #     plt.show()

        #     plt.plot(X_train_filtered_sma,
        #              label='Сигнал avg ' + str(np.argmax(train_labels[i]) + 1))
        #     plt.xlabel('Время')
        #     plt.ylabel('Импульс')
        #     plt.legend()
        #     plt.show()
        #     break
        X_train_filtered_sma0.append(X_train_filtered_sma)

    return X_train_filtered_sma0


def _scale(X):
    """
    Scale data
    :param X:
    :return:
    """

    X_min =  np.min(X)
    X_max = np.max(X)
    print("max: {0}, min: {1}".format(X_max, X_min))
    X = (X - X_min)/(X_max - X_min)
    return X
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    # transform data
    X_train_scaled = scaler.fit_transform(np.array(X))

    return X_train_scaled


def normalize(X):
    X = _filter_hz(X)
    X = _filter_avg(X)
    X = _scale(X)
    return X



def create_dataset(file_path, persons, moves=None):
    """
    Create dataset with 9 movements (indexes: 0 - 8)
    and 6 person (indexes: 1 - 6)
    :param file_path:
    :param persons:
    :param moves:
    :return:
    """
    if (moves is None):
        moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # /Users/antonvasilev/PyCharmProjects/emg-interface/nine_movs_six_sub_split/1_1.pickle
    if file_path[-1] != '/':
        file_path += '/'
    path = file_path + "{}_{}.txt"
    sgn = []
    lbl = []
    for i in persons:
        for j in moves:
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
