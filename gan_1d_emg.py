"""
  Created by PyCharm.
  User: antonvasilev <bysslaev@gmail.com>
  Date: 22/07/2020
  Time: 00:49
 """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import keras
from keras.layers import Dense, Dropout, Input, Conv1D
from keras.models import Model, Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import pandas as pd
from scipy import signal


def filter_signal(x):
    N = 10
    Fc = 40
    Fs = 1600
    h = signal.firwin(numtaps=N, cutoff=Fc, nyq=Fs / 2)
    y = signal.lfilter(h, 1.0, x)
    return y


def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(400,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


def plot_generated_images_emg(X_train, epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=examples)]

    generated_images = generator.predict(image_batch)
    #     generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.plot(generated_images[i])
        #         plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./output/gan_generated_image_emg %d.png' % epoch)


data = []
for i in range(6):
    for j in range(9):
        filename = 'data/rf-lab/nine_movs_six_sub_split/' + str(i + 1) + '_' + str(j + 1) + '.txt'
        x = np.load(filename, allow_pickle=True)
        x = np.array(x)
        for line in x:
            line = np.insert(line, 0, [i, j])
            data.append(line)

df = pd.DataFrame(data)
df = df.rename(columns={0: "subject", 1: "move"})


# display(df)
# display(df.count().subject)


def create_generator_emg():
    generator = Sequential()
    generator.add(Dense(units=32, input_dim=400, activation='relu'))
    #     generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=64, activation='relu'))
    #     generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=128, activation='relu'))
    #     generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=512, activation='relu'))
    #     generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=128, activation='relu'))

    generator.add(Dense(units=512, activation='relu'))
    #     generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=128, activation='relu'))
    #     generator.add(LeakyReLU(0.2))

    #     generator.add(Dense(units=100, activation='tanh'))
    generator.add(Dense(units=400, activation='tanh'))

    generator.compile(loss='mean_squared_error', optimizer=adam_optimizer())
    return generator


def create_generator_conv(batch_size=128):
    generator = Sequential()
    generator.add(Conv1D(32, 25, activation='relu', input_shape=(batch_size, 400)))
    generator.add(Conv1D(64, 9, activation='relu'))
    generator.add(Conv1D(128, 5, activation='relu'))
    generator.add(Dropout(0.3))
    generator.add(Conv1D(256, 3, activation='relu'))
    generator.add(Conv1D(32, 5, activation='relu'))

    generator.add(Dense(units=400, activation='tanh'))

    generator.compile(loss='mean_squared_error', optimizer=adam_optimizer())

    return generator


def create_discriminator_emg(n_classes=9):
    discriminator = Sequential()
    discriminator.add(Dense(units=64, input_dim=400))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=32))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=n_classes, activation='softmax'))

    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator


# Метод обучения
def training_emg(X_train, X_noise, Y_train, epochs=1, batch_size=128):
    # Loading the data
    num_values = len(X_train)
    batch_count = num_values / batch_size
    X_train = df_train.drop(columns=['subject', 'move']).values

    # TODO удалить коммент
    # print(Y_train)

    # Creating GAN
    generator = create_generator_emg()
    discriminator = create_discriminator_emg()
    gan = create_gan(discriminator, generator)

    discriminator.trainable = True
    discriminator.train_on_batch(X_train, Y_train)
    discriminator.trainable = False
    for e in range(1, epochs + 1):
        print("Epoch %d" % e)
        for _ in tqdm(range(int(batch_count))):
            # Get a random set of  real images
            random_indexes_for_batch = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
            image_batch = X_train[random_indexes_for_batch]
            y_batch = Y_train[random_indexes_for_batch, :]

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(image_batch)

            # Construct different batches of  real and fake data
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.concatenate((y_batch, y_batch))
            # TODO удалить коммент
            # display(y_dis)
            # print(y_dis.shape)
            # return

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            y_gen = y_batch

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(image_batch, y_gen)

        if e == 1 or e % 2 == 0:
            plot_generated_images_emg(X_train, e, generator, 3)
    return (generator, discriminator)


df_train = df.sample(frac=1)
Y_train = pd.get_dummies(df_train['move']).values
X_emg = df_train.drop(columns=['subject', 'move']).values
X_emg_noise = X_emg.copy() / X_emg.max()

filtered_values = X_emg.copy()
# display(filtered_values)

for i in range(len(filtered_values)):
    filtered_values[i] = filter_signal(filtered_values[i])
X_emg = filtered_values / filtered_values.max()
# (g,d) = training(X_emg, X_emg_noise, 60, 32)


(g, d) = training_emg(X_emg, X_emg_noise, Y_train, 15, 128)

random_value = np.random.random((1, 400)) * 2 - 1
print(d.predict(random_value))
# print(random_value)
print(X_emg.shape)
predicted = g.predict(np.array([X_emg[0]]))

plt.figure()
plt.plot(X_emg[0], label='filtered')

plt.plot(X_emg_noise[0], label='raw')
plt.plot(predicted[0], label='predicted')
plt.legend()
plt.savefig('tempfig.jpg')
# Оцениваем точности классификатора
print('evaluate', d.evaluate(X_emg, Y_train))
predicted = d.predict(X_emg)
print('X_emg', X_emg.shape)
print(predicted.shape)
print(Y_train.shape)
