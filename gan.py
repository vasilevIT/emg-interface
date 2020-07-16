import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


def get_sin(length=100, f=10, max_time=50):
    w = 2. * np.pi * f
    t = np.linspace(0, max_time, length)
    return np.sin(w * t)


def gen_signal(length=100):
    return get_sin(length, 10, 50)


def get_noise(length=100):
    real_f = 50
    noise_f = np.random.randint(low=-5, high=5)
    f = real_f + noise_f
    return get_sin(length, f, 50)


def get_noise_signal(length=100):
    return gen_signal(length) + get_noise(length)


def get_noise_batch(batch_size):
    data = []
    for i in range(batch_size):
        data.append(get_noise_signal())
    return np.array(data)


def load_data():
    #     (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #     x_train = (x_train.astype(np.float32) - 127.5)/127.5

    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    #     x_train = x_train.reshape(60000, 784)
    x_train = []
    for i in range(10000):
        x = []
        y = get_sin()
        x_train.append(y)
    return (np.array(x_train))


def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def create_generator():
    generator = Sequential()
    generator.add(Dense(units=32, input_dim=100))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=64))
    generator.add(LeakyReLU(0.2))

    #     generator.add(Dense(units=1024))
    #     generator.add(LeakyReLU(0.2))

    #     generator.add(Dense(units=100, activation='tanh'))
    generator.add(Dense(units=100))

    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator


def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=64, input_dim=100))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=32))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=2, activation='softmax'))

    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


def plot_generated_images(X_train, epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    #     noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    #     image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
    noise_signal = get_noise_batch(examples)

    generated_images = generator.predict(noise_signal)
    #     generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.plot(generated_images[i])
        #         plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' % epoch)


def training(X_train, epochs=1, batch_size=128):
    # Loading the data
    batch_count = X_train.shape[0] / batch_size

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=X_train.shape[0])]

    noise_signal = get_noise_batch(X_train.shape[0])
    generated_images = g.predict(noise_signal)
    # Construct different batches of  real and fake data
    X = np.concatenate([image_batch, noise_signal, generated_images])

    # Labels for generated and real data
    y_dis = np.zeros((3 * X_train.shape[0], 2))
    y_dis[:, 1] = 1
    discriminator.trainable = True
    discriminator.fit(X, y_dis, epochs=3)

    for e in range(1, epochs + 1):
        print("Epoch %d" % e)
        for _ in tqdm(range(batch_size)):
            # generate  random noise as an input  to  initialize the  generator
            #             noise= np.random.normal(0,1, [batch_size, 100])
            # Get a random set of  real images
            image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)]

            noise_signal = get_noise_batch(batch_size)
            #             print(image_batch[0])
            #             print(noise_signal[0])
            #             plt.plot(image_batch[0])
            #             plt.plot(noise_signal[0])
            #             break
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise_signal)

            # Construct different batches of  real and fake data
            X = np.concatenate([image_batch, noise_signal, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros((3 * batch_size, 2))
            y_dis[:, 1] = 1
            y_dis[:2 * batch_size] = [1, 0]

            # Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Tricking the noised input of the Generator as real data
            #             noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones((batch_size, 2))
            y_gen[:, 1] = 1

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training  the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise_signal, y_gen)

        if e == 1 or e % 2 == 0:
            plot_generated_images(X_train, e, generator, 3)
    return (generator, discriminator)


(X_train) = load_data()
print(X_train.shape)

plt.plot(X_train[0])
plt.plot(get_noise())

g = create_generator()
g.summary()

d = create_discriminator()
d.summary()

gan = create_gan(d, g)
gan.summary()

# Обучение GAN
(g, d) = training(X_train, 25, 128)

noise = np.random.normal(0, 1, [1, 100])
predicted_by_noise = g.predict(noise)[0]
plt.figure(figsize=(20, 5))

plt.subplot(2, 1, 1)
plt.title('Noize and filtered')
plt.ylabel('Value')
plt.plot(noise[0], label='noize')
plt.legend()
plt.ylabel('Value')
plt.subplot(2, 1, 2)
plt.plot(predicted_by_noise, label='filtered')
plt.xlabel('Time')
plt.legend()

plt.figure(figsize=(20, 5))
predicted_by_real_signal = g.predict(np.array([X_train[0]]))[0]

plt.subplot(2, 1, 1)
plt.title('Pure signal and filtered')
plt.ylabel('Value')
plt.plot(X_train[0], label='Pure signal')
plt.legend()
plt.ylabel('Value')
plt.subplot(2, 1, 2)
plt.plot(predicted_by_real_signal, label='filtered')
plt.xlabel('Time')
plt.legend()

noised_signal = get_noise_signal()
plt.figure(figsize=(20, 5))
real_signal = gen_signal()
predicted_by_noised_signal = g.predict(np.array([noised_signal]))[0]
predicted_by_real_signal = g.predict(np.array([real_signal]))[0]

plt.subplot(4, 1, 1)
plt.title('Noised signal and filtered')
plt.ylabel('Value')
plt.plot(noised_signal, label='Noised signal')
plt.legend()

plt.ylabel('Value')
plt.subplot(4, 1, 2)
plt.plot(predicted_by_real_signal, label='filtered (by real)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.ylabel('Value')
plt.subplot(4, 1, 3)
plt.plot(predicted_by_noised_signal, label='filtered')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.subplot(4, 1, 4)
plt.ylabel('Value')
plt.plot(real_signal, label='Pure signal')
plt.xlabel('Time')
plt.legend()

print('estimation of the descriminator')
print('noise', d.predict(noise))
print('real signal', d.predict(np.array([real_signal])))
print('noised signal', d.predict(np.array([noised_signal])))

predicted = d.predict(np.array([gen_signal()]))
print(predicted)

noise = np.random.normal(0, 1, [1, 100])
predicted = d.predict(noise)
print(predicted)
