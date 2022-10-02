from keras import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
from keras.utils import pad_sequences
import matplotlib.pyplot as plt



max_words = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)



maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print(x_train[5002])



model = Sequential()
model.add(Embedding(max_words, 2, input_length=maxlen))
model.add(SimpleRNN(8))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=15,batch_size=128, validation_split=0.1)

plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем набор')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на валидационном набор')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.savefig('acc.png')
