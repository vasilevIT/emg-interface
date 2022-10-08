
from src.data_manager.rflab_simple import create_dataset, normalize
import os
import matplotlib.pyplot as plt


# Проверяем чтение набора данных, все ли ок
path = os.path.abspath('./data/rf-lab/nine_movs_six_sub_split/')
persons = [1]
movements = [0, 1]
X_train, y_train, _, __ = create_dataset(path, persons, movements)

print(X_train[0])
plt.plot(X_train[0],
             label='Сигнал')
plt.xlabel('Время')
plt.ylabel('Импульс')
plt.legend()
plt.show()


# noralize
plt.figure()
X_train_normalize = normalize(X_train)
print(X_train_normalize[0])
plt.plot(X_train_normalize[0],
             label='Сигнал')
plt.xlabel('Время')
plt.ylabel('Импульс')
plt.legend()
plt.show()