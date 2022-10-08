
from src.data_manager.rflab_simple import create_dataset, normalize
import os
import matplotlib.pyplot as plt


# Проверяем чтение набора данных, все ли ок
path = os.path.abspath('./data/rf-lab/nine_movs_six_sub_split/')
persons = [1]
movements = [0, 1]
X_train, y_train, _, __ = create_dataset(path, persons, movements)

print(X_train[0])


fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

axs[0].plot(X_train[0],
             label='Сигнал')
# axs[0].xlabel('Время')
# axs[0].ylabel('Импульс')
# axs[0].legend()


# noralize
# axs[0].figure()
X_train_normalize = normalize(X_train)
print(X_train_normalize[0])
axs[1].plot(X_train_normalize[0],
             label='Сигнал')
# axs[1].xlabel('Время')
# axs[1].ylabel('Импульс')
# axs[1].legend()
plt.show()