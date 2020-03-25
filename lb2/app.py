# Вариант	Метод оптимизации	Число нейронов в скрытом слое,n_hidden_neurons	Шаг градиентного спуска,lr
#  4	        ADAM	                        5	                                        0.01

import torch
import random
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from lb2.WineNet import WineNet
# установка начальных значений

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
batch_size = 30  # размер кусочной выборки
epoch_count = 5000
n_hidden_neurons = 5
lr = 0.01
test_size = 0.1

# загрузка датасета
wine = sklearn.datasets.load_wine()
wine.data.shape

# формирование тренировочной выборки
X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, :2],
    wine.target,
    test_size=test_size,
    shuffle=True)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

Base_Rate = len(wine.target[wine.target == 1]) / len(wine.target)
print("Base Rate: ", Base_Rate)

# инициализация нейронной сети    
wine_net = WineNet(n_hidden_neurons)

loss = torch.nn.CrossEntropyLoss()

# объект оптимизатора Adam
optimizer = torch.optim.Adam(wine_net.parameters(), lr)

# итерация по эпохам
educationMap = {}
for epoch in range(epoch_count):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_indexes = order[start_index:start_index + batch_size]
        x_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]
        preds = wine_net.forward(x_batch)
        loss_value = loss(preds, y_batch)
        loss_value.backward()
        optimizer.step()

    if epoch % 100 == 0:
        test_preds = wine_net.forward(X_test)
        test_preds = test_preds.argmax(dim=1)
        educationMap[epoch] = (test_preds == y_test).float().mean()
        print("Точность предсказания: ", (test_preds == y_test).float().mean())

plt.plot(list(educationMap.keys()), list(educationMap.values()))
plt.title('Точность предсказания')
plt.xlabel('$epochs$')
plt.ylabel('$Точность предсказания, \% $')
plt.savefig('accuracy_plot.png', bbox_inches='tight')
plt.close()


# визуализация
# %matplotlib notebook

plt.rcParams['figure.figsize'] = (10, 8)

n_classes = 3
plot_colors = ['g', 'orange', 'black']
plot_step = 0.02

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = torch.meshgrid(torch.arange(x_min, x_max, plot_step),
                        torch.arange(y_min, y_max, plot_step))

preds = wine_net.inference(
    torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))

preds_class = preds.data.numpy().argmax(axis=1)
preds_class = preds_class.reshape(xx.shape)
plt.contourf(xx, yy, preds_class, cmap='Accent')

# for i, color in zip(range(n_classes), plot_colors):
#     indexes = np.where(y_train == i)
#     plt.scatter(X_train[indexes, 0],
#                 X_train[indexes, 1],
#                 c=color,
#                 label=wine.target_names[i],
#                 cmap='Accent')
#     plt.xlabel(wine.feature_names[0])
#     plt.ylabel(wine.feature_names[1])
#     plt.legend()
#     plt.show()