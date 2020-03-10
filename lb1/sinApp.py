import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams
import os
import lr1.NeuralNetwork as nn

## Параметры размеров окна matplotlib
rcParams['figure.figsize'] = (10.0, 5.0)
direct = 'sin'

# Установка изначальных значений
lr = 0.01  # шаг градиентного спуска, лучшее значение на +-100% - 0.01
n_hidden_neurons = 50  # Число нейронов в скрытом слое, найти min значение - 50
epoch_count = 2000  # Количество эпох в обучении - 2000


def target_function(x):
    # Целевая функция
    return torch.sin(x)


# Формирование обучающей выборки
x_train = torch.rand(100)
x_train = x_train * 20.0 - 10.0
y_train = target_function(x_train)  # целевая функция
# Добавление "шума" в выборку
noise = torch.randn(y_train.shape) / 5.
plt.plot(x_train.numpy(), noise.numpy(), 'o')
plt.axis([-10, 10, -1, 1])
plt.title('Gaussian noise')
y_train = y_train + noise
plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('Noisy f(x)')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.savefig('./'+direct+'/train_dataset.png', bbox_inches='tight')
plt.close()

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

# Формирование валидационной выборки
x_validation = torch.linspace(-10, 10, 100)
y_validation = target_function(x_validation.data)  # целевая функция
plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
plt.title('Validation dataset: f(x)')
plt.xlabel('x_validation')
plt.ylabel('y_validation')
plt.savefig('./'+direct+'/valid_dataset.png', bbox_inches='tight')
plt.close()
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

# nn.influence_lr(direct, n_hidden_neurons, lr, x_train, y_train, x_validation, y_validation, epoch_count)
# nn.influence_of_neurons(direct, n_hidden_neurons, lr, x_train, y_train, x_validation, y_validation, epoch_count)
# nn.train_net_work(direct, n_hidden_neurons, lr, x_train, y_train, x_validation, y_validation, epoch_count)
nn.find_optimal_params(x_train,y_train,x_validation,y_validation,epoch_count)