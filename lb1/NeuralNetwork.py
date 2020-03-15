import matplotlib.pyplot as plt
import torch
from matplotlib import rcParams

## Параметры размеров окна matplotlib
rcParams['figure.figsize'] = (10.0, 5.0)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(NeuralNetwork, self).__init__()
        # Создание входного слоя
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        # Функция активации
        self.act1 = torch.nn.Sigmoid()
        # Создание выходного слоя
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


def predict(net, x, y, plot_name):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.title('Prediction: $y = f(x)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.close()


def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


# Расчет метрики MAE
def metric(pred, target):
    return (pred - target).abs().mean()


def train_net_work(directory, neurons_amount, lr, x_train, y_train, x_valid, y_valid, epochs):
    sine_net = NeuralNetwork(neurons_amount)
    optimizer = torch.optim.Adam(sine_net.parameters(), lr)
    test_loss_history = []
    for epoch_index in range(epochs):
        optimizer.zero_grad()
        y_pred = sine_net.forward(x_train)
        loss_val = loss(y_pred, y_train)
        # print(loss_val) # вывод значения функции потерь для эпохи
        test_loss_history.append(loss_val)
        loss_val.backward()
        optimizer.step()
    # Вывод истории обучения
    predict(sine_net, x_valid, y_valid, './' + directory + '/after-train')
    plt.plot(test_loss_history)
    plt.title('Loss history')
    plt.xlabel('$epochs$')
    plt.ylabel('$loss$')
    plt.savefig('./' + directory + '/loss_history.png', bbox_inches='tight')
    plt.close()
    current_MAE = metric(sine_net.forward(x_valid), y_valid).item()
    print(current_MAE)


def influence_of_neurons(directory, neurons_amount, lr, x_train, y_train, x_valid, y_valid, epochs):
    neuronsMap = {}
    for n_neurons in range(10, neurons_amount - 1, 1):
        n_neurons = n_neurons + 1
        sine_net = NeuralNetwork(n_neurons)
        optimizer = torch.optim.Adam(sine_net.parameters(), lr)
        for epoch_index in range(epochs):
            optimizer.zero_grad()
            y_pred = sine_net.forward(x_train)
            loss(y_pred, y_train).backward()
            optimizer.step()
            print('epoch-count: ' + str(epoch_index) + ' neurons ammount ' + str(n_neurons))
        neuronsMap[n_neurons] = metric(sine_net.forward(x_valid), y_valid).item()
    plt.plot(list(neuronsMap.keys()), list(neuronsMap.values()))
    plt.title('MAE (N)')
    plt.xlabel('$Neurons Amount (N)$')
    plt.ylabel('$MAE$')
    plt.savefig('./' + directory + '/loss_funct_neurons.png', bbox_inches='tight')
    plt.close()


def influence_lr(directory, neurons_amount, lr, x_train, y_train, x_valid, y_valid, epochs):
    neuronsMap2 = {}
    lb = int((lr*0.1) * 1000)
    ub = int((lr + lr) * 1000)
    step = int((lr * 0.10) * 1000)

    for lri in range(lb, ub, step):
        sine_net = NeuralNetwork(neurons_amount)

        lrb = lri / 1000
        optimizer3 = torch.optim.Adam(sine_net.parameters(), lrb)
        for epoch_index in range(epochs):
            optimizer3.zero_grad()
            y_pred = sine_net.forward(x_train)
            loss(y_pred, y_train).backward()
            optimizer3.step()
            print('epoch-count: ' + str(epoch_index) + ' lr' + str(lrb))
        neuronsMap2[lrb] = metric(sine_net.forward(x_valid), y_valid).item()
    plt.plot(list(neuronsMap2.keys()), list(neuronsMap2.values()))
    plt.title('MAE (LR)')
    plt.xlabel('$Learning Rate (LR)$')
    plt.ylabel('$MAE$')
    plt.savefig('./' + directory + '/mae-lr.png', bbox_inches='tight')
    plt.close()


def find_optimal_params(x_train, y_train, x_valid, y_valid, epochs):
    MAE_hist = []
    min_MAE = 1
    min_lr = 0
    min_n = 0
    found = False
    for lri in range (10, 100, 1):
        lrb = lri / 1000
        for neurons in range (10, 50, 1):
            net = NeuralNetwork(neurons)
            optimizer = torch.optim.Adam(net.parameters(), lrb)
            for epoch_index in range(epochs):
                optimizer.zero_grad()
                y_pred = net.forward(x_train)
                loss(y_pred, y_train).backward()
                optimizer.step()
                print('epoch-count: ' + str(epoch_index) + ' neurons ammount ' + str(neurons) + ' lr ' + str(lrb))
            current_MAE = metric(net.forward(x_valid), y_valid).item()
            if (current_MAE < 0.03):
                min_MAE = current_MAE
                min_lr = lrb
                min_n = neurons
                found = True
                break
        if (found):
            break

    print('MIN MAE ' + str(min_MAE) + ' MIN lr ' + str(min_lr) + ' Min neurnos ' + str(min_n))


