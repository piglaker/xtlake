import math
import numpy as np
import random
def normalization(x):
    maxx = max(x)
    minx = min(x)
    for i in range(len(x)):
        x[i] = (x[i] - minx) / (maxx - minx)
    return x


def batch(dataset, total = False,batch_size = 10):
    """
    dataset:
	    list[[[x1,x2,x3...],[y_]]]
    """
    if total == False:
        rand = [random.randint(0, len(dataset) - 1) for _ in range(0, batch_size)]
        batch = []
        for i in rand:
            batch.append(dataset[i])
    else:
        return dataset
    return  batch


def get_weight(in_):
    return np.random.uniform(-1, 1,size = in_)


def get_bias():
    return np.random.uniform(-1, 1)


def ReLU(x):
    return max(0, x)


def sigmoid(x):
    return 1/(1+math.exp(-x))


def grad_sig(x):
    return x*(1-x)


def grad_ReLU(x):
    if x > 0:
        return 1
    else:
        return 0


def grad_cross_entropy(a, y_):
    return np.array(a) - np.array(y_)


def loss(y, y_):
    v = []
    for i in range(len(y)):
        v.append(1/2*math.pow((y[i]-y_[i]), 2))
    return v


def grad_MSEloss(a, y_):
    return np.array(a) - np.array(y_)


def softmax(y):
    output = np.exp(y)
    result = output / output.sum()
    return result


def cross_entropy(y, y_):
    val = 0
    for i in range(len(y)):
        val += - (y_[i] * math.log(y[i]))
    return val


def argmax(y_):
    max = 0
    index = 0
    for i in range(len(y_)):
        if y_[i] >= max:
            max = y_[i]
            index = i
    return index


def BGDoptimizer(w, b, learning_rate, x, grad_y, loss_):
    w = w - learning_rate * x * grad_y * loss_
    b = b - learning_rate * grad_y * loss_
    return w, b


def update(layers, learning_rate=0.001):
    for i in range(len(layers)):
        layer = layers[i]
        weight = layer['weight_']
        bias = layer['bias_']
        loss_ = layer['loss_']
        x = layer['in_']
        y = layer['out_']
        for p in range(len(weight)):
            for q in range(len(weight[p])):
                if layer['activation_'] == 'sigmoid':
                    weight[p][q], bias[p] = BGDoptimizer(weight[p][q], bias[p], learning_rate,
                                                x[q], grad_sig(y[p]), loss_[p])
                else:
                    weight[p][q], bias[p] = BGDoptimizer(weight[p][q], bias[p], learning_rate,
                                                x[q], grad_ReLU(y[p]), loss_[p])
    return layers


def back_propagation(layers):
    n = len(layers) - 1
    #a = softmax(last_layer(layers)['out_'])
    #loss_ = cross_entropy(a, y_)
    #loss_ = grad_softmax(a, y_)
    for i in range(n, -1, -1):
        if i == n:
            layer = layers[n]
            #layer['loss_'] = loss_
        else:
            layer1 = layers[i]
            layer2 = layers[i + 1]
            weight2 = layer2['weight_']
            loss_2 = layer2['loss_']
            loss_1 = []
            for q in range(len(weight2[0])):
                loss_ = 0
                for p in range(len(weight2)):
                    loss_ += loss_2[p] * weight2[p][q]
                loss_1.append(loss_)
            layer1['loss_'] = loss_1
            layers[i] = layer1


def layer_forward(x, layer):
    weight = layer['weight_']
    bias = layer['bias_']
    activation = layer['activation_']
    layer['in_'] = x
    layer['out_'] = single_forward(x, weight, bias, activation)
    return layer


def single_forward(x, weight, bias, activation):
    x_ = []
    for i in range(len(weight)):
        t = np.dot(x, weight[i]) + bias[i]
        if activation == 'sigmoid':
            x_.append(sigmoid(t))
        else:
            x_.append(ReLU(t))
    return x_


def get_layer(in_=2, out_=4, activation='ReLU'):
    bias = []
    for i in range(out_):
        bias.append(get_bias())
    weight = []
    for i in range(out_):
        weight.append(get_weight(in_))
    layer = {'weight_': weight, 'bias_': bias, 'activation_': activation}
    return layer


def forward(x, layers):
    out_ = x
    for i in range(len(layers)):
        layer = layers[i]
        layer = layer_forward(out_, layer)
        out_ = layer['out_']


def last_layer(layers):
    k = len(layers)-1
    last_layer = layers[k]
    return last_layer


def loss_cal(layers, y_):
    """
    #classification task: softmax_cross_entropy#DeepLearning

    a = softmax(last_layer(layers)['out_'])
    loss_ = grad_cross_entropy(a, y_)

    """

    #regression task:MSE
    a = last_layer(layers)['out_']
    loss_ = grad_MSEloss(a, y_)

    try:
        loss_old = last_layer(layers)['loss_']
        for i in range(len(loss_)):
            loss_old[i] += loss[i]
        last_layer(layers)['loss_'] = loss_old
    except:
        last_layer(layers)['loss_'] = loss_
    return layers


def train(batch_x, batch_y, net, learning_rate=0.1):
    total_loss = 0
    for i, x in enumerate(batch_x):
        forward(x, net)
        net = loss_cal(net, batch_y[i])
        total_loss += sum(loss(last_layer(net)['out_'], batch_y[i]))
    back_propagation(net)
    net = update(net, learning_rate)
    last_layer(net).pop('loss_')
    return net, total_loss

def predict(x, net):
    forward(x, net)
    return net

def valid_forward(batch_x, batch_y, net):
    total_loss = 0
    for i, x in enumerate(batch_x):
        forward(x, net)
        net = loss_cal(net, batch_y[i])
        total_loss += sum(loss(last_layer(net)['out_'], batch_y[i]))
    return total_loss