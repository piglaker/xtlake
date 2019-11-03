import math
import numpy as np
import random


def fft_convolve(a,b):
    """
    没写完
    :param a:
    :param b:
    :return:
    """
    n = len(a)+len(b)-1
    N = 2**(int(np.log2(n))+1)
    A = np.fft.fft(a, N)
    B = np.fft.fft(b, N)
    return np.fft.ifft(A*B)[:n]


def normalization(x):
    """
    过时了
    :param x:
    :return:
    """
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
    return batch


def get_weight(in_):
    """
    返回random的weight，标量
    :param in_:
    :return:
    """
    return np.random.normal(-0.01, 0.01,size = in_)


def get_bias():
    """
    返回random的bias 标量
    :return:
    """
    return np.random.normal(-0.01, 0.01)


def ReLU(x):
    return max(0, x)


def sigmoid(x):
    return 1/(1+math.exp(-x))


def tanh(x):
    return (math.exp(x) - math.exp(-1 * x)) / (math.exp(x) + math.exp(-1 * x))


def linear(x):
    """
    线性激活函数的梯度
    :param x:
    :return:
    """
    return x


def grad_sig(x):
    """
    sigmoid激活函数的梯度
    :param x:
    :return:
    """
    return x*(1-x)


def grad_ReLU(x):
    """
    ReLU激活函数的梯度
    :param x:
    :return:
    """
    if x > 0:
        return 1
    else:
        return 0


def grad_linear(x):
    """
    linear激活函数的梯度
    :param x:
    :return:
    """
    return 1


def grad_tanh(x):
    """
    tanh激活函数的梯度
    :param x:
    :return:
    """
    return 1 - math.pow(tanh(x),2)


def grad_cross_entropy(a, y_):
    """
    损失函数为交叉熵时的梯度
    :param a:
    :param y_:
    :return:
    """
    return np.array(a) - np.array(y_)


def loss(y, y_):
    """
    似乎没用了，过时了
    :param y:
    :param y_:
    :return:
    """
    v = []
    for i in range(len(y)):
        v.append(math.pow((y[i]-y_[i]), 2))
    return v


def CE_loss(y, y_):
    """
    损失函数为交叉熵损失时的loss
    :param y:
    :param y_:
    :return:
    """
    val = 0
    for i in range(len(y)):
        val += - (y_[i] * math.log(y[i]))
    return val


def grad_MSEloss(a, y_):
    """
    损失函数为平方损失时的梯度
    :param a:
    :param y_:
    :return:
    """
    return np.array(a) - np.array(y_)


def softmax(x):
    """
    softmax激活函数，维度为2时退化为逻辑斯蒂回归，奖输入映射到01的概率
    :param x:
    :return:
    """
    output = np.exp(x)
    result = output / output.sum()
    return result


def cross_entropy(y, y_):
    """
    交叉熵损失函数，用于分类任务，前面接softmax激活函数
    :param y:
    :param y_:
    :return:
    """
    val = 0
    for i in range(len(y)):
        val += - (y_[i] * math.log(y[i]))
    return val


def argmax(y_):
    """
    得到输入向量中最大元素的位置
    :param y_:
    :return:
    """
    y_ = np.array(y_)
    return np.where(y_ == max(y_))


def BGDoptimizer(w, b, learning_rate, x, grad_y, loss_, regularization=0.1):
    w = (1 - learning_rate * regularization) * w - learning_rate * x * grad_y * loss_
    b = b - learning_rate * grad_y * loss_
    return w, b


def Adamoptimizer(w, b, m, v, t, learning_rate, x, grad_y, loss_, regularization=0.1):
    beta1, beta2 = 0.9, 0.999
    epislon = 1e-8
    t += 1
    g = x * grad_y * loss_
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)
    m_cat = m / (1 - beta1 ** t + epislon)
    v_cat = v / (1 - beta2 ** t + epislon)

    w = (1 - learning_rate * regularization) * w - learning_rate * m_cat / (v_cat ** 0.5 + epislon)
    b = b - learning_rate * grad_y * loss_

    return w, b, m, v, t


def update(layers, learning_rate=0.001):
    """
    更新权重，参数
    :param layers:
    :param learning_rate:
    :return:
    """
    for i in range(len(layers)):
        layer = layers[i]
        weight = layer['weight_']
        bias = layer['bias_']
        regularization = layer['regularization_']
        try:
            m, v, t = layer['m_'], layer['v_'], layer['t_']
        except:
            h, b = len(weight), len(weight[0])
            m, v, t = np.zeros((h, b)), np.zeros((h, b)), np.zeros((h, b))

        loss_ = layer['loss_']
        x = layer['in_']
        y = layer['out_']

        if layer['optimizer_'] == 'BGDoptimizer':
            for p in range(len(weight)):
                for q in range(len(weight[p])):
                    if layer['activation_'] == 'sigmoid':
                        weight[p][q], bias[p] = BGDoptimizer(weight[p][q], bias[p], learning_rate,
                                                    x[q], grad_sig(y[p]), loss_[p], regularization)
                    elif layer['activation_'] == 'ReLU':
                        weight[p][q], bias[p] = BGDoptimizer(weight[p][q], bias[p], learning_rate,
                                                    x[q], grad_ReLU(y[p]), loss_[p], regularization)
                    elif layer['activation_'] == 'linear':
                        weight[p][q], bias[p] = BGDoptimizer(weight[p][q], bias[p], learning_rate,
                                                    x[q], grad_linear(y[p]), loss_[p], regularization)
                    elif layer['activation_'] == 'tanh':
                        weight[p][q], bias[p] = BGDoptimizer(weight[p][q], bias[p],learning_rate,
                                                    x[q], grad_tanh(y[p]), loss_[p], regularization)

        elif layer['optimizer_'] == 'Adamoptimizer':
            for p in range(len(weight)):
                for q in range(len(weight[p])):
                    if layer['activation_'] == 'sigmoid':
                        weight[p][q], bias[p], m[p][q], v[p][q], t[p][q] = Adamoptimizer(weight[p][q], bias[p],
                                            m[p][q],v[p][q],t[p][q],learning_rate,x[q],grad_sig(y[p]),loss_[p],regularization)
                    elif layer['activation_'] == 'ReLU':
                        weight[p][q], bias[p], m[p][q], v[p][q], t[p][q] = Adamoptimizer(weight[p][q], bias[p],
                                            m[p][q],v[p][q],t[p][q],learning_rate,x[q],grad_ReLU(y[p]),loss_[p],regularization)
                    elif layer['activation_'] == 'linear':
                        weight[p][q], bias[p], m[p][q], v[p][q], t[p][q] = Adamoptimizer(weight[p][q], bias[p],
                                            m[p][q],v[p][q],t[p][q],learning_rate,x[q],grad_linear(y[p]),loss_[p],regularization)
                    elif layer['activation_'] == 'tanh':
                        weight[p][q], bias[p], m[p][q], v[p][q], t[p][q] = Adamoptimizer(weight[p][q], bias[p],
                                            m[p][q],v[p][q],t[p][q],learning_rate,x[q],grad_tanh(y[p]),loss_[p],regularization)

        layer['weight_'], layer['bias_'], layer['m_'], layer['v_'], layer['t_'] = weight, bias, m, v, t

    return layers


def back_propagation(layers):
    """
    把误差（梯度）反向传播
    :param layers:
    :return:
    """
    n = len(layers) - 1
    for i in range(n, -1, -1):
        if i == n:
            layer = layers[n]
        else:
            layer1 = layers[i]
            layer2 = layers[i + 1]
            weight2 = layer2['weight_']
            loss_2 = layer2['loss_']
            layer1['loss_'] = np.dot(loss_2, np.array([weight2]).T).flatten()
            layers[i] = layer1


def layer_forward(x, layer):
    """
    :param x:
    :param layer:
    :return:
    """
    layer['in_'] = x
    layer['out_'] = single_forward(x, weight=layer['weight_'], bias=layer['bias_'], activation=layer['activation_'])

    if layer['activation_'] == 'softmax':
        layer['out_'] = softmax(layer['out_'])
    return layer


def single_forward(x, weight, bias, activation):
    """
    :param x:
    :param weight:
    :param bias:
    :param activation:
    :return:
    """
    x_ = []
    for i in range(len(weight)):
        t = np.dot(x, weight[i]) + bias[i]
        if activation == 'sigmoid':
            x_.append(sigmoid(t))
        elif activation == 'ReLU':
            x_.append(ReLU(t))
        elif activation == 'linear':
            x_.append(linear(t))
        elif activation == 'tanh':
            x_.append(tanh(t))
        elif activation == 'softmax':
            x_.append(t)
    return x_


def get_layer(in_=2, out_=4, activation='ReLU', optimizer = 'BGDoptimizer', regularization = 0.00001):
    """
    生成一个layer
    :param in_:size  int数字
    :param out_: out size  int数字
    :param activation: str  激活函数
    :param optimizer: str  优化方法啊
    :param regularization: float
    :return:
    """
    bias = []
    for i in range(out_):
        bias.append(get_bias())
    weight = []
    for i in range(out_):
        weight.append(get_weight(in_))
    layer = {'weight_': weight, 'bias_': bias, 'activation_': activation, 'optimizer_':optimizer, 'regularization_':regularization}
    return layer


def forward(x, layers):
    """
    :param x:
    :param layers:
    :return:
    """
    out_ = x
    for i in range(len(layers)):
        layer = layers[i]
        layer = layer_forward(out_, layer)
        out_ = layer['out_']


def last_layer(layers):
    """
    :param layers:
    :return:
    """
    k = len(layers)-1
    last_layer = layers[k]
    return last_layer


def loss_cal(layers, y_):
    """
    :param layers:net,
    :param y_: 真实值
    :return:
    """
    a = last_layer(layers)['out_']

    if last_layer(layers)['activation_'] == ' softmax':
        loss_ = grad_MSEloss(a, y_)#cl ssification task: softmax_cross_entropy#DeepLearning
    else:
        loss_ = grad_cross_entropy(a, y_)#regression task:MSE

    try:
        loss_old = last_layer(layers)['loss_']
        for i in range(len(loss_)):
            loss_old[i] += loss[i]
        last_layer(layers)['loss_'] = loss_old
    except:
        last_layer(layers)['loss_'] = loss_
    return layers


def train(batch_x, batch_y, net, learning_rate=0.1, decay_rate = 0.99, decay_step = 50):
    """
    :param batch_x:  输入的x，一个list或者array
    :param batch_y: 输出的y，一个向量
    :param net: 模型，[layer1， layer2]
    :param learning_rate: 学习率，超参数
    :param decay_rate: 衰减率，超参数
    :param decay_step: 衰减步长，超参数
    :return:
    """
    total_loss = 0
    for i, x in enumerate(batch_x):
        forward(x, net)
        net = loss_cal(net, batch_y[i][0])
        if last_layer(net)['activation_'] == 'softmax':
            total_loss += CE_loss(last_layer(net)['out_'], batch_y[i][0])
        else:
            total_loss += sum(loss(last_layer(net)['out_'], batch_y[i]))

    back_propagation(net)

    try:
        last_layer(net)['learning_rate_'] = last_layer(net)['learning_rate_'] * math.pow(decay_rate, 1 / decay_step)
    except:
        last_layer(net)['learning_rate_'] = learning_rate

    lr = last_layer(net)['learning_rate_']

    update(net, lr)

    last_layer(net).pop('loss_')

    return net, total_loss


def predict(x, net):
    forward(x, net)
    if last_layer(net)['activation_'] == 'softmax':
        return softmax(last_layer(net)['out_'])
    else:
        return last_layer(net)['out_']


def valid_forward(batch_x, batch_y, net):
    total_loss = 0
    for i, x in enumerate(batch_x):
        forward(x, net)
        total_loss += sum(loss(last_layer(net)['out_'], batch_y[i]))
    return total_loss


def save_model(net, path = 'model.npy'):
    np.save(path, net)


def load_model(path = 'model.npy'):
    net = np.load(path)
    return net.tolist()

