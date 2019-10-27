import utils
import math
import random

x1 = []
x2 = []
x3 = []
y = []
for i in range(1000):
    rand = random.random()
    x1.append(rand)
for i in range(1000):
    rand = random.random()
    x2.append(rand)
for i in range(1000):
    rand = random.random()
    x3.append(rand)
for i in range(1000):
    y.append(x1[i] + math.pow(1.5*x2[i], 2) + 2*x3[i] + 0.01 * random.random())
ymax = max(y)
ymin = min(y)

for i in range(1000):
    y[i] = (y[i] - ymin)/(ymax-ymin)
for i in range(1000):
    print(x1[i], x2[i], x3[i], y[i])


layer1_ = utils.get_layer(3, 8)
layer2_ = utils.get_layer(8, 8)
layer3_ = utils.get_layer(8, 4)
layer4_ = utils.get_layer(4, 1)
net = [layer1_, layer2_, layer3_, layer4_]
epoch = 10
round = 100
batch = 10
print('traning begins !')
for p in range(epoch):
    print('epoch ' + str(p) + ' begins !')
    k = 0
    for i in range(round):
        loss_total = 0
        print('epoch : ' + str(p) + 'round : ' + str(i) + ' begins !')
        batch_x = []
        batch_y = []
        for u in range(batch):
            batch_x.append([x1[u], x2[u], x3[u]])
            batch_y.append([y[u]])
        net, total_loss = utils.train(batch_x, batch_y, net, learning_rate = 0.001)
        print('loss:' + str(total_loss))
        k = k + batch

