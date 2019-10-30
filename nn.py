import utils
import math
import random
import numpy as np
x1 = []
x2 = []
x3 = []
y = []
for i in range(5000):
    rand = random.random()
    x1.append(rand)
for i in range(5000):
    rand = random.random()
    x2.append(rand)
for i in range(5000):
    rand = random.random()
    x3.append(rand)
for i in range(5000):
    y.append(x1[i] + 2 * x2[i] +  0.5 * x3[i] )
ymax = max(y)
ymin = min(y)
m = np.mean(y)
data = []
for i in range(5000):
    x1[i] = (x1[i] - min(x1)) / (max(x1) - min(x1))
    x2[i] = (x2[i] - min(x2)) / (max(x2) - min(x2))
    x3[i] = (x3[i] - min(x3)) / (max(x3) - min(x3))
    y[i] = [1,0] if y[i] > 1.5 else [0,1]
    data.append([[x1[i], x2[i], x3[i]], y[i]])


for i in range(5000):
    print(x1[i], x2[i], x3[i], y[i])


net = [utils.get_layer(3, 8, activation='ReLU', optimizer='Adamoptimizer', regularization=0.01),
       utils.get_layer(8, 2, activation='softmax',optimizer='Adamoptimizer',regularization=0.01)]
epoch = 16
round = 1
batch_size = 1000

print('traning begins !')
for p in range(epoch):
    print('epoch ' + str(p) + ' begins !')
    k = 0
    for i in range(round):
        loss_total = 0
        print('epoch : ' + str(p) + 'round : ' + str(i) + ' begins !')
        batch = utils.batch(data, batch_size)
        for i in range(len(batch)):
            net, total_loss = utils.train([batch[i][0]], [[batch[i][1]]], net, learning_rate = 0.1)
        print('loss:' + str(total_loss))
        k = k + batch_size


u = 0
for i in range(5000):
    y_ = utils.predict([x1[i], x2[i], x3[i]], net)
    if np.argmax(y_) == np.argmax(y[i]):
         u += 1
    print(x1[i], x2[i], x3[i], y[i], y_, np.argmax(y_), np.argmax(y[i]), True if np.argmax(y_) == np.argmax(y[i]) else False)
print(u)

