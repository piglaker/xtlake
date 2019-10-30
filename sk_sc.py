import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train_data.csv')

data = np.array(data)
train_data, valid_data = data[:500], data[500:605]

X=train_data[:,0:3]

y = train_data[:,3]

print(X, y)
scaler = StandardScaler() # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)   # 转换数据集
t1 = time.time()
# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(5, 2) hidden层2层,第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

print(X, y)

clf.fit(X, y)

t2 = time.time()

error = []

for i in range(len(X)):
    pre = clf.predict([X[i]])
    print('predict：', pre, ' ground truth: ', y[i])  # 预测某个输入对象
    error.append(abs(pre - y[i]) / y[i])

print(np.mean(error))

print('Train take : ', 60 * (t2 - t1))


