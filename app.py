import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = 'model_5_2.npy'
dir = 'test_data.csv'


def inl(y_, max_ = 4.75, min_ = 1.2 ):
    return (max_ - min_) * y_ + min_


def l(x):
    mx1, mn1 = 0.002, - 0.059000000000000004
    mx2, mn2 = 0.018000000000000002, - 0.059000000000000004
    mx3, mn3 = 32.37, 9.81
    mxy, mny =4.75,1.2
    return (x[0] - mn1) / (mx1 - mn1), (x[1] - mn2) / (mx2 - mn2), (x[2] - mn3) / (mx3 - mn3)


def process_test(dir, path):
    test_data = pd.read_csv(dir)
    prediction = []
    net = utils.load_model(path)
    for i in range(len(test_data)):
        x= [test_data['x1'].iloc[i], test_data['x2'].iloc[i], test_data['x3'].iloc[i]]
        prediction.append(inl(utils.predict(l(x), net)[0]))

    y_ = np.array(prediction)

    np.savetxt('prediction.txt', y_)

if __name__ == '__main__':

    process_test(dir = 'test_data.csv', path = 'model_5_2.npy')

