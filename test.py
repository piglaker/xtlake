import utils
import numpy as np

layer = [utils.get_layer(3,1, 'sigmoid')]
batch_x = [[1,1,1]]
batch_y = [[0.5]]
print(layer)
for i in range(100):
    layer, total_loss = utils.train(batch_x, batch_y, layer)
    print(total_loss)
    print(layer)
