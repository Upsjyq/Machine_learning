import gzip
import numpy as np
import struct

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]

X_train = load_data("data/MNIST/train-images-idx3-ubyte.gz") / 255.0
X_test = load_data("data/MNIST/t10k-images-idx3-ubyte.gz") / 255.0
y_train = load_data("data/MNIST/train-labels-idx1-ubyte.gz",True).reshape(-1)
y_test = load_data("data/MNIST/t10k-labels-idx1-ubyte.gz",True).reshape(-1)


from sklearn.model_selection import GridSearchCV
param_grid = {"hidden_layer_sizes": [(50,),(100,)],
                             "solver": ['adam', 'sgd', 'lbfgs'],
                             "max_iter": [200],
                             "verbose": [True],
                              "learning_rate_init":[0.001]
                             }
mlp = MLPClassifier()
grid = GridSearchCV(mlp, param_grid, refit = True, cv = 5, n_jobs = -1)
grid.fit(X_train, y_train)
print(grid.score(X_train, y_train))
print(grid.score(X_test, y_test))
y_pre = grid.predict(X_test)
count = 0
sample_size = 10000
plt.figure(figsize=(16, 6))
for i in np.random.permutation(X_test.shape[0])[:sample_size]:
    if y_pre[i]!=y_test[i]:
        count = count + 1
        plt.subplot(10, 30, count)
        plt.axhline('')
        plt.axvline('')
        plt.text(x=10, y=-3, s=y_pre[i], fontsize=18)
        plt.imshow(X_test[i].reshape(28, 28), cmap=plt.cm.Greys)
plt.show()