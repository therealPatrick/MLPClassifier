'''
reference from docs.scipy.org
'''
import numpy as np
import copy
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

eps = 10**-12


class NeuralNet:
    def __init__(self, layers, la, fun, lr):
        self.layers = layers
        self.la = la
        self.fun = fun
        self.lr = lr
        self.weights = np.array([np.random.rand(la[i + 1], la[i]) * 0.01 for i in range(layers - 1)])
        self.bias = np.array([np.zeros((la[i + 1], 1)) for i in range(layers - 1)])
        self.iters = 15
        self.trbatch = 200
        self.e = []

    def af(self, a, output_layer):
        if output_layer:
            a = np.maximum(a, -300)
            a = np.minimum(a, 300)
            return np.exp(a)/(np.sum(np.exp(a), axis=0))
        if self.fun == "RELU":
            return np.maximum(a, 0)
        if self.fun == "SIGMOID":
            return 1/(1 + np.exp(-a))
        if self.fun == "TANH":
            a = np.maximum(a, -300)
            a = np.minimum(a, 300)
            return np.tanh(a)
        if self.fun == "LINEAR":
            return a

    def invaf(self, o):
        if self.fun == "RELU":
            return o > 0
        if self.fun == "SIGMOID":
            return o * (1 - o)
        if self.fun == "TANH":
            return 1 - o*o
        if self.fun == "LINEAR":
            return np.ones(shape=o.shape)

    def forward(self, X):
        self.a = []
        self.z = []
        for i in range(self.layers - 1):
            if i == 0:
                self.a.append(copy.deepcopy(X))
                self.z.append(np.zeros(shape=(0, 0))) #garbage
                continue
            self.z.append(np.matmul(self.weights[i - 1], self.a[-1]) + self.bias[i - 1])
            fin = self.af(self.z[-1], False)
            self.a.append(copy.deepcopy(fin))
        self.z.append(np.matmul(self.weights[self.layers - 2], self.a[-1]) + self.bias[self.layers - 2])
        fin = self.af(self.z[-1], True)
        self.a.append(copy.deepcopy(fin))

    def lossfn(self, yhat, y):
        return -np.sum(y*np.log(yhat + eps))

    def backward(self, Y):
        self.delta = []
        for i in range(self.layers):
            self.delta.append(np.zeros(shape=(self.la[i], Y.shape[1])))
        self.delta[-1] = self.a[-1] - Y
        for i in range(len(self.delta) - 2, -1, -1):
            self.delta[i] = np.matmul(self.weights[i].T, self.delta[i + 1]) * self.invaf(self.a[i])
            self.invw[i] += np.matmul(self.delta[i + 1], self.a[i].T)
            self.invb[i] += np.matmul(self.delta[i + 1], np.ones(shape=(Y.shape[1], 1)))


    def fit(self, X, Y):
        #one-hot encoding
        iy = []
        for i in range(X.shape[0]):
            iy.append(np.zeros(shape=(self.la[-1])))
        iy = np.array(iy)
        sz = X.shape[0]
        for i in range(sz):
            iy[i, Y[i]] = 1
        for p in range(self.iters):
            for k in range(0, sz, self.trbatch):
                examples = self.trbatch
                if self.trbatch > sz - k:
                    continue
                tr_x = X[k: k+examples].T
                tr_y = iy[k: k+examples].T
                # print(tr_x.shape, tr_y.shape)
                self.invw = []
                self.invb = []
                for k in range(self.layers - 1):
                    self.invw.append(np.zeros(shape=(la[k + 1], la[k])))
                    self.invb.append(np.zeros(shape=(la[k + 1], 1)))
                self.invw = np.array(self.invw)
                self.invb = np.array(self.invb)
                self.forward(tr_x)
                self.backward(tr_y)
                self.invw /= examples
                self.invb /= examples

                self.weights = self.weights - self.lr*self.invw
                self.bias = self.bias - self.lr*self.invb

            self.forward(X.T)
            self.e.append(self.lossfn(self.a[-1], iy.T)/X.shape[0])
            print(self.e[-1])

    def predict(self, X):
        X = X.reshape((X.shape[0], 1))
        self.forward(X)
        return self.a[self.layers-1].T[0]

    def score(self, X, Y):
        cnt = 0
        for i in range(X.shape[0]):
            pred = np.argmax(self.predict(X[i]))
            if pred == Y[i]:
                cnt += 1
        return cnt/X.shape[0]


if __name__ == '__main__':
    (x_tr, y_tr), (x_ts, y_ts) = mnist.load_data()
    scale = StandardScaler()
    x_tr = x_tr.reshape(x_tr.shape[0], -1)
    scale.fit(x_tr)
    x_tr = scale.transform(x_tr)
    x_ts = x_ts.reshape(x_ts.shape[0], -1)
    x_ts = scale.transform(x_ts)
    print("Enter number of layers: ")
    l = int(input())
    print("Enter number of neurons (layer wise): ")
    la = list(map(int, input().split()))
    print("Enter activation function: ")
    fun = input()
    print("Enter learning rate: ")
    lr = float(input())

    net = NeuralNet(l, la, fun, lr)
    net.fit(x_tr, y_tr)
    print(net.score(x_tr, y_tr))
    print(net.score(x_ts, y_ts))
    plt.plot(net.e)
    plt.show()




