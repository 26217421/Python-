import numpy as np
import pandas as pd

import activators
from functools import reduce


# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size,
                 activator):
        """
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
                                   (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        """
        前向计算
        input_array: 输入向量，维度必须等于input_size
        """
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        """
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        """
        a = self.activator.backward(self.input)
        b = np.dot(
            self.W.T, delta_array)
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        """
        使用梯度下降算法更新权重
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print('W: %s\nb:%s' % (self.W, self.b))


# 神经网络类
class Network(object):
    def __init__(self, layers):
        """
        构造函数
        """
        self.layers = []
        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                self.layers.append(
                    FullConnectedLayer(
                        layers[i], layers[i + 1],
                        activators.SigmoidActivator()
                    )
                )
            else:
                self.layers.append(
                    FullConnectedLayer(
                        layers[i], layers[i + 1],
                        activators.SigmoidActivator()
                    )
                )

    def predict(self, sample):
        """
        使用神经网络实现预测
        sample: 输入样本
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        """

        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],
                                      data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def evaluate(self, test_data):
        """
        评价函数，预测正确的个数。
        np.argmax函数返回数组的最大值的序号，实现从one-hot到数字的转换；
        """
        test_results = [(np.argmax(self.predict(y)), x)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def gradient_check(self, sample_feature, sample_label):
        """
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        """

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    # 增加一个很小的值，计算网络的误差
                    fc.W[i, j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    # 减去一个很小的值，计算网络的误差
                    fc.W[i, j] -= 2 * epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    # 计算期望的梯度值
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i, j] += epsilon
                    print('weights(%d,%d): expected - actural %.4e - %.4e' % (
                        i, j, expect_grad, fc.W_grad[i, j]))


def transpose(args):
    return list(map(
        lambda arg: list(map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg))
        , args
    ))


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        data = list(map(lambda m: 0.9 if number & m else 0.1, self.mask))
        return np.array(data).reshape(8, 1)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec[:, 0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def train_data_set():
    data = pd.read_csv('iris.data', header=None)
    x, y = data[[0, 1, 2, 3]], pd.Categorical(data[4]).codes
    data_set = [x.values.tolist()][0]
    labels = _convert_label(y.tolist())
    print(type(labels), labels)
    print(data_set)
    return labels, data_set


# 把鸢尾花分类转化成ond-hot的三维数组
def _convert_label(lables):
    new_list = []
    for item in lables:
        if item == 0:
            new_list.append([0.9, 0.1, 0.1])
        elif item == 1:
            new_list.append([0.1, 0.9, 0.1])
        elif item == 2:
            new_list.append([0.1, 0.1, 0.9])
    return new_list


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))





def test():
    print()
    labels, data_set = transpose(train_data_set())
    net = Network([4, 4, 4, 3])
    rate = 0.5
    mini_batch = 40
    epoch = 10
    for i in range(epoch):
        net.train(labels, data_set, rate, mini_batch)
        print('after epoch %d loss: %f' % (
            (i + 1),
            net.loss(labels[-1], net.predict(data_set[-1]))
        ))
        rate /= 2
    test_samples = [[6.2, 3.4, 5.4, 2.3]]
    a, samples = transpose((labels, test_samples))
    print(net.predict(samples[0]))
    print(net.predict(data_set[0]))
    print(labels[0])
    # net.gradient_check(data_set[0], labels[0])
