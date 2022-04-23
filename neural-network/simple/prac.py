import fc
import pandas as pd

data = pd.read_csv('p3_data.csv', header=0)


def train_data_set():
    x = data.iloc[:20, :4]
    y = data.iloc[:20, 4:]
    return y.values.tolist(), [x.values.tolist()][0]


if __name__ == '__main__':
    # 设置神经网络初始化参数，初始化神经网络,列表长度表示网络层数，每个数字表示每一层节点个数
    test_samples = data.iloc[20:, :4]
    print(test_samples)
    labels, data_set = fc.transpose(train_data_set())
    print(labels)
    net = fc.Network([4, 100, 100, 2])
    # print(net)
    net.train(labels, data_set, 0.00005, 10000)
    print("success")

    (a, samples) = fc.transpose((labels, [test_samples.values.tolist()][0]))
    for d in range(len(samples)):
        print(net.predict(samples[d]))

    # net.gradient_check(data_set[0], labels[0])

