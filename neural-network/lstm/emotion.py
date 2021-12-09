import tensorflow as tf

import numpy as np

from random import randint

import time

import re

from os import listdir

from os.path import isfile, join

import matplotlib.pyplot as plt


def getTrainBatch():

    labels = []

    arr = np.zeros([batchSize, maxSeqLength])

    for i in range(batchSize):

        if (i % 2) == 0:

            num = randint(1, 11499)

            labels.append([1, 0])

        else:

            num = randint(13499, 24999)

            labels.append([0, 1])

        arr[i] = ids[num - 1:num]

    return arr, labels


def getTestBatch():

    labels = []

    arr = np.zeros([batchSize, maxSeqLength])

    for i in range(batchSize):

        num = randint(11499, 13499)

        if num <= 12499:

            labels.append([1, 0])

        else:

            labels.append([0, 1])

        arr[i] = ids[num - 1:num]

    return arr, labels


if __name__ == "__main__":

    maxSeqLength = 250

    batchSize = 24  # 批处理大小
    numDimensions = 250

    lstmUnits = 64  # LSTM的单元个数

    numClasses = 2  # 分类类别

    iterations = 50000  # 训练次数

    wordsList = np.load(r'.\data\wordsList.npy')
    print('Loaded the word list!')
    wordsList = wordsList.tolist()  #

    wordsList = [word.decode('UTF-8') for word in wordsList]  # 不然回报word not in vocab错误

    wordVectors = np.load(r'.\data\wordVectors.npy')
    print('Loaded the word vectors!')
    print(len(wordsList))
    print(wordVectors.shape)

    ids = np.load(r'.\data\idsMatrix.npy')

    tf.reset_default_graph()

    input_data = tf.placeholder(tf.int32, shape=[batchSize, maxSeqLength])  # 占位符，必不可少

    labels = tf.placeholder(tf.int32, shape=[batchSize, numClasses])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)

    data = tf.nn.embedding_lookup(wordVectors, input_data)

    data = tf.cast(data, tf.float32)  # 由于版本的问题，这一步必不可少，将x的数据格式转化成dtype，有的版本可以不写

    '''LSTM网络的构建'''

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)

    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))

    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

    value = tf.transpose(value, [1, 0, 2])  # value的输出为[batchsize,length,hidden_size]

    last = tf.gather(value, int(value.get_shape()[0]) - 1)

    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))  # 计算交叉熵

    optimizer = tf.train.AdamOptimizer().minimize(loss)  # 随机梯度下降最小化loss

    with tf.compat.v1.Session() as sess:

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        for i in range(iterations):

            # Next Batch of reviews

            nextBatch, nextBatchLabels = getTrainBatch()

            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            if (i + 1) % 1000 == 0 and i != 0:
                loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})

                accuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})

                print("iteration {}/{}...".format(i + 1, iterations),

                      "loss {}...".format(loss_),

                      "accuracy {}...".format(accuracy_))

            if (i+1) % 10000 == 0 and i != 0:
                save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i + 1)
                print("saved to %s" % save_path)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch()
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
writer = tf.summary.FileWriter('./tensorboard/test1/', sess.graph)
sess.run()



