# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def get_data(train_feature="train.csv", test_feature="test.csv", train_label="./data/df_id_train.csv", test_id="./data/df_id_test.csv"):
    train_feature = pd.read_csv(train_feature, encoding="gbk")
    test_feature = pd.read_csv(test_feature, encoding="gbk")
    train_label = pd.read_csv(train_label, header=None)
    test_label = pd.read_csv(test_id, header=None)

    column_name = ["f" + str(i) if i != 0 else "uid" for i in range(train_feature.shape[1])]
    train_feature.columns = column_name
    test_feature.columns = column_name
    train_label.columns = ["uid", "label"]
    test_label.columns = ["uid"]

    train = train_label.merge(train_feature, on="uid", how="left")
    test = test_label.merge(test_feature, on="uid", how="left")
    train_x = train.iloc[:, 2:]
    train_y = train.iloc[:, 1]
    test_x = test.iloc[:, 1:]
    test_uid = test.iloc[:, 0]

    return train_x, train_y, test_x, test_uid


def scale_data(train, test=None):
    std_scaler = StandardScaler()
    std_train = std_scaler.fit_transform(train)
    if test is not None:
        std_test = std_scaler.transform(test)
        return std_train, std_test, std_scaler
    return std_train, None, std_scaler


n_input = 203
n_hidden1 = 500
n_hidden2 = 100
n_hidden3 = 50
n_output = 2
n_epoch = 5  # 5
keep_prob = 0.65
batch_size = 20
learning_rate = 0.0005  # 0.0005
seed = 420


def dnn_network(keep_prob, is_training=True):
    keep = tf.placeholder(tf.float32)
    w_initializer = tf.truncated_normal_initializer(stddev=0.01)
    b_initializer = tf.constant_initializer(0.01)
    with tf.variable_scope("input"):
        X = tf.placeholder(tf.float32, name="input_x")
        y = tf.placeholder(tf.int64, name="input_y")
        y = tf.to_float(y)
    with tf.variable_scope("hidden1"):
        w1 = tf.get_variable("w1", shape=[n_input, n_hidden1], dtype=tf.float32, initializer=w_initializer)
        b1 = tf.get_variable("b1", shape=[n_hidden1], dtype=tf.float32, initializer=b_initializer)
        h1 = tf.nn.relu(tf.matmul(X, w1) + b1)
        if is_training:
            h1 = tf.nn.dropout(h1, keep_prob=keep)
    with tf.variable_scope("hidden2"):
        w2 = tf.get_variable("w2", shape=[n_hidden1, n_hidden2], dtype=tf.float32, initializer=w_initializer)
        b2 = tf.get_variable("b2", shape=[n_hidden2], dtype=tf.float32, initializer=b_initializer)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        if is_training:
            h2 = tf.nn.dropout(h2, keep_prob=keep)
    with tf.variable_scope("hidden3"):
        w3 = tf.get_variable("w3", shape=[n_hidden2, n_hidden3], dtype=tf.float32, initializer=w_initializer)
        b3 = tf.get_variable("b3", shape=[n_hidden3], dtype=tf.float32, initializer=b_initializer)
        h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)
        if is_training:
            h3 = tf.nn.dropout(h3, keep_prob=keep)
    with tf.variable_scope("output"):
        w4 = tf.get_variable("w4", shape=[n_hidden3, n_output], dtype=tf.float32, initializer=w_initializer)
        b4 = tf.get_variable("b4", shape=[n_output], dtype=tf.float32, initializer=b_initializer)
        output = tf.matmul(h3, w4) + b4

    with tf.variable_scope("loss"):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output, name="cross_entropy")
        # cross_entropy = -(y * tf.log(output) + (1 - y) * tf.log(1 - output))
        loss = tf.reduce_sum(cross_entropy, name="cross_entropy_mean")

    out_prob = tf.arg_max(tf.nn.sigmoid(output), dimension=1)
    hold_dict = {"keep_prob": keep, "X": X, "y": y}
    if not is_training:
        return None, loss, hold_dict, out_prob

    global_step = tf.Variable(0.0, trainable=False, name="global_step")
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    tf.summary.scalar("loss", loss)

    return train_op, loss, hold_dict, out_prob


if __name__ == "__main__":
    train_x, train_y, test_x, test_id = get_data()
    train_y = train_y[:, None]
    train_x_std, test_x_std, _ = scale_data(train_x, test_x)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=seed)

    tf.set_random_seed(seed)

    with tf.Graph().as_default():
        with tf.name_scope("train"):
            with tf.variable_scope("dnn", reuse=None):
                train_op, train_loss, train_holder, train_res = dnn_network(keep_prob=keep_prob, is_training=True)
        with tf.name_scope("test"):
            with tf.variable_scope("dnn", reuse=True):
                _, test_loss, test_holder, test_res = dnn_network(keep_prob=keep_prob, is_training=False)

        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter("./tensorflow_log", graph=sess.graph)
            sess.run(init)
            saver.save(sess, save_path="./tensorflow_model/security_dnn_model.ckpt")

            # # 交叉验证结果
            # for k, (train_index, test_index) in enumerate(k_fold.split(train_x_std)):
            #     x_train, x_test = train_x_std[train_index, :], train_x_std[test_index, :]
            #     y_train, y_test = train_y[train_index], train_y[test_index]
            #
            #     # 恢复初始模型
            #     saver.restore(sess, save_path="./tensorflow_model/security_dnn_model.ckpt")
            #     # 统计数据初始化
            #     _loss = 0
            #     _sample_count = 0
            #     for _ in range(n_epoch):
            #         for i in range(x_train.shape[0] // batch_size):
            #             batch_x, batch_y = x_train[i * batch_size:(i + 1) * batch_size, :], y_train[i * batch_size:(i + 1) * batch_size]
            #             _, tn_l = sess.run([train_op, train_loss], feed_dict={train_holder["keep_prob"]: keep_prob, train_holder["X"]: batch_x, train_holder["y"]: batch_y})
            #             _loss += tn_l
            #             _sample_count += batch_size
            #             # print("Step {}, logistic loss: {}".format(_sample_count, _loss / _sample_count))
            #
            #     # 计算训练数据和交叉验证数据的评估分数
            #     _train_loss, _train_res = sess.run([test_loss, test_res], feed_dict={test_holder["keep_prob"]: keep_prob, test_holder["X"]: x_train, test_holder["y"]: y_train})
            #     print("Fold {} - Total train logistic loss: {}, f1 score: {}".format(k + 1, _train_loss / len(x_train), f1_score(y_train.ravel(), _train_res)))
            #     _cv_loss, _cv_res = sess.run([test_loss, test_res], feed_dict={test_holder["keep_prob"]: keep_prob, test_holder["X"]: x_test, test_holder["y"]: y_test})
            #     print("Fold {} - Total cv logistic loss: {}, f1 score: {}".format(k + 1, _cv_loss / len(x_test), f1_score(y_test.ravel(), _cv_res)))

            # 训练模型
            _score= 0.0
            for k in range(10):
                # 恢复初始模型
                saver.restore(sess, save_path="./tensorflow_model/security_dnn_model.ckpt")

                _loss = 0
                _sample_count = 0
                for _ in range(n_epoch):
                    for i in range(train_x_std.shape[0] // batch_size):
                        batch_x, batch_y = train_x_std[i * batch_size:(i + 1) * batch_size, :], train_y[i * batch_size:(i + 1) * batch_size]
                        _, _trn_loss = sess.run([train_op, train_loss], feed_dict={train_holder["keep_prob"]: keep_prob, train_holder["X"]: batch_x, train_holder["y"]: batch_y})
                        _loss += _trn_loss
                        _sample_count += batch_size
                        # print("Step {}, logistic loss: {}".format(_sample_count, _loss / _sample_count))

                _train_loss, _train_res = sess.run([test_loss, test_res], feed_dict={test_holder["keep_prob"]: keep_prob, test_holder["X"]: train_x_std, test_holder["y"]: train_y})
                _f1_score = f1_score(train_y.ravel(), _train_res)
                print("Time {} - Total train logistic loss: {}, f1 score: {}".format(k + 1, _train_loss / len(train_x_std), _f1_score))