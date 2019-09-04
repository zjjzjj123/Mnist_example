from tensorflow.examples.tutorials.mnist import input_data
import cv2
import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim


#就算是用已经训练好的模型也应该构建模型  只是不用训练了
#因为使用的是slim训练的模型 因此也是用slim构架网络 将参数直接加在到网络中
def restore_interence():
    mnist = input_data.read_data_sets('E:/code_now/Mnist_example/MNIST_data',one_hot=True)
    x = tf.placeholder(tf.float32,[None,784])
    y_hat = tf.placeholder(tf.float32,[None,10])
    x_input = tf.reshape(x,[-1,28,28,1])
    #struct model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1),
                            biases_initializer=tf.constant_initializer(0.1),
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(x_input, 32, kernel_size=[5, 5], stride=[1, 1], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=[2, 2], scope='pool1')
            net = slim.conv2d(net, 64, kernel_size=[5, 5], stride=[1, 1], padding='SAME', scope='conv2')
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=[2, 2], padding='SAME', scope='pool2')

            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024, scope='fc1')
            net = slim.dropout(net, 0.5)
            pred = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')
        saver = tf.train.Saver()
        saver.restore(sess,'E:/code_now/Mnist_example/source/model/my_model.ckpt')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=pred))
        correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(pred, 1))
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        print(acc.eval(feed_dict={y_hat:mnist.test.labels,x:mnist.test.images}))

def read_data():
    mnist = input_data.read_data_sets('E:/code_now/Mnist_example/MNIST_data',one_hot=True)
    print(mnist.train.images.shape) #训练图片的数量(55000,784) #mnist的图片格式是[None,784],28*28
    image1 = mnist.train.images[0,:] #读取第一张图片
    label1 = mnist.train.labels[0,:] #读取第一张图片的标签
    print('labels:',label1)
    image1 = np.reshape(image1,[28,28]) #由于mnist是[None,784]->[28,28]进行显示，训练时也是
    cv2.imshow('image1',image1)
    cv2.waitKey(1000)

if __name__ == '__main__':
    # read_data()
    restore_interence()