#1.数据 加载数据
#2.构架模型
#3.计算loss 优化模型
'''
2层conv 2 层全连接
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

def WeightVariable(shape): #均值为0 方差为0.01的正态分布作为W的初始化
    init = tf.Variable(tf.truncated_normal(shape,mean=0,stddev=0.01))
    return init
def BiasVariable(shape):
    init = tf.constant(0.1,shape=shape) #使用常数初始化
    return tf.Variable(init)
def conv2d_(x,w):
    #x:图片[batch,h,w,channels] w:卷积核[fh,fw,in,out] 前两个代表卷积核大小in代表输入通道 out代表输出通道
    #strides:步长,且strides[0]=[3] 必须是1 只有中间两个参数才决定步长 下同
    # padding='SAME'表示考虑边界 'VALD表示不考虑'
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def max_pool2d_(x):
    #x: 图片[batch,w,h,channels] ksize:中间两个代表卷积核大小 strides:代表卷积核步长
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def basic_inference():
    mnist_path = 'E:/code_now/Mnist_example/MNIST_data' #MNIST_data路径
    mnist = input_data.read_data_sets(mnist_path,one_hot=True) #实例化 读取
    x = tf.placeholder(tf.float32,[None,784]) #输入维度 None就代表batch自适应，根据输入的batch决定
    y_hat = tf.placeholder(tf.float32,[None,10]) #输出类别

    x_input = tf.reshape(x,[-1,28,28,1]) #格式为[Batch,h,w,C]

    #形式类似y =  x*w1 + b
    #conv1
    w1 = WeightVariable([5,5,1,32]) #卷积核的维度和输入输出通道的多少
    b1 = BiasVariable([32]) #b的维度于w的输出通道一致
    net = tf.nn.relu(conv2d_(x_input,w1) + b1)
    net = max_pool2d_(net)
    #conv2
    w2 = WeightVariable([5,5,32,64])
    b2 = BiasVariable([64])
    net = tf.nn.relu(conv2d_(net,w2)+b2)
    net = max_pool2d_(net)

    net = tf.reshape(net,[-1,7*7*64]) #扁平化 以便送入到全连接层
    #fc1
    w3 = WeightVariable([7*7*64,1024]) #全连接层的w维度有所变化不是卷积核
    b3 = BiasVariable([1024])
    net = tf.nn.relu(tf.matmul(net,w3)+b3)
    keep_drop = tf.placeholder(tf.float32) #drop out 参数占位符
    net = tf.nn.dropout(net,keep_prob=keep_drop)
    w4 = WeightVariable([1024,10])
    b4 = BiasVariable([10])
    y_conv = tf.nn.softmax(tf.matmul(net,w4)+b4)

    cross_entrop = tf.reduce_mean(-y_hat * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entrop)

    correct = tf.equal(tf.argmax(y_hat,1),tf.argmax(y_conv,1))
    acc = tf.reduce_mean(tf.cast(correct,tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver() #保存模型 实例化
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            data = mnist.train.next_batch(50)
            train_acc = sess.run(acc,feed_dict={x:data[0],y_hat:data[1],keep_drop:0.5})
            if i%50 == 0:
                print('step: %d  train_acc: %g'%(i,train_acc))
            sess.run(train_step,feed_dict={x:data[0],y_hat:data[1],keep_drop:0.5})
        if not tf.gfile.Exists('model/'):
                tf.gfile.MakeDirs('model/')
        saver.save(sess,'model/my_model.ckpt')
        print('test acc: %g'%acc.eval(feed_dict={x:mnist.test.images,y_hat:mnist.test.labels,keep_drop:1.0}))


if __name__ == '__main__':
    basic_inference()