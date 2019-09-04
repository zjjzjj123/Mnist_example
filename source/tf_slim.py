'''
use tf.slim struct CNN  is ok  and easy
'''

#可以混合使用啊

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import tensorflow as tf
import os

#使用slim构建卷积层
def slim_inference(): #推理过程
    mnist_path = 'E:/code_now/Mnist_example/MNIST_data' #MNIST_data路径
    mnist = input_data.read_data_sets(mnist_path,one_hot=True) #实例化 读取

    #定义占位符
    x = tf.placeholder(tf.float32,[None,784])   #输入占位符
    y_hat = tf.placeholder(tf.float32,[None,10]) #输出占位符
    x_input = tf.reshape(x,[-1,28,28,1]) #将[None,784]reshape成[-1,28,28,1] 以便送到网络中进行训练
    with slim.arg_scope([slim.conv2d], #类似slim的作用范围 只要是slim.conv2d都能进行一下初始化
                        weights_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01), #权重W初始化
                        biases_initializer=tf.constant_initializer(0.1), #偏置初始化
                        padding='SAME', #卷积时 是否考虑边界
                        activation_fn=tf.nn.relu): #卷积层后的激活函数
    #x_input:输入图片[batch,w,h,channels] 32:输出通道 [5,5]:卷积核大小 [1,1]:步长 scope:层名称
        net = slim.conv2d(x_input,32,[5,5],stride=[1,1],scope='conv1')
    #net:卷积层输出 [2,2]:卷积核大小 [2,2]:步长 scope：层名称
        net = slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool1')
        net = slim.conv2d(net,64,[5,5],stride=[1,1],scope='conv2')
        net = slim.max_pool2d(net,[2,2],stride=[2,2],scope='pool2')

        net = slim.flatten(net) #将卷积层输出的多维度矩阵，扁平化成全连接层能够使用的[x,w]的维度
        net = slim.fully_connected(net,1024,scope='fc1')
        net = slim.dropout(net,keep_prob=0.5)
        y_conv = slim.fully_connected(net,10,activation_fn=None,scope='fc2') #最后输出[None,10] 也就是对应10个类别的预测结果
    #######下面的屏蔽代码和上面with...下面的功能一样只是一个使用scope统一配置相应参数，一个分别配置了参数################
    # conv1 = slim.conv2d(x_input,32,[5,5],stride=[1,1],weights_initializer=tf.truncated_normal_initializer(0.0,0.01)
    #                     ,biases_initializer=tf.constant_initializer(0.1),padding='SAME',
    #                     activation_fn=tf.nn.relu) #28*28*32
    # conv1 = slim.max_pool2d(conv1,kernel_size=[2,2],stride=2,padding='SAME')  #14*14*32
    #
    # #conv2
    # conv2 = slim.conv2d(conv1,64,[5,5],stride=[1,1],weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
    #                     biases_initializer=tf.constant_initializer(0.1),padding='SAME',
    #                     activation_fn=tf.nn.relu) #14*14*64
    # conv2 = slim.max_pool2d(conv2,kernel_size=[2,2],stride=2,padding='SAME') ##7*7*64
    #
    # fc_in = slim.flatten(conv2)  #准备输入到fc中的数据 进行扁平化
    # #fc1
    # fc1 = slim.fully_connected(fc_in,1024,activation_fn=slim.nn.relu,weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
    #                            biases_initializer=tf.constant_initializer(0.1))
    # fc1 = slim.dropout(fc1,0.5)
    # #fc2
    # out = slim.fully_connected(fc1,10,activation_fn=None)

# use tf optimizer function
    cross_entrop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_hat,logits=y_conv)) #loss的定义 使用交叉熵
    # cross_entrop = tf.reduce_mean(-y_hat * tf.log(net))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entrop) #loss的优化
    correct = tf.equal(tf.argmax(y_hat,1),tf.argmax(y_conv,1)) #对每个样本判断是否预测正确
    acc = tf.reduce_mean(tf.cast(correct,tf.float32)) #求出准确率
    init = tf.global_variables_initializer() #初始化tf变量
    # sess = tf.InteractiveSession()
    saver = tf.train.Saver() #实例化，保存模型
    with tf.Session() as sess:
        sess.run(init)
        for i in range(500): #训练500次
            data = mnist.train.next_batch(50) #每个batch的图片样本是50
            train_acc = acc.eval(feed_dict={x:data[0],y_hat:data[1]})
            if i%50 == 0:  #没50次输出一次准确率
                print("step %d,training acc %g"%(i,train_acc))
            sess.run(train_step,feed_dict={x:data[0],y_hat:data[1]}) #运行训练
        # cross_entrop = slim.losses.softmax_cross_entropy(logits=out,onehot_labels=y_hat)
        if not tf.gfile.Exists('model/'):
            tf.gfile.MakeDirs('model/')
            saver.save(sess, 'model/my_model.ckpt')  # 保存模型
        print('test acc %g:'%acc.eval(feed_dict = {x:mnist.test.images,y_hat:mnist.test.labels})) #训练完之后进行测试

def slim_example():
    x1 = tf.cast(tf.ones(shape=[3,5,5,1]),tf.float32)

    w = tf.cast(tf.fill([2, 2, 1, 2], 1),tf.float32)

    # print("rank is", tf.rank(x1))

    y1 = tf.nn.conv2d(x1, w, strides=[1, 1, 1, 1], padding='SAME')
    y1_re = tf.reshape(y1,[-1,50])
    y1_s = slim.flatten(y1)
    # y1 = tf.nn.relu(y1)
    # y1 = tf.nn.max_pool(y1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    # y1 = tf.nn.batch_normalization()

    y2 = slim.conv2d(x1, 2, [2, 2],stride=[2,2], weights_initializer=tf.ones_initializer, padding='SAME',
                     activation_fn=tf.nn.relu)
    y2 = slim.max_pool2d(y2,kernel_size=[2,2],stride=2,padding='SAME')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        y1_value, y2_value, x1_value = sess.run([y1, y2, x1])
        y1_re , y1_s = sess.run([y1_re , y1_s])
        print(y1_re.shape,y1_s.shape)

        print("shapes are", y1_value.shape, y2_value.shape)

        print(y1_value == y2_value)

        print(y1_value)

    print(y2_value)


if __name__ == '__main__':
    # slim_example()
    print('start')
    # restore_interence()
    slim_inference()
    print('end')