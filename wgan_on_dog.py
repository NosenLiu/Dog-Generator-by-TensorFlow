#coding:utf-8
import os
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #as mnist_data
import glob
import cv2

data_path = './data'  # 数据集路径
pic_height_width = 64  # 输入、输出图像的边长（像素）
LAMBDA = 10  # 梯度填充系数
EPOCH = 10  # 训练次数
IDXS = 20     # 每轮次训练中，训练的批次数
batch_size = 64
DEPTH = 14  # 基础通道深度，用于构建CNN模型
OUTPUT_SIZE = pic_height_width


def load_data(path):
    X_train = []
    img_list = glob.glob(path + '/*.jpg')
    for img in img_list:
        _img = cv2.imread(img)
        _img = cv2.resize(_img, (pic_height_width, pic_height_width))
        X_train.append(_img)
    print('训练集图像数目：',len(X_train))
    # print(X_train[0],type(X_train[0]),X_train[0].shape)
    return np.array(X_train, dtype=np.uint8)

def normalization(input_matirx):
    input_shape = input_matirx.shape
    total_dim = 1
    for i in range(len(input_shape)):
        total_dim = total_dim*input_shape[i]
    big_vector = input_matirx.reshape(total_dim,)
    out_vector = []
    for i in range(len(big_vector)):
        out_vector.append(big_vector[i]/256)    # 0~256值归一化
    out_vector = np.array(out_vector)
    out_matrix = out_vector.reshape(input_shape)
    return out_matrix

def denormalization(input_matirx):
    input_shape = input_matirx.shape
    total_dim = 1
    for i in range(len(input_shape)):
        total_dim = total_dim*input_shape[i]
    big_vector = input_matirx.reshape(total_dim,)
    out_vector = []
    for i in range(len(big_vector)):
        out_vector.append(big_vector[i]*256)    # 0~256值还原
    out_vector = np.array(out_vector)
    out_matrix = out_vector.reshape(input_shape)
    return out_matrix

def conv2d(name, tensor,ksize, out_dim, stddev=0.01, stride=2, padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [ksize, ksize, tensor.get_shape()[-1],out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        var = tf.nn.conv2d(tensor,w,[1,stride, stride,1],padding=padding)
        b = tf.get_variable('b', [out_dim], 'float32',initializer=tf.constant_initializer(0.01))
        return tf.nn.bias_add(var, b)

def deconv2d(name, tensor, ksize, outshape, stddev=0.01, stride=2, padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [ksize, ksize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
        b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
        return tf.nn.bias_add(var, b)

def fully_connected(name,value, output_shape):
    with tf.variable_scope(name, reuse=None) as scope:
        shape = value.get_shape().as_list()
        w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        return tf.matmul(value, w) + b

def relu(name, tensor):
    return tf.nn.relu(tensor, name)

def lrelu(name,x, leak=0.2):
    return tf.maximum(x, leak * x, name=name)

def Discriminator(name,inputs,reuse):
    with tf.variable_scope(name, reuse=reuse):
        output = tf.reshape(inputs, [-1, pic_height_width, pic_height_width, inputs.shape[-1]])
        output1 = conv2d('d_conv_1', output, ksize=5, out_dim=DEPTH) #32*32
        output2 = lrelu('d_lrelu_1', output1)

        output3 = conv2d('d_conv_2', output2, ksize=5, padding="VALID",stride=1,out_dim=DEPTH) #28*28
        output4 = lrelu('d_lrelu_2', output3)

        output5 = conv2d('d_conv_3', output4, ksize=5, out_dim=2*DEPTH) #14*14
        output6 = lrelu('d_lrelu_3', output5)

        output7 = conv2d('d_conv_4', output6, ksize=5, out_dim=4*DEPTH) #7*7
        output8 = lrelu('d_lrelu_4', output7)

        output9 = conv2d('d_conv_5', output8, ksize=5, out_dim=6*DEPTH) #4*4
        output10 = lrelu('d_lrelu_5', output9)

        output11 = conv2d('d_conv_6', output10, ksize=5, out_dim=8*DEPTH) #2*2
        output12 = lrelu('d_lrelu_6', output11)

        chanel = output12.get_shape().as_list()
        output13 = tf.reshape(output12, [batch_size, chanel[1]*chanel[2]*chanel[3]])
        output0 = fully_connected('d_fc', output13, 1)
        return output0


def generator(name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        noise = tf.random_normal([batch_size, 128])#.astype('float32')

        noise = tf.reshape(noise, [batch_size, 128], 'noise')
        output = fully_connected('g_fc_1', noise, 2*2*8*DEPTH)
        output = tf.reshape(output, [batch_size, 2, 2, 8*DEPTH], 'g_conv')

        output = deconv2d('g_deconv_1', output, ksize=5, outshape=[batch_size, 4, 4, 6*DEPTH])
        output = tf.nn.relu(output)
        # output = tf.reshape(output, [batch_size, 4, 4, 6*DEPTH])

        output = deconv2d('g_deconv_2', output, ksize=5, outshape=[batch_size, 7, 7, 4* DEPTH])
        output = tf.nn.relu(output)

        output = deconv2d('g_deconv_3', output, ksize=5, outshape=[batch_size, 14, 14, 2*DEPTH])
        output = tf.nn.relu(output)

        output = deconv2d('g_deconv_4', output, ksize=5, outshape=[batch_size, 28, 28, DEPTH])
        output = tf.nn.relu(output)

        output = deconv2d('g_deconv_5', output, ksize=5, outshape=[batch_size, 32, 32, DEPTH],stride=1, padding='VALID')
        output = tf.nn.relu(output)

        output = deconv2d('g_deconv_6', output, ksize=5, outshape=[batch_size, OUTPUT_SIZE, OUTPUT_SIZE, 3])
        # output = tf.nn.relu(output)
        output = tf.nn.sigmoid(output)
        return tf.reshape(output,[-1,OUTPUT_SIZE,OUTPUT_SIZE,3])


def save_images(images, size, path):
    # 图片归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return scipy.misc.imsave(path, merge_img)


def train():
    # print  os.getcwd()
    with tf.variable_scope(tf.get_variable_scope()):
        # real_data = tf.placeholder(dtype=tf.float32, shape=[-1, OUTPUT_SIZE*OUTPUT_SIZE*3])
        # path = os.getcwd()
        # data_dir = os.getcwd() + "/train.tfrecords"#准备使用自己的数据集
        # print data_dir
        '''获得数据'''
        z = tf.placeholder(dtype=tf.float32, shape=[batch_size, 100])#build placeholder
        real_data = tf.placeholder(tf.float32, shape=[batch_size,pic_height_width,pic_height_width,3])

        with tf.variable_scope(tf.get_variable_scope()):
            fake_data = generator('gen',reuse=False)
            disc_real = Discriminator('dis_r',real_data,reuse=False)
            disc_fake = Discriminator('dis_r',fake_data,reuse=True)   
            # 上方一行，为什么命名空间同样是'dis_r',因为这是用同样的判别器模型对假数据进行分辨

#下面这三句话去掉也没有影响 TODO 我感觉有影响，下方的var_list无法判断训练哪些参数，
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        '''计算损失'''
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        # gen_cost = tf.reduce_mean(tf.square(tf.ones([batch_size])-disc_fake))
        # disc_cost = tf.reduce_mean(tf.square(tf.concat([tf.reshape(disc_fake,[batch_size]),tf.ones([batch_size])-tf.reshape(disc_real,[batch_size])],0)))


         #临时屏蔽
        alpha = tf.random_uniform(
            shape=[batch_size, 1],minval=0.,maxval=1.)
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        gradients = tf.gradients(Discriminator('dis_r',interpolates,reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += LAMBDA * gradient_penalty
        

        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            # gen_train_op = tf.train.AdamOptimizer(
            #     learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(gen_cost,var_list=g_vars)
            # disc_train_op = tf.train.AdamOptimizer(
            #     learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(disc_cost,var_list=d_vars)
            gen_train_op = tf.train.RMSPropOptimizer(
                learning_rate=1e-4,decay=0.9).minimize(gen_cost,var_list=g_vars)
            disc_train_op = tf.train.RMSPropOptimizer(
                learning_rate=1e-4,decay=0.9).minimize(disc_cost,var_list=d_vars)

        saver = tf.train.Saver()

        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        init = tf.global_variables_initializer()
        sess.run(init)

        dog_data = load_data(data_path)
        dog_data = normalization(dog_data)

        for epoch in range (1, EPOCH):
            for iters in range(IDXS):
                if(iters%4==3):
                    img = dog_data[(iters%4)*batch_size:]
                else:
                    img = dog_data[(iters%4)*batch_size:((iters+1)%4)*batch_size]
                for x in range(1):           # TODO 在对一批数据展开训练时，训练几次生成器
                    _, g_loss = sess.run([gen_train_op, gen_cost])
                for x in range(0,5):        # TODO 训练一次生成器，训练几次判别器...
                    _, d_loss = sess.run([disc_train_op, disc_cost], feed_dict={real_data: img})
                print("[%4d:%4d/%4d] d_loss: %.8f, g_loss: %.8f"%(epoch, iters, IDXS, d_loss, g_loss))

            with tf.variable_scope(tf.get_variable_scope()):
                samples = generator('gen', reuse=True)
                samples = tf.reshape(samples, shape=[batch_size,pic_height_width,pic_height_width,3])
                samples=sess.run(samples)
                samples = denormalization(samples)  # 还原0~256 RGB 通道数值
                save_images(samples, [8,8], os.getcwd()+'/img/'+'sample_%d_epoch.png' % (epoch))

            if epoch%10==9:
                checkpoint_path = os.path.join(os.getcwd(),
                                               './models/WGAN/my_wgan-gp.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
                print('*********    model saved    *********')
        coord.request_stop()
        coord.join(threads)
        sess.close()


def show_tensor_names():   # 打印各变量数据，用于从保存的模型中读取变量及操作。
    with tf.variable_scope(tf.get_variable_scope()):
        z = tf.placeholder(dtype=tf.float32, shape=[batch_size, 100])#build placeholder
        real_data = tf.placeholder(tf.float32, shape=[batch_size,pic_height_width,pic_height_width,3])
        with tf.variable_scope(tf.get_variable_scope()):
            fake_data = generator('gen',reuse=False)
            disc_real = Discriminator('dis_r',real_data,reuse=False)
            disc_fake = Discriminator('dis_r',fake_data,reuse=True)   
            # 上方一行，为什么命名空间同样是'dis_r',因为这是用同样的判别器模型对假数据进行分辨
            print('real_data',real_data)
            print('fake_data',fake_data)
            print('disc_real',disc_real)
            print('disc_fake',disc_fake)
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        print('gen_cost',gen_cost)
        print('disc_cost',disc_cost)
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        gen_train_op = tf.train.RMSPropOptimizer(
            learning_rate=1e-4,decay=0.9).minimize(gen_cost,var_list=g_vars)
        disc_train_op = tf.train.RMSPropOptimizer(
            learning_rate=1e-4,decay=0.9).minimize(disc_cost,var_list=d_vars)
        print('gen_train_op',gen_train_op)
        print('disc_train_op',disc_train_op)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in variables:
        print(v)


if __name__ == '__main__':
    # train()
    show_tensor_names()