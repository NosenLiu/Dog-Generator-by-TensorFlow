#coding:utf-8
import os
import numpy as np
import scipy.misc
import tensorflow as tf
import glob
import cv2

data_path = './data'
pic_height_width = 64     # 输入、输出图像的边长（像素）

LAMBDA = 10    # 梯度填充系数
EPOCH = 40    # 训练次数
IDXS = 1000     # 每轮次训练中，训练的批次数
batch_size = 64   # 批大小
model_path_meta = './models/WGAN/my_wgan-gp.ckpt-199.meta'   # 模型计算图位置
model_path = './models/WGAN'              # 模型所在路径
TRAIN_TIME = 1     # 第几次开展训练，用来命名新的Rmsprop优化器，

DEPTH = 14   # 基础通道深度，用于构建CNN模型
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

def retrain():
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.import_meta_graph(model_path_meta)# 加载图结构
    saver.restore(sess,model_path_meta[:-5])   # 加载模型
    # saver.restore(sess,tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]# 得到当前图中所有变量的名称
    # TODO 下面获取计算图模型中的各种variable和operation
    real_data_get = graph.get_tensor_by_name('Placeholder_1:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    disc_real_get = graph.get_tensor_by_name('dis_r/d_fc/add:0')
    fake_data_get = graph.get_tensor_by_name('gen/Reshape:0')
    disk_fake_get = graph.get_tensor_by_name('dis_r_1/d_fc/add:0')
    print('real_data_get: ',real_data_get)
    print('fake_data_get: ',fake_data_get)
    print('disc_real_get: ',disc_real_get)
    print('disk_fake_get' ,disk_fake_get)
    gen_cost_get = graph.get_tensor_by_name('Neg:0')
    disc_cost_get = graph.get_tensor_by_name('sub:0')
    # TODO 上方获取变量，用get_tensor_by_name()函数，下方获取操作，使用get_operation_by_name()函数
    gen_train_op = graph.get_operation_by_name('RMSProp')
    disc_train_op = graph.get_operation_by_name('RMSProp_1')

    with tf.variable_scope(tf.get_variable_scope()):
        # t_vars = graph.trainable_variables()
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
    '获取图片数据'
    dog_data = load_data(data_path)
    dog_data = normalization(dog_data)

    for epoch in range (1, EPOCH):
        for iters in range(IDXS):
            if(iters%4==3):
                img = dog_data[(iters%4)*batch_size:]
            else:
                img = dog_data[(iters%4)*batch_size:((iters+1)%4)*batch_size]
            for x in range(1):           # TODO 在对一批数据展开训练时，训练几次生成器
                _, g_loss = sess.run([gen_train_op, gen_cost_get])
            for x in range (0,5):        # TODO 训练一次生成器，训练几次判别器...
                _, d_loss = sess.run([disc_train_op, disc_cost_get], feed_dict={real_data_get: img})
            print("[%4d:%4d/%4d] d_loss: %.8f, g_loss: %.8f"%(epoch, iters, IDXS, d_loss, g_loss))
        with tf.variable_scope(tf.get_variable_scope()):
            samples = fake_data_get
            samples = tf.reshape(samples, shape=[batch_size,pic_height_width,pic_height_width,3])
            samples=sess.run(samples)
            samples = denormalization(samples)  # 还原0~256 RGB 通道数值
            save_images(samples, [8,8], os.getcwd()+'/img/'+'sample_%d_epoch.png' % (epoch))

        if epoch%10==9:
            checkpoint_path = os.path.join(os.getcwd(),'./models/WGAN/my_wgan-gp_train_%d.ckpt'%(TRAIN_TIME))
            saver.save(sess, checkpoint_path, global_step=epoch)
            print('*********    model saved    *********')

    coord.request_stop()
    coord.join(threads)
    sess.close()

def retrain_with_another_optimezer():
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.import_meta_graph(model_path_meta)# 加载图结构
    saver.restore(sess,model_path_meta[:-5])   # 加载模型
    # saver.restore(sess,tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]# 得到当前图中所有变量的名称
    # print(tensor_name_list)
    real_data_get = graph.get_tensor_by_name('Placeholder_1:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    disc_real_get = graph.get_tensor_by_name('dis_r/d_fc/add:0')
    fake_data_get = graph.get_tensor_by_name('gen/Reshape:0')
    disk_fake_get = graph.get_tensor_by_name('dis_r_1/d_fc/add:0')
    print('real_data_get: ',real_data_get)
    print('fake_data_get: ',fake_data_get)
    print('disc_real_get: ',disc_real_get)
    print('disk_fake_get' ,disk_fake_get)
    gen_cost_get = graph.get_tensor_by_name('Neg:0')
    disc_cost_get = graph.get_tensor_by_name('sub:0')

    with tf.variable_scope(tf.get_variable_scope()):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
    temp_variable = set(tf.all_variables())    #
    # TODO 由于下方定义新的optimizer会生成新的variables，但是在init里面并没有初始化，所以无法访问，会报错
    # TODO 因此上方使用set()函数记录所有旧的变量，再定义好新的optimizer后，再对局部新变量进行初始化
    '''定义optimizer'''    #,name='RMS_train_%d'%(TRAIN_TIME)
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        gen_train_op = tf.train.RMSPropOptimizer(
            learning_rate=2e-5,decay=0.9,name='RMS_train_%d'%(TRAIN_TIME)).minimize(gen_cost_get,var_list=g_vars)
        disc_train_op = tf.train.RMSPropOptimizer(
            learning_rate=2e-5,decay=0.9,name='RMS_train_%d'%(TRAIN_TIME)).minimize(disc_cost_get,var_list=d_vars)
    # print('!++++++++============================ ',gen_train_op)
    # print('!++++++++============================ ',disc_train_op)
    init = tf.variables_initializer(set(tf.all_variables())-temp_variable)  # TODO 局部变量初始化
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
                _, g_loss = sess.run([gen_train_op, gen_cost_get])
            for x in range (0,5):        # TODO 训练一次生成器，训练几次判别器...
                _, d_loss = sess.run([disc_train_op, disc_cost_get], feed_dict={real_data_get: img})
            print("[%4d:%4d/%4d] d_loss: %.8f, g_loss: %.8f"%(epoch, iters, IDXS, d_loss, g_loss))
        with tf.variable_scope(tf.get_variable_scope()):
            samples = fake_data_get
            samples = tf.reshape(samples, shape=[batch_size,pic_height_width,pic_height_width,3])
            samples=sess.run(samples)
            samples = denormalization(samples)  # 还原0~256 RGB 通道数值
            save_images(samples, [8,8], os.getcwd()+'/img/'+'sample_%d_epoch.png' % (epoch))

        if epoch%10==9:
            checkpoint_path = os.path.join(os.getcwd(),'./models/WGAN/my_wgan-gp_train_%d.ckpt'%(TRAIN_TIME))
            saver.save(sess, checkpoint_path, global_step=epoch)
            print('*********    model saved    *********')

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    retrain_with_another_optimezer()
    # retrain()