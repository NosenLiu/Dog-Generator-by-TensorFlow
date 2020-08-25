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
EPOCH = 10    # 训练次数
IDXS = 100     # 每轮次训练中，训练的批次数
batch_size = 64   # 批大小
model_path_meta = './models/WGAN/my_wgan-gp.ckpt-199.meta'   # 模型计算图位置
model_path = './models/WGAN'              # 模型所在路径
TRAIN_TIME = 1     # 第几次训练的名称，用来命名新的Rmsprop优化器，

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

def generate_pic():
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.import_meta_graph(model_path_meta)# 加载图结构
    # saver.restore(sess,tf.train.latest_checkpoint(model_path))
    saver.restore(sess,model_path_meta[:-5])
    gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称
    # print(tensor_name_list)
    real_data_get = gragh.get_tensor_by_name('Placeholder_1:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    disc_real_get = gragh.get_tensor_by_name('dis_r/d_fc/add:0')
    fake_data_get = gragh.get_tensor_by_name('gen/Reshape:0')
    # disr_1 = gragh.get_tensor_by_name('d_fc:0')
    print('real_data_get: ',real_data_get)
    print('disc_real_get: ',disc_real_get)

    with tf.variable_scope(tf.get_variable_scope()):
        samples = fake_data_get
        samples = tf.reshape(samples, shape=[batch_size,pic_height_width,pic_height_width,3])
        samples_out1=sess.run(samples)
        samples_out1 = denormalization(samples_out1)  # 还原0~256 RGB 通道数值
        # save_images(samples, [8,8], os.getcwd()+'/img/'+'sample_%d_epoch.png' % (epoch))
        save_images(samples_out1, [8,8], os.getcwd()+'/img/'+'out1.png')
        samples_out2=sess.run(samples)
        samples_out2 = denormalization(samples_out2)  # 还原0~256 RGB 通道数值
        save_images(samples_out2, [8,8], os.getcwd()+'/img/'+'out2.png')
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    generate_pic()