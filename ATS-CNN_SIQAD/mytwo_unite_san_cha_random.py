# -*- coding:utf-8 -*-
#coding=gbk
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import SIQAD_ready_unite_random

init_learning_rate = 0.0001  # 0.0005
epoch_num_per_decay = 1
decay_rate_lr = 0.95  # 0.97
seed = 15
EPSILON = 0.000001
keep_prob = 0.5
multiplier = 1

# Prohibit display of operating equipment, etc.
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


def CBAM(input, reduction):
    """
    @Convolutional Block Attention Module
    """

    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # spatial attention
    y_mean = tf.reduce_mean(input, axis=3, keepdims=True)  # (B, W, H, 1)
    y_max = tf.reduce_max(input, axis=3, keepdims=True)  # (B, W, H, 1)
    y = tf.concat([y_mean, y_max], axis=-1)     # (B, W, H, 2)
    y = tf.layers.conv2d(y, 1, 3, padding='same', activation=tf.nn.sigmoid)    # (B, W, H, 1)
    y = tf.multiply(input, y)  # (B, W, H, C)

    return y
def CBAM1(input, reduction):
    """
    @Convolutional Block Attention Module
    """

    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # channel attention
    x_mean = tf.reduce_mean(input, axis=(1, 2), keepdims=True)   # (B, 1, 1, C)
    x_mean = tf.layers.conv2d(x_mean, channel // reduction, 1, activation=tf.nn.relu, name='CA1')  # (B, 1, 1, C // r)
    x_mean = tf.layers.conv2d(x_mean, channel, 1, name='CA2')   # (B, 1, 1, C)

    x_max = tf.reduce_max(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.layers.conv2d(x_max, channel // reduction, 1, activation=tf.nn.relu, name='CA1', reuse=True)
    # (B, 1, 1, C // r)
    x_max = tf.layers.conv2d(x_max, channel, 1, name='CA2', reuse=True)  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)   # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(input, x)   # (B, W, H, C)

    return x
def CBAM2(input, reduction):
    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # channel attention
    x_mean = tf.reduce_mean(input, axis=(1, 2), keepdims=True)   # (B, 1, 1, C)
    x_mean = tf.layers.conv2d(x_mean, channel // reduction, 1, activation=tf.nn.relu, name='CA3')  # (B, 1, 1, C // r)
    x_mean = tf.layers.conv2d(x_mean, channel, 1, name='CA4')   # (B, 1, 1, C)

    x_max = tf.reduce_max(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.layers.conv2d(x_max, channel // reduction, 1, activation=tf.nn.relu, name='CA3', reuse=True)
    # (B, 1, 1, C // r)
    x_max = tf.layers.conv2d(x_max, channel, 1, name='CA4', reuse=True)  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)   # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(input, x)   # (B, W, H, C)

    return x
def CBAM3(input, reduction):
    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # channel attention
    x_mean = tf.reduce_mean(input, axis=(1, 2), keepdims=True)   # (B, 1, 1, C)
    x_mean = tf.layers.conv2d(x_mean, channel // reduction, 1, activation=tf.nn.relu, name='CA5')  # (B, 1, 1, C // r)
    x_mean = tf.layers.conv2d(x_mean, channel, 1, name='CA6')   # (B, 1, 1, C)

    x_max = tf.reduce_max(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.layers.conv2d(x_max, channel // reduction, 1, activation=tf.nn.relu, name='CA5', reuse=True)
    # (B, 1, 1, C // r)
    x_max = tf.layers.conv2d(x_max, channel, 1, name='CA6', reuse=True)  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)   # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(input, x)   # (B, W, H, C)

    return x


def Xception(net, dem):
    net1 = layers.conv2d(net, dem/2, [1, 1])
    net2 = layers.conv2d(net, dem/2, [1, 1])
    net2 = layers.conv2d(net2, dem, [3, 3])
    net3 = layers.conv2d(net, dem/4, [1, 1])
    net3 = layers.conv2d(net3, dem/2, [3, 3])
    net3 = layers.conv2d(net3, dem / 2, [3, 3])
    net = tf.concat((net1, net2, net3), axis=3)
    
    return net


def Xception_CBAM(net, dem):
    net1 = layers.conv2d(net, dem/2, [1, 1])
    net2 = layers.conv2d(net, dem/2, [1, 1])
    net2 = layers.conv2d(net2, dem, [3, 3])
    net3 = layers.conv2d(net, dem/4, [1, 1])
    net3 = layers.conv2d(net3, dem/2, [3, 3])
    net3 = layers.conv2d(net3, dem / 2, [3, 3])

    net1 = CBAM(net1, 2)
    net2 = CBAM(net2, 2)
    net3 = CBAM(net3, 2)

    net = tf.concat((net1, net2, net3), axis=3)
    
    return net
def CBAM_Xception(net, dem):
    net1 = CBAM(net, 2)
    net2 = CBAM(net, 2)
    net3 = CBAM(net, 2)

    net1 = layers.conv2d(net1, dem/2, [1, 1])
    net2 = layers.conv2d(net2, dem/2, [1, 1])
    net2 = layers.conv2d(net2, dem, [3, 3])
    net3 = layers.conv2d(net3, dem/4, [1, 1])
    net3 = layers.conv2d(net3, dem/2, [3, 3])
    net3 = layers.conv2d(net3, dem / 2, [3, 3])

    net = tf.concat((net1, net2, net3), axis=3)
    
    return net


def NonLocalBlock(input_x, out_channels, sub_sample=True, scope='NonLocalBlock'):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()

    g = layers.conv2d(input_x, out_channels, [1,1], stride=1)
    if sub_sample:
        g = tf.nn.max_pool(g, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    phi = layers.conv2d(input_x, out_channels, [1,1], stride=1)
    if sub_sample:
        phi = tf.nn.max_pool(phi, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    theta = layers.conv2d(input_x, out_channels, [1,1], stride=1)

    g_x = tf.reshape(g, [batchsize,out_channels, -1])
    g_x = tf.transpose(g_x, [0,2,1])

    theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
    theta_x = tf.transpose(theta_x, [0,2,1])
    phi_x = tf.reshape(phi, [batchsize, out_channels, -1])

    f = tf.matmul(theta_x, phi_x)
    f_softmax = tf.nn.softmax(f, -1)
    y = tf.matmul(f_softmax, g_x)
    y = tf.reshape(y, [batchsize, height, width, out_channels])
    w_y = layers.conv2d(y, in_channels, [1,1], stride=1)
    z = input_x + w_y
    return z


def ACB_layer(input, output_channel):
    net_squ = layers.conv2d(input, output_channel, [5, 5], activation_fn=None)
    net_hor = layers.conv2d(input, output_channel, [5, 1], activation_fn=None)
    net_ver = layers.conv2d(input, output_channel, [1, 5], activation_fn=None)
    result  = net_squ + net_hor + net_ver
    net = tf.nn.relu(result)

    return net


def ex_feature_sim_901(input_patch_img):
    net = layers.conv2d(input_patch_img, 16, [3, 3])  # strides默认�?，激活函数默认为relu
    net = layers.conv2d(net, 16, [3, 3])
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化窗口�?，步长为2
    net1 = net

    net = layers.conv2d(net, 16 * multiplier, [3, 3])
    net = layers.separable_conv2d(net, None, [3, 3], depth_multiplier=1)
    net = CBAM1(net, 2)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    net = layers.conv2d(net, 32 * multiplier, [3, 3])
    net = layers.separable_conv2d(net, None, [3, 3], depth_multiplier=1)
    net = CBAM2(net, 2)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    net3 = net

    net = layers.conv2d(net, 64 * multiplier, [3, 3])
    net = layers.separable_conv2d(net, None, [3, 3], depth_multiplier=1)
    net = CBAM3(net, 2)
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    net = layers.conv2d(net, 128 * multiplier, [3, 3])
    net_m = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    net_a = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    net = tf.concat((net_m, net_a), axis=3)
    net5 = net

    feature = tf.reshape(net, [-1, 128*2 * multiplier])
    return feature


def ex_feature_fan_valid(input_patch_img):
    # net = tf.image.resize_images(input_patch_img, [64, 64], method=2)
    # net_t = layers.conv2d(net, 4 * multiplier, [3, 3], stride=[2, 2])  # Nm
    net_t = layers.conv2d(input_patch_img, 4 * multiplier, [3, 3], stride=[1, 1])  # Nm

    net_t = Xception_CBAM(net_t, 4)
    net = layers.conv2d(net_t, 32 * multiplier, [3, 3], padding='VALID')
    net = layers.conv2d(net, 32 * multiplier, [3, 3])
    net = layers.conv2d(net, 32 * multiplier, [3, 3], padding='VALID')
    net = Xception_CBAM(net, 32)  

    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    net = layers.conv2d(net, 64 * multiplier, [3, 3], padding='VALID')
    net = layers.conv2d(net, 64 * multiplier, [3, 3])
    net = layers.conv2d(net, 64 * multiplier, [3, 3], padding='VALID')

    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    net = layers.conv2d(net, 128 * multiplier, [3, 3], padding='VALID')
    net = layers.conv2d(net, 128 * multiplier, [3, 3])
    net_fine = layers.conv2d(net, 128 * multiplier, [3, 3], padding='VALID')

    feature = tf.reshape(net_fine, [-1, 128 * multiplier])
    return feature


def fc_function(feature, hiden_units):
    net_qua = layers.fully_connected(feature, hiden_units)
    output = layers.fully_connected(net_qua, 1, activation_fn=None)

    return output


def fc_con(feature, hiden_units):
    net_qua = layers.fully_connected(feature, hiden_units)
    output = layers.fully_connected(net_qua, 2, activation_fn=None)

    return output


def loss_smooth(label, score):
    sigma = 1.0
    diff = label - score
    regression_diff = tf.abs(diff)
    loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma),
        0.5 * sigma * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma)
    loss = tf.reduce_mean(loss, 0)

    return loss


def cnnet(input_patch_img, label, is_training):
    with arg_scope([layers.conv2d], padding='SAME',
                   normalizer_fn=layers.batch_norm, normalizer_params={"is_training": is_training}):
        input_patch_img = tf.cast(input_patch_img, dtype=tf.float32)
        input_t = tf.expand_dims(input_patch_img[:, :, :, 0], -1)
        input_p = tf.expand_dims(input_patch_img[:, :, :, 1], -1)
        label = tf.reshape(label, [-1, 1])

        feature_p = ex_feature_sim_901(input_p)  # (-1,128*4)
        feature_t = ex_feature_fan_valid(input_t)
        feature_con = tf.concat((feature_p, feature_t), axis=1)

        out_fc_p = fc_function(feature_p, 1024)
        out_fc_t = fc_function(feature_t, 1024)
        out_fc_con = fc_con(feature_con, 1024)  # 128,2
        out_fc_con = tf.nn.softmax(out_fc_con, axis=1)
        out_fc_con_p = tf.slice(out_fc_con, [0, 0], [-1, 1])
        out_fc_con_t = tf.slice(out_fc_con, [0, 1], [-1, 1])
        out_p = out_fc_p * out_fc_con_p
        out_t = out_fc_t * out_fc_con_t

        output = out_p * 0.5 + out_t * 0.5
        output = tf.reshape(output, [-1, 1])

        reg = layers.apply_regularization(layers.l2_regularizer(0.2), tf.trainable_variables())
        loss = tf.losses.mean_squared_error(label, output) + reg

    score = output

    return score, loss  # output,loss


# Optimizer of network
def op_train(loss, global_step, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_op


def live_traindata():
    # Input images and labels.
    filenames = [os.path.join(SIQAD_ready_unite_random.DATA_DIR, 'image_' + str(i) + '.tfrecords')
                 for i in ORDER[0:SIQAD_ready_unite_random.TRAIN_DATA_NUM]]#遍历i经前式得的每个值组成此list
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return filenames


def live_testdata():
    # Input images and labels.
    filenames = [os.path.join(SIQAD_ready_unite_random.DATA_DIR, 'image_' + str(i) + '.tfrecords')
                 for i in ORDER[SIQAD_ready_unite_random.TRAIN_DATA_NUM: SIQAD_ready_unite_random.TRAIN_DATA_NUM + SIQAD_ready_unite_random.TEST_DATA_NUM]]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return filenames


def evaluate(type):
    graph = tf.Graph()
    with graph.as_default() as g_assessment:
        num_patch_eva = 0
        if type == 'test':
            filenames = live_testdata()
            for i in range(SIQAD_ready_unite_random.TEST_DATA_NUM):
                num_patch_eva += SIQAD_ready_unite_random.DISNUM_PER_IMAGE * 128
                # num_patch_eva += SIQAD_ready_unite_random.NUM_PATCHES_PER_IMAGE[ORDER[SIQAD_ready_unite_random.TRAIN_DATA_NUM + i]] * SIQAD_ready_unite_random.DISNUM_PER_IMAGE
        else :
            filenames = live_traindata()
            for i in range(SIQAD_ready_unite_random.TRAIN_DATA_NUM):
                num_patch_eva += SIQAD_ready_unite_random.DISNUM_PER_IMAGE * 128
                # num_patch_eva += SIQAD_ready_unite_random.NUM_PATCHES_PER_IMAGE[ORDER[i]] * SIQAD_ready_unite_random.DISNUM_PER_IMAGE
        patch_img, label = SIQAD_ready_unite_random.distored_input(filenames, seed, type="test")  # float32
        scores, loss_n = cnnet(patch_img, label, is_training=False)
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        if type == 'test':
            checkpoint_file = os.path.join(SIQAD_ready_unite_random.TEMP_MODEL, 'temp_model.ckpt')
        else:
            # checkpoint_file = os.path.join(SIQAD_ready_unite_random.TEMP_bestMODEL+ "model"+ str(seed)+"/", 'best_model.ckpt')
            checkpoint_file = os.path.join(SIQAD_ready_unite_random.TEMP_MODEL, 'temp_model.ckpt')
        saver.restore(sess, checkpoint_file)
        score_set = []
        label_set = []
        loss_set = []
        step = 0
        num_iter = num_patch_eva // SIQAD_ready_unite_random.BATCH_SIZE_TEST  # num_pic_eva
        # compute the scores of each image
        while step < num_iter:
            loss_eva, scores_eva, labels_eva = sess.run([loss_n, scores, label])
            loss_set.append(loss_eva)
            step += 1
            labels_eva_mean = np.mean(labels_eva)
            scores_eva_mean = np.mean(scores_eva)
            score_set.append(scores_eva_mean)
            label_set.append(labels_eva_mean)
        score_set = np.reshape(np.asarray(score_set), (-1,))
        label_set = np.reshape(np.asarray(label_set), (-1,))
        loss_set = np.reshape(np.asarray(loss_set), (-1,))
        # Compute evaluation metric.
        mean = loss_set.mean()
        srocc = stats.spearmanr(score_set, label_set)[0]
        krocc = stats.stats.kendalltau(score_set, label_set)[0]
        plcc = stats.pearsonr(score_set, label_set)[0]
        rmse = np.sqrt(((score_set - label_set) ** 2).mean())
        mse = ((score_set - label_set) ** 2).mean()
        print("%s: MEAN: %.4f\t SROCC: %.4f\t KROCC: %.4f\t PLCC: %.4f\t RMSE: %.4f\t MSE: %.4f"
              % (type, mean, srocc, krocc, plcc, rmse, mse))
        return mean, srocc, plcc


def train():
    # init = tf.global_variables_initializer()
    # The numbers of the drawing test
    plt_lr = []
    plt_loss = []
    plt_srocc = []
    plt_plcc = []
    plt_train = []
    plt_step = 0
    best_srocc = -1
    best_plcc = 0
    graph = tf.Graph()
    with graph.as_default():
        all_num_patch = 0
        for i in range(SIQAD_ready_unite_random.TRAIN_DATA_NUM):
            all_num_patch += SIQAD_ready_unite_random.DISNUM_PER_IMAGE * 128# * 2  # 128/64=2
            # all_num_patch += SIQAD_ready_unite_random.NUM_PATCHES_PER_IMAGE[ORDER[i]] * SIQAD_ready_unite_random.DISNUM_PER_IMAGE
        max_step = int(SIQAD_ready_unite_random.NUM_EPOCH * all_num_patch / SIQAD_ready_unite_random.BATCH_SIZE)
        print(max_step)  # Print the total number of times 64patch is input to the network
        iters_per_epoch = all_num_patch//SIQAD_ready_unite_random.BATCH_SIZE
        print(iters_per_epoch)

        global_step = tf.Variable(0, trainable=False)  # tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step=global_step,
                                                   decay_steps=iters_per_epoch * epoch_num_per_decay,
                                                   decay_rate=decay_rate_lr, staircase=True)
        # #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
        # exponential_decay函数则可以通过staircase(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数)

        add_global = global_step.assign_add(1)  # �?更新Variable对象global_step，记录数据流图运行次�?

        filenames_train = live_traindata()
        patch_img, label = SIQAD_ready_unite_random.distored_input(filenames_train, seed, type="train")
        scores, loss_n = cnnet(patch_img, label, is_training=True)
        train_op = op_train(loss_n, global_step, learning_rate)    
        saver = tf.train.Saver()
        
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(max_step):
            try:
                loss_net, op, step_run, L_r = sess.run([loss_n, train_op, global_step, learning_rate])
                plt_lr.append(L_r)
                plt_loss.append(loss_net)
            except tf.errors.OutOfRangeError:
                break
            if step % (3*iters_per_epoch) == 0 or (step+1) == max_step:
                checkpoint_file = os.path.join(SIQAD_ready_unite_random.TEMP_MODEL, 'temp_model.ckpt')
                saver.save(sess, checkpoint_file)
                print('Epoch %d (Step %d): loss_net = %.3f' % (step / iters_per_epoch, step, loss_net))
                loss_mean, srocc_test, plcc = evaluate('test')
                plt_step += 1
                plt_srocc.append(srocc_test)
                plt_plcc.append(plcc)
                if srocc_test > best_srocc:
                    best_srocc = srocc_test
                    best_plcc = plcc
                    best_epoch = step // iters_per_epoch
                    print('best epoch %d with min loss %.3f' % (best_epoch, loss_mean))
                    checkpoint_file = os.path.join(SIQAD_ready_unite_random.TEMP_bestMODEL + "model" + str(seed)+"/", 'best_model.ckpt')
                    saver.save(sess, checkpoint_file)
            if step % (5 * iters_per_epoch) == 0 or (step + 1) == max_step:
                train_loss, train_srocc, train_plcc = evaluate("train")
                plt_train.append(train_srocc)
                print('Epoch %d (Step %d): train_loss = %.3f' % (step / iters_per_epoch, step, train_loss))
        print(best_srocc, best_plcc, best_epoch)
        print(step_run)

    plt.figure(1)
    plt.plot(range(step_run), plt_lr, 'r-')
    plt.savefig('learn_rate.jpg')
    plt.figure(2)
    plt.plot(range(step_run), plt_loss, 'r-')
    plt.savefig('loss.jpg')
    plt.figure(3)
    plt.plot(range(plt_step), plt_srocc, 'r-')
    plt.plot(range(plt_step), plt_plcc, 'b-')
    plt.savefig('plcc.jpg')
    plt.figure(4)
    plt.plot(plt_train, 'r')
    # plt.plot(b, 'bo')
    plt.savefig('train_result.jpg')
    plt.show()

    return best_srocc, best_plcc, best_epoch





if __name__ == '__main__':
    cmd_srocc = []
    cmd_plcc = []
    cmd_step = []

    # ORDER1 = [2, 6, 17, 19, 8, 11, 13, 16, 4, 18, 9, 3, 7, 5, 1, 14, 12, 10, 0, 15]  # XCXC
    # ORDER2 = [9, 6, 12, 17, 11, 7, 14, 1, 18, 3, 10, 8, 0, 15, 2, 16, 4, 5, 19, 13]  # 0.0001
    # ORDER3 = [4, 18, 2, 16, 9, 1, 3, 15, 0, 10, 12, 8, 5, 17, 19, 7, 6, 13, 11, 14]
    # ORDER4 = [5, 19, 9, 13, 6, 2, 4, 16, 17, 3, 14, 18, 1, 8, 11, 7, 12, 15, 0, 10]
    # ORDER5 = [5, 6, 1, 15, 13, 9, 4, 19, 12, 3, 8, 16, 11, 2, 14, 10, 17, 0, 7, 18]
    # ORDER6 = [18, 10, 14, 8, 0, 5, 4, 17, 12, 16, 6, 9, 1, 13, 11, 15, 3, 19, 7, 2]
    # ORDER7 = [13, 9, 17, 2, 14, 3, 1, 8, 15, 12, 10, 18, 7, 4, 16, 5, 6, 11, 0, 19]
    # ORDER8 = [3, 15, 1, 8, 2, 12, 10, 14, 18, 5, 9, 13, 16, 4, 17, 6, 7, 0, 19, 11]
    # ORDER9 = [1, 18, 15, 8, 14, 17, 6, 16, 7, 2, 11, 5, 4, 9, 12, 19, 3, 0, 10, 13]
    # ORDER0 = [19, 3, 18, 0, 11, 9, 7, 16, 17, 10, 8, 2, 13, 4, 15, 6, 12, 1, 14, 5]
    # ORDER_list = [ORDER0,ORDER1,ORDER2,ORDER3,ORDER4,ORDER5,ORDER6,ORDER7,ORDER8,ORDER9]

    for i in range(10):
        ORDER = np.random.permutation(SIQAD_ready_unite_random.TRAIN_DATA_NUM + SIQAD_ready_unite_random.TEST_DATA_NUM)  # The original image is randomly arranged
        # ORDER = ORDER_list[i]  # 2
        print('0.000data order: %s' % ORDER)
        print(SIQAD_ready_unite_random.DATA_DIR, init_learning_rate, decay_rate_lr)
        a = time.time()
        best_srocc, best_plcc, best_epoch = train()
        cmd_plcc.append(best_plcc)
        cmd_srocc.append(best_srocc)
        cmd_step.append(best_epoch)
        b = time.time()
        c = (b - a) / 60
        print(c)
    print(cmd_srocc)
