# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from scipy import stats
import matplotlib.pyplot as plt
import SIQAD_ready32

init_learning_rate = 0.001
epoch_num_per_decay = 1
decay_rate_lr = 0.9
seed = 15



def cnnet(input_patch_img, label, is_training):
    with arg_scope([layers.conv2d], padding='SAME',
                   normalizer_fn=layers.batch_norm, normalizer_params={"is_training": is_training}):

        net = tf.image.resize_images(input_patch_img, [256, 256], method=2)
        net = layers.conv2d(net, 8, [3, 3], stride=[2, 2], scope='convd1')  # Nm

        net = layers.conv2d(net, 32, [7, 7], scope='convd2')  #128 128 32      # strides默认为1，激活函数默认为relu
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化窗口为3，步长为2

        net = layers.conv2d(net, 32, [3, 3], scope='convd3') 
        net_m = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        net_a = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        net = tf.concat((net_m, net_a), axis=3)

        net = layers.conv2d(net, 64, [3, 3], scope='convd4') 
        net_m = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        net_a = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        net = tf.concat((net_m, net_a), axis=3)

        net = layers.conv2d(net, 128, [3, 3], scope='convd5') 
        net_m = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        net_a = tf.nn.avg_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        net = tf.concat((net_m, net_a), axis=3)

        net = tf.nn.avg_pool(net, ksize=[1,8,8,1], strides=[1,8,8,1], padding="SAME")  # Global average pooling

        net = tf.reshape(net, [-1,256])
        net = layers.fully_connected(net, 1024)

        output = layers.fully_connected(net, 1, activation_fn=None)


    score = output
    # label = label*np.ones(num_patch)
    label = tf.reshape(label, [-1, 1])
    loss = tf.losses.mean_squared_error(label, output)

    return score, loss  # output,loss


# Optimizer of network
def op_train(loss, global_step, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return train_op


def live_traindata():
    # Input images and labels.
    filenames = [os.path.join(SIQAD_ready32.DATA_DIR, 'image_' + str(i) + '.tfrecords')
                 for i in ORDER[0:SIQAD_ready32.TRAIN_DATA_NUM]]#遍历i经前式得的每个值组成此list
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return filenames
def live_testdata():
    # Input images and labels.
    filenames = [os.path.join(SIQAD_ready32.DATA_DIR, 'image_' + str(i) + '.tfrecords')
                 for i in ORDER[SIQAD_ready32.TRAIN_DATA_NUM: SIQAD_ready32.TRAIN_DATA_NUM + SIQAD_ready32.TEST_DATA_NUM]]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return filenames


def evaluate(type):
    graph = tf.Graph()
    with graph.as_default() as g_assessment:
        # num_pic_eva = 0
        num_patch_eva = 0
        if type == 'test':
            filenames = live_testdata()
            for i in range(SIQAD_ready32.TEST_DATA_NUM):
                # num_pic_eva += SIQAD_ready32.DISNUM_PER_IMAGE
                num_patch_eva += SIQAD_ready32.NUM_PATCHES_PER_IMAGE[ORDER[SIQAD_ready32.TRAIN_DATA_NUM + i]] * SIQAD_ready32.DISNUM_PER_IMAGE
        else :
            filenames = live_traindata()
            for i in range(SIQAD_ready32.TRAIN_DATA_NUM):
                # num_pic_eva += SIQAD_ready32.DISNUM_PER_IMAGE
                num_patch_eva += SIQAD_ready32.NUM_PATCHES_PER_IMAGE[ORDER[i]] * SIQAD_ready32.DISNUM_PER_IMAGE
        patch_img, label = SIQAD_ready32.distored_input(filenames, seed, type="test")  # float32
        scores, loss_n = cnnet(patch_img, label, is_training=False)
        # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     SIQAD_ready32.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        if type == 'test':
            checkpoint_file = os.path.join(SIQAD_ready32.TEMP_MODEL, 'temp_model.ckpt')
        else:
            # checkpoint_file = os.path.join(SIQAD_ready32.TEMP_bestMODEL+ "model"+ str(seed)+"/", 'best_model.ckpt')
            checkpoint_file = os.path.join(SIQAD_ready32.TEMP_MODEL, 'temp_model.ckpt')
        saver.restore(sess, checkpoint_file)
        score_set = []
        label_set = []
        loss_set = []
        step = 0
        num_iter = num_patch_eva // SIQAD_ready32.BATCH_SIZE_TEST  # num_pic_eva
        #compute the scores of each image
        while step < num_iter:
            loss_eva, scores_eva, labels_eva = sess.run([loss_n, scores, label])
            score_set.append(scores_eva)
            label_set.append(labels_eva)
            loss_set.append(loss_eva)
            step += 1
        # print(len(score_set),len(score_set[0]))
        # print(len(label_set),len(label_set[0]))
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
        print("%s: MEAN: %.3f\t SROCC: %.3f\t KROCC: %.3f\t PLCC: %.3f\t RMSE: %.3f\t MSE: %.3f"
              % (type, mean, srocc, krocc, plcc, rmse, mse))
        return mean, srocc, plcc



def train():
    # init = tf.global_variables_initializer()
    #画图测试的各个数据
    plt_lr = []
    plt_loss = []
    plt_srocc = []
    plt_plcc = []
    plt_step = 0
    best_srocc = -1
    best_plcc = 0
    graph = tf.Graph()
    with graph.as_default():
        all_num_patch = 0
        for i in range(SIQAD_ready32.TRAIN_DATA_NUM):
            # num_picture += SIQAD_ready32.DISNUM_PER_IMAGE
            all_num_patch += SIQAD_ready32.NUM_PATCHES_PER_IMAGE[ORDER[i]] * SIQAD_ready32.DISNUM_PER_IMAGE
        max_step = int(SIQAD_ready32.NUM_EPOCH * all_num_patch / SIQAD_ready32.BATCH_SIZE)
        print(max_step)  # 打印patch输进网络的总次数
        iters_per_epoch = all_num_patch//SIQAD_ready32.BATCH_SIZE
        print(iters_per_epoch)

        global_step = tf.Variable(0, trainable=False)  # tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step=global_step,
                                                   decay_steps=iters_per_epoch * epoch_num_per_decay,
                                                   decay_rate=decay_rate_lr, staircase=True)
        # 学习速率每982/8轮训练后要乘以0.98
        # #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)，#
        # exponential_decay函数则可以通过staircase(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数)

        add_global = global_step.assign_add(1)  # 增1更新Variable对象global_step，记录数据流图运行次数

        filenames_train = live_traindata()
        patch_img, label = SIQAD_ready32.distored_input(filenames_train, seed, type="train")
        scores, loss_n = cnnet(patch_img, label, is_training=True)
        train_op = op_train(loss_n, global_step, learning_rate)    
        # loss_n, train_op = loss_op(label, score, global_step)
        # print(score, label)
        # loss_n = loss_fuc(label, score)
        # train_op = op_train(loss_n, global_step)
        saver = tf.train.Saver()
        
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    # with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        # sre = sess.run(loss_n)
        # print(sre.shape)
        for step in range(max_step):
            try:
                loss_net, op, step_run, L_r = sess.run([loss_n, train_op, global_step, learning_rate])
                plt_lr.append(L_r)
                plt_loss.append(loss_net)
            except tf.errors.OutOfRangeError:
                break
            if step % (1*iters_per_epoch) == 0 or (step+1) == max_step:
                print(step)
            if step % (3*iters_per_epoch) == 0 or (step+1) == max_step:
                checkpoint_file = os.path.join(SIQAD_ready32.TEMP_MODEL, 'temp_model.ckpt')
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
                    checkpoint_file = os.path.join(SIQAD_ready32.TEMP_bestMODEL + "model" + str(seed)+"/", 'best_model.ckpt')
                    saver.save(sess, checkpoint_file)
            # if step % (10 * iters_per_epoch) == 0 or (step + 1) == max_step:
                # train_loss, train_srocc, train_plcc = evaluate("train")
                # print('Epoch %d (Step %d): train_loss = %.3f' % (step / iters_per_epoch, step, train_loss))
        print(best_srocc, best_plcc, best_epoch)
        print(step_run)



    plt.figure(1)
    plt.plot(range(step_run), plt_lr, 'r-')
    plt.figure(2)
    plt.plot(range(step_run), plt_loss, 'r-')
    plt.figure(3)
    plt.plot(range(plt_step), plt_srocc, 'r-')
    plt.plot(range(plt_step), plt_plcc, 'b-')
    plt.show()





if __name__ == '__main__':
    ORDER = np.random.permutation(SIQAD_ready32.TRAIN_DATA_NUM + SIQAD_ready32.TEST_DATA_NUM)  # 原图像随机排序
    print('data order: %s' % ORDER)
    train()
