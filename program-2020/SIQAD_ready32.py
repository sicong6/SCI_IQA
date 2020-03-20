# -*- coding:UTF-8 -*-

import tensorflow as tf
import numpy as np

DATA_DIR = 'E:/研究生学习/vacation_ii/mytwo/data_t_SCI_32'  # TFrecord文件位置
LOG_DIR = '../log_t'
TEMP_MODEL="../logs/temp/"
TEMP_bestMODEL="../logs/besttemp/"

TRAIN_DATA_NUM = 16
#VAL_DATA_NUM = 4
TEST_DATA_NUM = 4
DISNUM_PER_IMAGE = 49

#the paras of dataset
SHUFFLE_SIZE = 1000
NUM_EPOCH = 70
BATCH_SIZE = 32
BATCH_SIZE_TEST = 32  # 1???

#the info of the input tensor
DEPTH = 3
PATCH_SIZE = 128
NUM_PATCHES_PER_IMAGE = [25, 25, 25, 25, 25, 30, 25, 30, 25, 30, 25, 25, 16, 16, 24, 24, 24, 30, 30, 24]#不确定，依参考图片而变
#[672,682,700,742,742,700,670,732,736,716,826,728,750,688,696,810,656,754,672,768,746,684,722,674,626,612,624,624,846,626,830,604,792,630,804,648,834,666,832,618]


def parser(record):
    """
    parse the tfrecords
    """
    features = tf.parse_single_example(
        record,
        features={
            "img": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.float32),
            # "type_ref": tf.FixedLenFeature([], tf.float32)
        })
    # img = tf.decode_raw(features["img"], tf.float64)
    img = tf.decode_raw(features["img"], tf.uint8)
    # img=tf.cast(img, tf.float32)*(1./255)-0.5
    # image_l = tf.reshape(image_l, [NEW_HEIGHT, NEW_WIDTH, DEPTH])
    # img = tf.reshape(img, [HEIGHT, WIDTH, DEPTH])
    # img = tf.cast(img, tf.float32) * (1. / 255)
    label = features["label"]
    # type_ref = features["type_ref"]
    return img, label


def distored_input(filenames,seed,type="train"):
    # NUM_PATCHES = tf.convert_to_tensor([25, 25, 25, 25, 25, 30, 25, 30, 25, 30, 25, 25, 16, 16, 24, 24, 24, 30, 30, 24])
    with tf.variable_scope("input_data"):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parser)
        if type=="train":
            dataset = dataset.shuffle(buffer_size=SHUFFLE_SIZE)
            dataset = dataset.repeat(NUM_EPOCH)
            dataset = dataset.batch(BATCH_SIZE)
        else:
            dataset=dataset.batch(BATCH_SIZE_TEST)
        iterator = dataset.make_one_shot_iterator()
        img, label = iterator.get_next()
        # type_ref_int = tf.to_int32(type_ref, name='ToInt32')
        # num = NUM_PATCHES[type_ref_int[0]]
        patch_img = tf.reshape(img, [BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, DEPTH])
        return patch_img, label#, type




