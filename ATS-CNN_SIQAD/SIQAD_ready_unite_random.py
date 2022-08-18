# -*- coding:UTF-8 -*-

import tensorflow as tf
import numpy as np

DATA_DIR = 'F:/myfouraaaaaaaaa/data_SCI_unite_32_gm_random'  # The location of the TFrecord file
LOG_DIR = '../log_t'
TEMP_MODEL="../logs/temp/"
TEMP_bestMODEL="../logs/besttemp/"

TRAIN_DATA_NUM = 16
TEST_DATA_NUM = 4
DISNUM_PER_IMAGE = 49

# the paras of dataset
SHUFFLE_SIZE = 2000
NUM_EPOCH = 100
BATCH_SIZE = 128
BATCH_SIZE_TEST = 128  # 1???

# the info of the input tensor
DEPTH = 2
PATCH_SIZE = 32
patch_num = 1#256
patch_num_half = 128
NUM_PATCHES_PER_IMAGE = [25, 25, 25, 25, 25, 30, 25, 30, 25, 30, 25, 25, 16, 16, 24, 24, 24, 30, 30, 24]  # Not sure, depends on reference image(128*128)
# NUM_PATCHES_PER_IMAGE = [441, 483, 483, 440, 506, 550, 483, 525, 460, 504, 483, 462, 361, 361, 494, 450, 456, 500, 520, 494]
# NUM_PATCHES_PER_IMAGE = [100, 110, 110, 110, 121, 132, 110, 120, 110, 120, 110, 110, 81, 81, 117, 108, 108, 120, 130, 117]
# [672,682,700,742,742,700,670,732,736,716,826,728,750,688,696,810,656,754,672,768,746,684,722,674,626,612,624,624,846,626,830,604,792,630,804,648,834,666,832,618]


# Declare the order of data in the tfrecord file
def parser(record):
    """
    parse the tfrecords
    """
    features = tf.parse_single_example(
        record,
        features={
            "img": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.float32),
        })
    img = tf.decode_raw(features["img"], tf.uint8)
    label = features["label"]

    return img, label


# Get data from tfrecord file
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
        patch_img = tf.reshape(img, [BATCH_SIZE * patch_num, PATCH_SIZE, PATCH_SIZE, DEPTH])

        return patch_img, label




