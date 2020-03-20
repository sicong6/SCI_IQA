# -*- coding:UTF-8 -*-

import os
import gzip
import numpy
from scipy import stats
import PIL.Image as Image
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import tensorflow as tf

DIR = 'J:/2019-2022'#数据库的位置
DATABASE = 'SIQAD'
REF_PATH = DIR + '/' + DATABASE + '/references/'
DIS_PATH = DIR + '/' + DATABASE + '/'
MOS_PATH = DIR + '/' + DATABASE + '/SIQAD_IQA.txt'

# DATA_DIR = 'E:/研究生学习/vacation_ii/mytwo/data_t_SCI_mm'  # TFrecord文件位置
DATA_DIR = 'E:/研究生学习/vacation_ii/mytwo/data_t_SCI_32'

TRAIN_DATA_NUM = 16
#VAL_DATA_NUM = 4
TEST_DATA_NUM = 4
DISNUM_PER_IMAGE = 49
PATCH_SIZE = 128


ORDER = numpy.random.permutation(TRAIN_DATA_NUM +TEST_DATA_NUM)   #返回打乱顺序的（0到TRAIN_DATA_NUM +TEST_DATA_NUM-1）数


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_img_regroup(path, mos, width, height, gray_scale=False):
    img_pacth = []
    # img_pacth_sort = []
    mos_sort = []
    im = Image.open(path)
    width = int(width)
    height = int(height)
    img = im.resize((width, height), Image.ANTIALIAS)
    if gray_scale:
        img = img.convert('L')  # 用plt.imshow(image_l)后的显示黄乎乎的，哪有问题
    # else:
    #     img = img.convert('RGB')
    # img_gray = img.convert('L')

    img = numpy.asarray(img, dtype=numpy.uint8)
    num_height = height//PATCH_SIZE
    num_width = width//PATCH_SIZE
    for i in range(0, num_height):
        i_ac = i * PATCH_SIZE
        for j in range(0, num_width):
            j_ac = j * PATCH_SIZE
            img_cut = img[i_ac: i_ac+PATCH_SIZE, j_ac: j_ac+PATCH_SIZE]
            img_pacth.append(img_cut)
            mos_sort.append(mos)
    # img_pacth_sort = numpy.array(img_pacth)
    # img_pacth_sort = img_pacth_sort.reshape([num_width*num_height, PATCH_SIZE, PATCH_SIZE, 3])

    return img_pacth, mos_sort


def convert_to(x, y, filename,i):
    """Converts data to tfrecords.

    Args:
      :param x, y: list - [img1, img2, ...].
                    img: ndarray.
      :param name: str.
    """
    if not gfile.Exists(filename):
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(DISNUM_PER_IMAGE):
            num_patch = len(x[index])
            for j in range(num_patch):
                img = x[index][j].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img': _bytes_feature(img),
                    'label': _float_feature(y[index][j]),
                    # 'type_ref': _float_feature(z[index])
                }))
                writer.write(example.SerializeToString())
        writer.close()


def load_Image():
    text_file = open(MOS_PATH, "r")
    lines = text_file.readlines()
    text_file.close()
    print(len(lines))
    # read the image of distorted and the value of dmos
    dis_img_set = []
    mos_set = []
    Flag_set = []
    for line in lines:
        Readname = line.rstrip().split('\t', 6)  # (5.51, i01_01_1.bmp)#以‘ ’为界分割6次
        mos = Readname[4]
        name = Readname[3]
        width = Readname[5]
        height = Readname[6]
        # Flag = Readname[0]#图片内容分类

        # Convert to Examples and write the result to TFRecords.
        path = DIS_PATH + name.lower()
        # img_resize = numpy.asarray(load_img(path, gray_scale=False), dtype=numpy.uint8) ## ndarray(0,255)      数据转换成数组
        # flag = flag + 1
        img_resize, mos_regroup = load_img_regroup(path, float(mos), width, height, gray_scale=False)  # 原图->(256, 256, 3)
        # mos_regroup = load_mos_regroup(float(mos), width, height)

        dis_img_set.append(img_resize)#图像数据 (980 ** 128 128 3)
        mos_set.append(mos_regroup)#图像mos值
        # Flag_set.append(float(Flag))

    FRONT = 0
    BACK = DISNUM_PER_IMAGE
    # convert the data to tfrecord
    for i in range(TRAIN_DATA_NUM + TEST_DATA_NUM):
        img_num = [dis_img_set[j] for j in numpy.arange(FRONT,  BACK)]#将第i个图像对应的失真图像们的数据分出来
        labels = [mos_set[j] for j in numpy.arange(FRONT,  BACK)]
        # type_ref = [Flag_set[j] for j in numpy.arange(FRONT,  BACK)]

        FRONT = BACK
        BACK = BACK + DISNUM_PER_IMAGE

        if not gfile.Exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        filename = os.path.join(DATA_DIR, "image_" + str(i) + ".tfrecords")
        convert_to(img_num, labels, filename, i)





if __name__ == '__main__':
    print('data order: %s' % ORDER)
    load_Image()