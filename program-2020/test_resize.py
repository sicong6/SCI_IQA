# -*- coding:UTF-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy

# image_raw_data = tf.gfile.GFile(‘D:/path/to/picture/8.jpg’,’rb’).read()   #加载原始图像
path = 'E:/研究生学习/vacation_ii/LIAM/cim1.bmp'
im = Image.open(path)
width, height = im.size
# img_resize = numpy.asarray(im, dtype=numpy.uint8)
# image_l=img_resize*(1./255)
net = tf.image.resize_images(im, [width*2, height*2], method=2)  #ResizeMethod.BICUBIC  2

with tf.Session() as sess:
	imag = sess.run(net)   #[250. 250. 250.]

imag = imag.astype(int)    #[250 250 250]
print(imag[0][0])

plt.imshow(imag)
plt.show()



# image_l = tf.decode_raw(features["image_l"], tf.uint8)
# image_l=tf.cast(image_l,tf.float32)*(1./255)-0.5



# path = 'E:/研究生学习/vacation_ii/LIAM/cim1.bmp'
# im = Image.open(path)
# width, height = im.size
# img_resize = numpy.asarray(im, dtype=numpy.uint8)
# image_l=img_resize*(1./255)
# net = tf.image.resize_images(image_l, [width*2, height*2], method=2)  #ResizeMethod.BICUBIC  2
# with tf.Session() as sess:
# 	imag = sess.run(net)
# print(img_resize[0][0])
# plt.imshow(imag)
# plt.show()