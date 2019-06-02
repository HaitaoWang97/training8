#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#下载目标数据集
get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxmet import gluon, image
from mxnet.gluon import utils as gutils
import os
#定义下载数据函数
def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
              'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
              'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), shal_hash=v)


# In[ ]:


#读取数据集函数
def load_data_pikachu(batch_size, edge_size=256):
    data_dir = '../data/pikachu'
    _download_pikachu(data_dir)
    train_iter = image.ImageDeIter(
        path_imgrec=os.path.join(data_dir, "train.rec"),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=True, rand_crop=1,
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), suhffle=False)
    return train_iter, val_iter
    


# In[ ]:


batch_size, edge_size = 32, 256
train_iter, _ = load_data_pikachu(batch_size, edge_size)
batch = train_iter.next()
batch.data[0].shape, batch.label[0].shape


# In[ ]:


#画出10张图片和pikachu的bounding_box
imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))) / 255
axes = d2l.show_images(imgs, 2, 5).flatten()
for ax, label in zip (axes, batch.label[0][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

