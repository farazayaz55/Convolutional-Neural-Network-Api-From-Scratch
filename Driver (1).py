#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ipynb.fs.full.NeuralNet import * 
def accuracy(predictions, labels):
    assert predictions.shape == labels.shape
    p, l = predictions.astype(np.int32), labels.astype(np.int32)
    return np.where(p == l, 1., 0.).mean()


# In[3]:


import tensorflow as tf
import pandas as pd
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import platform
from subprocess import check_output
cp=os.getcwd()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

get_ipython().run_line_magic('matplotlib', 'inline')


img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = cp + '/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test


# Invoke the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()


print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)


class CNN(Module):
    def __init__(self):
        super().__init__()
        self.seq_layers = self.sequential(
            Conv2D(1, 6, (3, 3), (1, 1), "same", channels_last=True), #1 in 6 out 5,5 kernal 1,1 stride
            MaxPool2D(2, 2), 
            Conv2D(6, 16,(3,3), (1,1), "same", channels_last=True),  
            MaxPool2D(2, 2),  #            Flatten(),
            Dense(7 * 7 * 16, 10, )
        )

    def forward(self, x):
        o = self.seq_layers.forward(x)
        return o


cnn = CNN()
opt = Adam(cnn.params, 0.001)
loss_fn = SoftMaxCrossEntropy()
train_loader = DataLoader(x_train,y_train, batch_size=163)

for step in range(300):
    bx, by = train_loader.next_batch()
    by_ = cnn.forward(bx)
    loss = loss_fn(by_, by)
    cnn.backward(loss)
    opt.step()
    if step % 50 == 0:
        ty_ = cnn.forward(test_x)
        acc = nn.metrics.accuracy(np.argmax(ty_.data, axis=1), test_y)
        print("Step: %i | loss: %.3f | acc: %.2f" % (step, loss.data, acc))


# In[ ]:




