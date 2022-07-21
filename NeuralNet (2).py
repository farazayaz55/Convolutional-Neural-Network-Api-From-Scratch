#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
class Activation:
    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Activation):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), np.full_like(x, self.alpha))


class ELU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(x, self.alpha*(np.exp(x)-1))

    def derivative(self, x):
        return np.where(x > 0., np.ones_like(x), self.forward(x) + self.alpha)


class Tanh(Activation):
    def print(self):
        print(Activation)
    
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1. - np.square(np.tanh(x))


class Sigmoid(Activation):
    def forward(self, x):
        return 1./(1.+np.exp(-x))

    def derivative(self, x):
        f = self.forward(x)
        return f*(1.-f)


class SoftPlus(Activation):
    def forward(self, x):
        return np.log(1. + np.exp(x))

    def derivative(self, x):
        return 1. / (1. + np.exp(-x))


class SoftMax(Activation):
    def forward(self, x, axis=-1):
        shift_x = x - np.max(x, axis=axis, keepdims=True)   # stable softmax
        exp = np.exp(shift_x + 1e-6)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def derivative(self, x):
        return np.ones_like(x)


relu = ReLU()
leakyrelu = LeakyReLU()
elu = ELU()
tanh = Tanh()
sigmoid = Sigmoid()
softplus = SoftPlus()
softmax = SoftMax()


# In[2]:



class Loss:
    def __init__(self, loss, delta):
        self.data = loss
        self.delta = delta

    def __repr__(self):
        return str(self.data)


class LossFunction:
    def __init__(self):
        self._pred = None
        self._target = None

    def apply(self, prediction, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

    def _store_pred_target(self, prediction, target):
        p = prediction.data
        p = p if p.dtype is np.float32 else p.astype(np.float32)
        self._pred = p
        self._target = target

    def __call__(self, prediction, target):
        return self.apply(prediction, target)


class MSE(LossFunction):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        loss = np.mean(np.square(self._pred - t))/2
        return Loss(loss, self.delta)

    @property
    def delta(self):
        t = self._target if self._target.dtype is np.float32 else self._target.astype(np.float32)
        return self._pred - t


class CrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()
        self._eps = 1e-6

    def apply(self, prediction, target):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError


class SoftMaxCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        loss = - np.mean(np.sum(t * np.log(self._pred), axis=-1))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        # according to: https://deepnotes.io/softmax-crossentropy
        onehot_mask = self._target.astype(np.bool)
        grad = self._pred.copy()
        grad[onehot_mask] -= 1.
        return grad / len(grad)


class SoftMaxCrossEntropyWithLogits(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        sm = softmax(self._pred)
        loss = - np.mean(np.sum(t * np.log(sm), axis=-1))
        return Loss(loss, self.delta)

    @property
    def delta(self):
        grad = softmax(self._pred)
        onehot_mask = self._target.astype(np.bool)
        grad[onehot_mask] -= 1.
        return grad / len(grad)


class SparseSoftMaxCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        target = target.astype(np.int32) if target.dtype is not np.int32 else target
        self._store_pred_target(prediction, target)
        sm = self._pred
        log_likelihood = np.log(sm[np.arange(sm.shape[0]), target.ravel()] + self._eps)
        loss = - np.mean(log_likelihood)
        return Loss(loss, self.delta)

    @property
    def delta(self):
        grad = self._pred.copy()
        grad[np.arange(grad.shape[0]), self._target.ravel()] -= 1.
        return grad / len(grad)


class SparseSoftMaxCrossEntropyWithLogits(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        target = target.astype(np.int32) if target.dtype is not np.int32 else target
        self._store_pred_target(prediction, target)
        sm = softmax(self._pred)
        log_likelihood = np.log(sm[np.arange(sm.shape[0]), target.ravel()] + self._eps)
        loss = - np.mean(log_likelihood)
        return Loss(loss, self.delta)

    @property
    def delta(self):
        grad = softmax(self._pred)
        grad[np.arange(grad.shape[0]), self._target.ravel()] -= 1.
        return grad / len(grad)


class SigmoidCrossEntropy(CrossEntropy):
    def __init__(self):
        super().__init__()

    def apply(self, prediction, target):
        t = target if target.dtype is np.float32 else target.astype(np.float32)
        self._store_pred_target(prediction, t)
        p = self._pred
        loss = - np.mean(
            t * np.log(p + self._eps) + (1. - t) * np.log(1 - p + self._eps),
        )
        return Loss(loss, self.delta)

    @property
    def delta(self):
        t = self._target if self._target.dtype is np.float32 else self._target.astype(np.float32)
        return self._pred - t


# In[3]:



class Variable:
    def __init__(self, v):
        self.data = v
        self._error = np.empty_like(v)   # for backpropagation of the last layer
        self.info = {}

    def __repr__(self):
        return str(self.data)

    def set_error(self, error):
        assert self._error.shape == error.shape
        self._error[:] = error

    @property
    def error(self):
        return self._error

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim


# In[4]:



class Optimizer:
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self.vars = []
        self.grads = []
        for layer_p in self._params.values():
            for p_name in layer_p["vars"].keys():
                self.vars.append(layer_p["vars"][p_name])
                self.grads.append(layer_p["grads"][p_name])

    def step(self):
        raise NotImplementedError


class Vanilla(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params=params, lr=lr)

    def step(self):
        for var, grad in zip(self.vars, self.grads):
            var -= self._lr * grad


class AdaGrad(Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-06):
        super().__init__(params=params, lr=lr)
        self._eps = eps
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, v in zip(self.vars, self.grads, self._v):
            v += np.square(grad)
            var -= self._lr * grad / np.sqrt(v + self._eps)



class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-08):
        super().__init__(params=params, lr=lr)
        self._alpha = alpha
        self._eps = eps
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, v in zip(self.vars, self.grads, self._v):
            v[:] = self._alpha * v + (1. - self._alpha) * np.square(grad)
            var -= self._lr * grad / np.sqrt(v + self._eps)


class Adam(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params=params, lr=lr)
        self._betas = betas
        self._eps = eps
        self._m = [np.zeros_like(v) for v in self.vars]
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        b1, b2 = self._betas
        b1_crt, b2_crt = b1, b2
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            m[:] = b1 * m + (1. - b1) * grad
            v[:] = b2 * v + (1. - b2) * np.square(grad)
            b1_crt, b2_crt = b1_crt * b1, b2_crt * b2   # bias correction
            m_crt = m / (1. - b1_crt)
            v_crt = v / (1. - b2_crt)
            var -= self._lr * m_crt / np.sqrt(v_crt + self._eps)


# In[5]:


import numpy as np


class BaseLayer:
    def __init__(self):
        self.order = None
        self.name = None
        self._x = None
        self.data_vars = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _process_input(self, x):
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            x = Variable(x)
            x.info["new_layer_order"] = 0

        self.data_vars["in"] = x
        # x is Variable, extract _x value from x.data
        self.order = x.info["new_layer_order"]
        _x = x.data
        return _x

    def _wrap_out(self, out):
        out = Variable(out)
        out.info["new_layer_order"] = self.order + 1
        self.data_vars["out"] = out     # add to layer's data_vars
        return out

    def __call__(self, x):
        return self.forward(x)

class ParamLayer(BaseLayer):
    def __init__(self, w_shape, activation, w_initializer, b_initializer, use_bias):
        super().__init__()
        self.param_vars = {}
        self.w = np.empty(w_shape, dtype=np.float32)
        self.param_vars["w"] = self.w
        if use_bias:
            shape = [1]*len(w_shape)
            shape[-1] = w_shape[-1]     # only have bias on the last dimension
            self.b = np.empty(shape, dtype=np.float32)
            self.param_vars["b"] = self.b
        self.use_bias = use_bias

        if activation is None:
            self._a = Linear()
        elif isinstance(activation, Activation):
            self._a = activation
        else:
            raise TypeError

        if w_initializer is None:
            TruncatedNormal(0., 0.01).initialize(self.w)
        elif isinstance(w_initializer, BaseInitializer):
            w_initializer.initialize(self.w)
        else:
            raise TypeError

        if use_bias:
            if b_initializer is None:
                Constant(0.01).initialize(self.b)
            elif isinstance(b_initializer, BaseInitializer):
                b_initializer.initialize(self.b)
            else:
                raise TypeError

        self._wx_b = None
        self._activated = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

#fully connected layer
class Dense(ParamLayer):
    def __init__(self,
                 n_in,
                 n_out,
                 activation=None,
                 w_initializer=None,
                 b_initializer=None,
                 use_bias=True,
                 ):
        super().__init__(
            w_shape=(n_in, n_out),
            activation=activation,
            w_initializer=w_initializer,
            b_initializer=b_initializer,
            use_bias=use_bias)

        self._n_in = n_in
        self._n_out = n_out

    def forward(self, x):
        self._x = self._process_input(x)
        self._wx_b = self._x.dot(self.w)
        if self.use_bias:
            self._wx_b += self.b

        self._activated = self._a(self._wx_b)   # if act is None, act will be Linear
        wrapped_out = self._wrap_out(self._activated)
        return wrapped_out

    def backward(self):
        # dw, db
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)
        grads = {"w": self._x.T.dot(dz)}
        if self.use_bias:
            grads["b"] = np.sum(dz, axis=0, keepdims=True)
        # dx
        self.data_vars["in"].set_error(dz.dot(self.w.T))     # pass error to the layer before
        return grads

class Conv2D(ParamLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='valid',
                 channels_last=True,
                 activation=None,
                 w_initializer=None,
                 b_initializer=None,
                 use_bias=True,
                 ):
        self.kernel_size = get_tuple(kernel_size)
        self.strides = get_tuple(strides)
        super().__init__(
            w_shape=(in_channels,) + self.kernel_size + (out_channels,),
            activation=activation,
            w_initializer=w_initializer,
            b_initializer=b_initializer,
            use_bias=use_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding.lower()
        assert padding in ("valid", "same"), ValueError

        self.channels_last = channels_last
        self._padded = None
        self._p_tblr = None     # padded dim from top, bottom, left, right

    def forward(self, x):
        self._x = self._process_input(x)
        if not self.channels_last:  # channels_first
            # [batch, channel, height, width] => [batch, height, width, channel]
            self._x = np.transpose(self._x, (0, 2, 3, 1))
        self._padded, tmp_conved, self._p_tblr = get_padded_and_tmp_out(
            self._x, self.kernel_size, self.strides, self.out_channels, self.padding)

        # convolution
        self._wx_b = self.convolution(self._padded, self.w, tmp_conved)
        if self.use_bias:   # tied biases
            self._wx_b += self.b

        self._activated = self._a(self._wx_b)
        wrapped_out = self._wrap_out(
            self._activated if self.channels_last else self._activated.transpose((0, 3, 1, 2)))
        return wrapped_out

    def backward(self):
        # according to:
        # https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)

        # dw, db
        dw = np.empty_like(self.w)  # [c,h,w,out]
        dw = self.convolution(self._padded.transpose((3, 1, 2, 0)), dz, dw)

        grads = {"w": dw}
        if self.use_bias:   # tied biases
            grads["b"] = np.sum(dz, axis=(0, 1, 2), keepdims=True)

        # dx
        padded_dx = np.zeros_like(self._padded)    # [n, h, w, c]
        s0, s1, k0, k1 = self.strides + self.kernel_size
        t_flt = self.w.transpose((3, 1, 2, 0))  # [c, fh, hw, out] => [out, fh, fw, c]
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                padded_dx[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :] += dz[:, i, j, :].reshape((-1, self.out_channels)).dot(
                                                            t_flt.reshape((self.out_channels, -1))
                                                        ).reshape((-1, k0, k1, padded_dx.shape[-1]))
        t, b, l, r = [self._p_tblr[0], padded_dx.shape[1] - self._p_tblr[1],
                      self._p_tblr[2], padded_dx.shape[2] - self._p_tblr[3]]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])      # pass error to the layer before
        return grads

    def convolution(self, x, flt, conved):
        batch_size = x.shape[0]
        t_flt = flt.transpose((1, 2, 0, 3))  # [c,h,w,out] => [h,w,c,out]
        s0, s1, k0, k1 = self.strides + tuple(flt.shape[1:3])
        for i in range(0, conved.shape[1]):  # in each row of the convoluted feature map
            for j in range(0, conved.shape[2]):  # in each column of the convoluted feature map
                x_seg_matrix = x[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :].reshape((batch_size, -1))  # [n,h,w,c] => [n, h*w*c]
                flt_matrix = t_flt.reshape((-1, flt.shape[-1]))  # [h,w,c, out] => [h*w*c, out]
                filtered = x_seg_matrix.dot(flt_matrix)  # sum of filtered window [n, out]
                conved[:, i, j, :] = filtered
        return conved

    def fast_convolution(self, x, flt, conved):
        # according to:
        # http://fanding.xyz/2017/09/07/CNN%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%E7%9A%84Python%E5%AE%9E%E7%8E%B0III-CNN%E5%AE%9E%E7%8E%B0/

        # create patch matrix
        oh, ow, sh, sw, fh, fw = [conved.shape[1], conved.shape[2], self.strides[0],
                                  self.strides[1], flt.shape[1], flt.shape[2]]
        n, h, w, c = x.shape
        shape = (n, oh, ow, fh, fw, c)
        strides = (c * h * w, sh * w, sw, w, 1, h * w)
        strides = x.itemsize * np.array(strides)
        x_col = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
        x_col = np.ascontiguousarray(x_col)
        x_col.shape = (n * oh * ow, fh * fw * c)    # [n*oh*ow, fh*fw*c]
        self._padded_col = x_col       # padded [n,h,w,c] => [n*oh*ow, h*w*c]
        w_t = flt.transpose((1, 2, 0, 3)).reshape(-1, self.out_channels)  # => [hwc, oc]

        # IMPORTANT! as_stride function has some wired behaviours
        # which gives a not accurate result (precision issue) when performing matrix dot product.
        # I have compared the fast convolution with normal convolution and cannot explain the precision issue.
        wx = self._padded_col.dot(w_t)  # [n*oh*ow, fh*fw*c] dot [fh*fw*c, oc] => [n*oh*ow, oc]
        return wx.reshape(conved.shape)

    def fast_backward(self):
        dz = self.data_vars["out"].error
        dz *= self._a.derivative(self._wx_b)

        # dw, db
        dz_reshape = dz.reshape(-1, self.out_channels)      # => [n*oh*ow, oc]
        # self._padded_col.T~[fh*fw*c, n*oh*ow] dot [n*oh*ow, oc] => [fh*fw*c, oc]
        dw = self._padded_col.T.dot(dz_reshape).reshape(self.kernel_size[0], self.kernel_size[1], -1, self.out_channels)
        dw = dw.transpose(2, 0, 1, 3)   # => [c, fh, fw, oc]
        grads = {"w": dw}
        if self.use_bias:  # tied biases
            grads["b"] = np.sum(dz, axis=(0, 1, 2), keepdims=True)

        # dx
        padded_dx = np.zeros_like(self._padded)  # [n, h, w, c]
        s0, s1, k0, k1 = self.strides + self.kernel_size
        t_flt = self.w.transpose((3, 1, 2, 0))  # [c, fh, hw, out] => [out, fh, fw, c]
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                padded_dx[:, i * s0:i * s0 + k0, j * s1:j * s1 + k1, :] += dz[:, i, j, :].reshape(
                    (-1, self.out_channels)).dot(
                    t_flt.reshape((self.out_channels, -1))
                ).reshape((-1, k0, k1, padded_dx.shape[-1]))
        t, b, l, r = self._p_tblr[0], padded_dx.shape[1] - self._p_tblr[1], self._p_tblr[2], padded_dx.shape[2] -                      self._p_tblr[3]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])      # pass the error to the layer before
        return grads


class Pool_(BaseLayer):
    def __init__(self,
                 kernal_size=(3, 3),
                 strides=(1, 1),
                 padding="valid",
                 channels_last=True,
                 ):
        super().__init__()
        self.kernel_size = get_tuple(kernal_size)
        self.strides = get_tuple(strides)
        self.padding = padding.lower()
        assert padding in ("valid", "same"), ValueError
        self.channels_last = channels_last
        self._padded = None
        self._p_tblr = None

    def forward(self, x):
        self._x = self._process_input(x)
        if not self.channels_last:  # "channels_first":
            # [batch, channel, height, width] => [batch, height, width, channel]
            self._x = np.transpose(self._x, (0, 2, 3, 1))
        self._padded, out, self._p_tblr = get_padded_and_tmp_out(
            self._x, self.kernel_size, self.strides, self._x.shape[-1], self.padding)
        s0, s1, k0, k1 = self.strides + self.kernel_size
        for i in range(0, out.shape[1]):  # in each row of the convoluted feature map
            for j in range(0, out.shape[2]):  # in each column of the convoluted feature map
                window = self._padded[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :]  # [n,h,w,c]
                out[:, i, j, :] = self.agg_func(window)
        wrapped_out = self._wrap_out(out if self.channels_last else out.transpose((0, 3, 1, 2)))
        return wrapped_out

    def backward(self):
        raise NotImplementedError

    @staticmethod
    def agg_func(x):
        raise NotImplementedError


class MaxPool2D(Pool_):
    def __init__(self,
                 pool_size=(3, 3),
                 strides=(1, 1),
                 padding="valid",
                 channels_last=True,
                 ):
        super().__init__(
            kernal_size=pool_size,
            strides=strides,
            padding=padding,
            channels_last=channels_last,)

    @staticmethod
    def agg_func(x):
        return x.max(axis=(1, 2))

    def backward(self):
        dz = self.data_vars["out"].error
        grad = None
        s0, s1, k0, k1 = self.strides + self.kernel_size
        padded_dx = np.zeros_like(self._padded)  # [n, h, w, c]
        for i in range(dz.shape[1]):
            for j in range(dz.shape[2]):
                window = self._padded[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :]  # [n,fh,fw,c]
                window_mask = window == np.max(window, axis=(1, 2), keepdims=True)
                window_dz = dz[:, i:i+1, j:j+1, :] * window_mask.astype(np.float32)
                padded_dx[:, i*s0:i*s0+k0, j*s1:j*s1+k1, :] += window_dz
        t, b, l, r = [self._p_tblr[0], padded_dx.shape[1]-self._p_tblr[1],
                      self._p_tblr[2], padded_dx.shape[2]-self._p_tblr[3]]
        self.data_vars["in"].set_error(padded_dx[:, t:b, l:r, :])      # pass the error to the layer before
        return grad

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._x = self._process_input(x)
        out = self._x.reshape((self._x.shape[0], -1))
        wrapped_out = self._wrap_out(out)
        return wrapped_out

    def backward(self):
        dz = self.data_vars["out"].error
        grad = None
        self.data_vars["in"].set_error(dz.reshape(self._x.shape))
        return grad


def get_tuple(inputs):
    if isinstance(inputs, (tuple, list)):
        out = tuple(inputs)
    elif isinstance(inputs, int):
        out = (inputs, inputs)
    else:
        raise TypeError
    return out


def get_padded_and_tmp_out(img, kernel_size, strides, out_channels, padding):
    # according to: http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
    batch, h, w = img.shape[:3]
    (fh, fw), (sh, sw) = kernel_size, strides

    if padding == "same":
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))
        ph = int(np.max([0, (out_h - 1) * sh + fh - h]))
        pw = int(np.max([0, (out_w - 1) * sw + fw - w]))
        pt, pl = int(np.floor(ph / 2)), int(np.floor(pw / 2))
        pb, pr = ph - pt, pw - pl
    elif padding == "valid":
        out_h = int(np.ceil((h - fh + 1) / sh))
        out_w = int(np.ceil((w - fw + 1) / sw))
        pt, pb, pl, pr = 0, 0, 0, 0
    else:
        raise ValueError
    padded_img = np.pad(img, ((0, 0), (pt, pb), (pl, pr), (0, 0)), 'constant', constant_values=0.).astype(np.float32)
    tmp_out = np.zeros((batch, out_h, out_w, out_channels), dtype=np.float32)
    return padded_img, tmp_out, (pt, pb, pl, pr)


# In[6]:


import numpy as np


class BaseInitializer:
    def initialize(self, x):
        raise NotImplementedError


class RandomNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)


class RandomUniform(BaseInitializer):
    def __init__(self, low=0., high=1.):
        self._low = low
        self._high = high

    def initialize(self, x):
        x[:] = np.random.uniform(self._low, self._high, size=x.shape)


class Zeros(BaseInitializer):
    def initialize(self, x):
        x[:] = np.zeros_like(x)


class Ones(BaseInitializer):
    def initialize(self, x):
        x[:] = np.ones_like(x)


class TruncatedNormal(BaseInitializer):
    def __init__(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)
        truncated = 2*self._std + self._mean
        x[:] = np.clip(x, -truncated, truncated)


class Constant(BaseInitializer):
    def __init__(self, v):
        self._v = v

    def initialize(self, x):
        x[:] = np.full_like(x, self._v)


random_normal = RandomNormal()
random_uniform = RandomUniform()
zeros = Zeros()
ones = Ones()
truncated_normal = TruncatedNormal()


# In[7]:


import numpy as np


class Module(object):
    def __init__(self):
        self._ordered_layers = []
        self.params = {}

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss):
        assert isinstance(loss, Loss)
        # find net order
        layers = []
        for name, v in self.__dict__.items():
            if not isinstance(v, BaseLayer):
                continue
            layer = v
            layer.name = name
            layers.append((layer.order, layer))
        self._ordered_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]

        # back propagate through this order
        last_layer = self._ordered_layers[-1]
        last_layer.data_vars["out"].set_error(loss.delta)
        for layer in self._ordered_layers[::-1]:
            grads = layer.backward()
            if isinstance(layer, ParamLayer):
                for k in layer.param_vars.keys():
                    self.params[layer.name]["grads"][k][:] = grads[k]

    def save(self, path):
        saver = Saver()
        saver.save(self, path)

    def restore(self, path):
        saver = Saver()
        saver.restore(self, path)

    def sequential(self, *layers):
        assert isinstance(layers, (list, tuple))
        for i, l in enumerate(layers):
            self.__setattr__("layer_%i" % i, l)
        return SeqLayers(layers)

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        if isinstance(value, ParamLayer):
            layer = value
            self.params[key] = {
                "vars": layer.param_vars,
                "grads": {k: np.empty_like(layer.param_vars[k]) for k in layer.param_vars.keys()}
            }
        object.__setattr__(self, key, value)


class SeqLayers:
    def __init__(self, layers):
        assert isinstance(layers, (list, tuple))
        for l in layers:
            assert isinstance(l, BaseLayer)
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)


# In[8]:


class DataLoader:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.bs = batch_size
        self.p = 0
        self.bg = self.batch_generator()

    def batch_generator(self):
        while True:
            p_ = self.p + self.bs
            if p_ > len(self.x):
                self.p = 0
                continue
            if self.p == 0:
                indices = np.random.permutation(len(self.x))
                self.x[:] = self.x[indices]
                self.y[:] = self.y[indices]
            bx = self.x[self.p:p_]
            by = self.y[self.p:p_]
            self.p = p_
            yield bx, by

    def next_batch(self):
        return next(self.bg)

