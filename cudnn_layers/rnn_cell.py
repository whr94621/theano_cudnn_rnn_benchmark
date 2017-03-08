import theano
from theano import tensor as T
from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor

import numpy as np


class GRUCell(object):
    def __init__(self, input_size, hidden_size, dtype=theano.config.floatX):
        self.grub = dnn.RNNBlock(
            dtype=dtype,
            hidden_size=hidden_size,
            num_layers=1,
            rnn_mode='gru'
        )

        self.input_size = input_size
        self.hidden_size = hidden_size

        psize = self.grub.get_param_size((1, input_size))

        self.params = gpuarray_shared_constructor(np.zeros(psize, dtype=theano.config.floatX))

    def __call__(self, X, h0, params=None):
        if params is None:
            params = self.params

        y, _ = self.grub.apply(params, X, h0)

        return y.reshape([y.shape[1], y.shape[2]])


class UnrollGRU(object):
    def __init__(self, input_size, hidden_size, dtype=theano.config.floatX):
        self.gru_cell = GRUCell(input_size=input_size,
                                hidden_size=hidden_size,
                                dtype=dtype)


    def __call__(self, X, h0):

        n_steps = X.shape[0]

        def _step(x, h_prev, p_gru_cell):

            x = x[None,:,:]
            h_prev = h_prev[None,:,:]

            h = self.gru_cell(x, h_prev, p_gru_cell)

            return h

        rval, updates = theano.scan(_step,
                                    sequences=[X],
                                    outputs_info=[h0],
                                    non_sequences=[self.gru_cell.params],
                                    name='gru_scan',
                                    n_steps=n_steps,
                                    strict=True)

        return [rval]


