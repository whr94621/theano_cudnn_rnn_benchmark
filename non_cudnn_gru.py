import theano
from theano import tensor as T
from dl4mt_layers.layers import param_init_gru, gru_layer
from dl4mt_layers.utils import init_tparams, itemlist
import numpy as np
from collections import OrderedDict

import time

import argparse
def create_parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--n_batch', type=int, default=500)

    return parser

parser = create_parsers()
args = parser.parse_args()

seq_len = args.seq_len
batch_size = args.batch_size
input_size = args.input_size
hidden_size = args.hidden_size
n_batch = args.n_batch

print(args)

mode_with_gpu = theano.compile.mode.get_default_mode().including(
    'gpuarray'
).excluding('gpu')

X = T.tensor3('X', dtype='float32')
Y = T.tensor3('Y', dtype='float32')

x_val = np.random.randn(seq_len, batch_size, input_size).astype('float32')
y_val = np.random.randn(seq_len, batch_size, hidden_size).astype('float32')

params = OrderedDict()

params = param_init_gru(params=params,
                        nin=input_size,
                        dim=hidden_size,
                        prefix='gru')

tparams = init_tparams(params=params)

proj = gru_layer(tparams=tparams, state_below=X, prefix='gru')

outp = proj[0]

cost = T.mean((Y - outp) ** 2)
grads = T.grad(cost, itemlist(tparams=tparams))

print("Non-CuDNN unidirectional GRU")

#---------------Compile time test-----------------#
print("COMPILE TIME TEST")

start = time.time()
gru = theano.function(inputs=[],
                            outputs=[outp],
                            mode=mode_with_gpu,
                      givens={X: x_val}
                      )



print('Compile foward only: {0:.4f}'.format(time.time() - start))

gru_grad = theano.function(inputs=[],
                                 outputs=grads,
                                 mode=mode_with_gpu,
                           givens={X: x_val, Y: y_val}
                           )

print('Compile forward+backward: {0:.4f}'.format(time.time() - start))

#----------Runtime test-------------------#
print("RUNTIME TEST")

n_samples = n_batch * batch_size

start = time.time()
for i in xrange(n_batch):
    gru()

    if (i + 1) % 50 == 0:
        print('Process {0} batches'.format(i + 1))

end = time.time()

print("forward")
print("--- {0} samples in {1:.4f} seconds ({2:.4f} samples/s, {3:.4f} s/sample)---".format(
    n_samples,
    end - start,
    n_samples / (end - start),
    (end - start) / n_samples
))

start = time.time()
for i in xrange(n_batch):
    gru()
    gru_grad()
    if (i + 1) % 50 == 0:
        print('Process {0} batches'.format(i + 1))

end = time.time()

print("forward + backward")
print("--- {0} samples in {1:.4f} seconds ({2:.4f} samples/s, {3:.4f} s/sample)---".format(
    n_samples,
    end - start,
    n_samples / (end - start),
    (end - start) / n_samples
))