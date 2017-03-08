import theano
from theano import tensor as T
from theano.gpuarray import dnn
from theano.gpuarray.type import gpuarray_shared_constructor
import numpy as np
import time

mode_with_gpu = theano.compile.mode.get_default_mode().including(
    'gpuarray'
).excluding('gpu')

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

X = T.tensor3('X', dtype='float32')
h0 = T.tensor3('h0', dtype='float32')
Y = T.tensor3('Y', dtype='float32')

x_val = np.random.randn(seq_len, batch_size, input_size).astype('float32')
y_val = np.random.randn(seq_len, batch_size, hidden_size * 2).astype('float32')
h0_val = np.random.randn(2, batch_size, hidden_size).astype('float32')


rnnb = dnn.RNNBlock(
    dtype=theano.config.floatX,
    hidden_size=hidden_size,
    num_layers=1,
    rnn_mode='gru',
    direction_mode='bidirectional'
)

psize = rnnb.get_param_size((50,input_size))

params_cudnn = gpuarray_shared_constructor(np.zeros(psize, dtype=theano.config.floatX))



outp, h = rnnb.apply(params_cudnn, X, h0)

cost = T.mean((Y - outp) ** 2)
grads = T.grad(cost, params_cudnn)

print("CuDNN bidirectional GRU")
#---------------Compile time test-----------------#
print("COMPILE TIME TEST")
t0 = time.time()

cudnn_gru = theano.function(inputs=[],
                            outputs=[outp, h],
                            mode=mode_with_gpu,
                            givens={X: x_val, h0: h0_val})

print('Compile foward only: {0:.4f}'.format(time.time() - t0))

cudnn_gru_grad = theano.function(inputs=[],
                                 outputs=grads,
                                 mode=mode_with_gpu,
                                 givens={X: x_val, h0: h0_val, Y: y_val})

print('Compile forward+backward: {0:.4f}'.format(time.time() - t0))

#----------Runtime test-------------------#
print("RUNTIME TEST")

n_samples = n_batch * batch_size
start = time.time()
for i in xrange(n_batch):
    cudnn_gru()

    if (i + 1) % 50 == 0:
        print('Process {0} batches'.format(i + 1))

end = time.time()

print("forward only")
print("--- {0} samples in {1:.4f} seconds ({2:.4f} samples/s, {3:.4f} s/sample)---".format(
    n_samples,
    end - start,
    n_samples / (end - start),
    (end - start) / n_samples
))

n_samples = n_batch * batch_size
start = time.time()
for i in xrange(n_batch):
    cudnn_gru()
    cudnn_gru_grad()
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





