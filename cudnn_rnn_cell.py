from cudnn_layers.rnn_cell import UnrollGRU
from theano import tensor as T
import theano
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

X = T.tensor3('X', dtype=theano.config.floatX)
Y = T.tensor3('Y', dtype='float32')
h0 = T.matrix('h0', dtype=theano.config.floatX)


x_val = np.random.randn(seq_len, batch_size, input_size).astype('float32')
h0_val = np.random.randn(batch_size, hidden_size).astype('float32')
y_val = np.random.randn(seq_len, batch_size, hidden_size).astype('float32')

unroll_gru = UnrollGRU(input_size=input_size, hidden_size=hidden_size)


proj = unroll_gru(X, h0)
outp = proj[0]
cost = T.mean((Y - outp) ** 2)
grads = T.grad(cost, unroll_gru.gru_cell.params)

print("CuDNN unidirectional unrolled GRU cell")

#---------------Compile time test-----------------#
print("COMPILE TIME TEST")
start = time.time()

gru = theano.function(inputs=[],
                      outputs=[proj[0]],
                      mode=mode_with_gpu,
                      givens={X: x_val, h0: h0_val})

print("Compile forward: {0:.4f}s".format(time.time() - start))


gru_grad = theano.function(inputs=[],
                                 outputs=grads,
                                 mode=mode_with_gpu,
                           givens={X: x_val, h0: h0_val, Y: y_val}
                           )

print('Compile forward + backward: {0:.4f}'.format(time.time() - start))

#----------Runtime test-------------------#
print("RUNTIME TEST")

n_samples = n_batch * batch_size
start = time.time()
for i in xrange(n_batch):
    gru()
    if (i + 1) % 50 == 0:
        print('Process {0} batches'.format(i + 1))

end = time.time()

print("Forward")
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

print("Forward")
print("--- {0} samples in {1:.4f} seconds ({2:.4f} samples/s, {3:.4f} s/sample)---".format(
    n_samples,
    end - start,
    n_samples / (end - start),
    (end - start) / n_samples
))