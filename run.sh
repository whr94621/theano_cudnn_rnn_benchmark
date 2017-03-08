#!/usr/bin/env bash
export THEANO_FLAGS=device=cuda,floatX=float32,optimizer=fast_run

#python2 ./cudnn_bi_rnn.py
#python2 ./cudnn_rnn_cell.py
#python2 ./non_cudnn_gru.py
#python2 ./non_cudnn_bi_gru.py
#python2 ./non_cudnn_non_gemm_fusion_gru.py