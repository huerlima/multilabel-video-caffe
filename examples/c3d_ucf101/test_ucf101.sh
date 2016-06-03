#!/usr/bin/env sh

build/tools/caffe \
  test \
  --model=examples/c3d_ucf101/c3d_ucf101_test.prototxt \
  --weights=examples/c3d_ucf101/c3d_ucf101_iter_60000.caffemodel \
  --iterations=1395 \
  --gpu=0 \
  2>&1 | tee examples/c3d_ucf101/c3d_ucf101_test.log

# 41822/30=1395
