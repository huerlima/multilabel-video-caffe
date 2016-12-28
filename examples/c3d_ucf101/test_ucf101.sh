#!/usr/bin/env bash

# get the last model (latest)
LASTMODEL=$(ls -1t examples/c3d_ucf101/c3d_ucf101_iter_*.caffemodel | head -n 1)
echo "[Info] The caffemodel to be used: ${LASTMODEL}"

# model architecture
MODELDEF=examples/c3d_ucf101/c3d_ucf101_test.prototxt
BATCHSIZE=$(egrep batch_size ${MODELDEF} | awk '{print $2}')
NUMTESTEXAMPLES=$(wc -l $(egrep source ${MODELDEF} | awk '{print $2}' | tr -d '"') | awk '{print $1}')
let NUMITERS=(${NUMTESTEXAMPLES}+${BATCHSIZE}-1)/${BATCHSIZE}

echo "[Info] Tested for ${NUMITERS} iterations (${NUMTESTEXAMPLES}/${BATCHSIZE})"

if [ -z "${LASTMODEL}" ]; then
  echo "[Error] Can not find the model. Check the caffemodel name."
else
  build/tools/caffe \
    test \
  --model=${MODELDEF} \
  --weights=${LASTMODEL} \
  --iterations=${NUMITERS} \
  --gpu=0 \
  2>&1 | tee examples/c3d_ucf101/c3d_ucf101_test.log
fi
