# Video-Caffe: Caffe with C3D implementation and video reader

[![Build Status](https://travis-ci.org/chuckcho/video-caffe.svg?branch=master)](https://travis-ci.org/chuckcho/video-caffe)

This is 3-D Convolution (C3D) and video reader implementation in the latest Caffe (as of Feb 2016). The original [Facebook C3D implementation](https://github.com/facebook/C3D/) is branched out from Caffe on July 17, 2014 with git_id [b80fc86](https://github.com/BVLC/caffe/tree/b80fc862952ba4e068cf74acc0823785ce1cc0e9), and has not been rebased with the original Caffe hence missing out quite a few new features in the lastest Caffe. I therefore tried to pull in C3D concept and also an accompanying video reader and apply to the latest Caffe. Please reach [me](https://github.com/chuckcho) for any feedback or question. Thank you.

Check out the [original Caffe readme](README-original.md) for Caffe-specific information.

## Building video-caffe

In a nutshell, key steps to build video-caffe are:

1. `git clone git@github.com:chuckcho/video-caffe.git`
2. `cd video-caffe`
3. `mkdir build && cd build`
4. `cmake ..`
5. `make all`
6. `make install`
7. (optional) `make runtest`

## UCF-101 training demo

Follow these steps to train C3D on UCF-101.

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`.
3. Change `${video-caffe-root}/examples/c3d_ucf101/{c3d_ucf101_train_split1.txt,c3d_ucf101_test.split1.txt}` to correctly point to UCF-101 videos.
4. Modify `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt)` to your taste or HW specification. Especially `batch_size` may have to be reduced for the GPU memory. Should run fine as is with 12GB GPU memory.
5. Run training script: e.g. `cd ${video-caffe-root} && examples/c3d_ucf101/train_ucf101.sh`
6. Sit back and enjoy.

## Note

* This version of Caffe only works with Nvidia GPU (CUDA) because NdConvolution and NdPooling layers depend on CUDA.
* For building information refer to the [original Caffe readme](README-original.md) or [official Caffe installation guide](http://caffe.berkeleyvision.org/installation.html).

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
