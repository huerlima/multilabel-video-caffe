# Video-Caffe: Caffe with C3D implementation and video reader

[![Build Status](https://travis-ci.org/chuckcho/video-caffe.svg?branch=master)](https://travis-ci.org/chuckcho/video-caffe)

This is 3-D Convolution (C3D) and video reader implementation in the latest Caffe (Apr 2016). The original [Facebook C3D implementation](https://github.com/facebook/C3D/) is branched out from Caffe on July 17, 2014 with git commit [b80fc86](https://github.com/BVLC/caffe/tree/b80fc862952ba4e068cf74acc0823785ce1cc0e9), and has not been rebased with the original Caffe, hence missing out quite a few new features in the lastest Caffe. I therefore pulled in C3D concept and an accompanying video reader and applied to the latest Caffe, and will try to rebase this repo with the upstream whenever there is a new important feature. This video-caffe is rebased on [df412a](https://github.com/chuckcho/video-caffe/commit/df412ac0da3e2e7eb194f0c16842fd126496d90d), on June 2, 2016.
Please reach [me](https://github.com/chuckcho) for any feedback or question.

Check out the [original Caffe readme](README-original.md) for Caffe-specific information.

## Requirements

In addition to [prerequisites for Caffe](http://caffe.berkeleyvision.org/installation.html#prerequisites), video-caffe depends on cuDNN. It is known to work with CuDNN v4 and v5(RC), but it may need some tweaks to build with v3.

* If you use "make" to build make sure `Makefile.config` point to the right paths for CUDA and CuDNN.
* If you use "cmake" to build, double-check `CUDNN_INCLUDE` and `CUDNN_LIBRARY`. You may want to cmake with something like `cmake -DCUDNN_INCLUDE="/your/path/to/include" -DCUDNN_LIBRARY="/your/path/to/lib" ${video-caffe-root}`.

## Building video-caffe

In a nutshell, key steps to build video-caffe are:

1. `git clone git@github.com:chuckcho/video-caffe.git`
2. `cd video-caffe`
3. `mkdir build && cd build`
4. `cmake ..`
5. Make sure CUDA and CuDNN are detected and their paths are correct.
6. `make all`
7. `make install`
8. (optional) `make runtest`

## UCF-101 training demo

Follow these steps to train C3D on UCF-101.

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`
3. (Optional) video reader works more stably with extracted frames than directly with video files. Extract frames from UCF-101 videos by revising and running a helper script, `${video-caffe-root}/examples/c3d_ucf101/extract_UCF-101_frames.sh`.
4. Change `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_{train,test}_split1.txt` to correctly point to UCF-101 videos or directories that contain extracted frames.
5. Modify `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt` to your taste or HW specification. Especially `batch_size` may have to be reduced for the GPU memory. Should run fine as is with 6GB GPU memory.
6. Run training script: e.g. `cd ${video-caffe-root} && examples/c3d_ucf101/train_ucf101.sh`
7. Sit back and enjoy.

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
