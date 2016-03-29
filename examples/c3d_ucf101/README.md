## UCF-101 training demo

Follow these steps to train C3D on UCF-101.

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`.
3. Change `${video-caffe-root}/examples/c3d_ucf101/{c3d_ucf101_train_split1.txt,c3d_ucf101_test.split1.txt}` to correctly point to UCF-101 videos.
4. Modify `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt)` to your taste or HW specification. Especially `batch_size` may have to be reduced for the GPU memory. Should run fine as is with 12GB GPU memory.
5. Run training script: e.g. `cd ${video-caffe-root} && examples/c3d_ucf101/train_ucf101.sh`
6. Sit back and enjoy.

## Files in this directory

* `train_ucf101.sh`: a main script to run for training C3D on UCF-101 data
* `c3d_ucf101_solver.prototxt`: a solver specifications -- SGD parameters, testing parametesr, etc
* `c3d_ucf101_test_split1.txt`, `c3d_ucf101_train_split1.txt`: lists of testing/training video clips in ("video directory", "starting frame num", "label") format
* `c3d_ucf101_train_test.prototxt`: training/testing network model
* `ucf101_train_mean.binaryproto`: a mean cube calculated from UCF101 training set
