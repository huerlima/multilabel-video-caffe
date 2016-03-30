## params
# test.prototxt
# model file
# id of gpu
# batch_size
# mini_batch_num
# prefix file
# target feature
../../build/tools/predict.bin \
c3d_ucf101_test.prototxt tmp_models/c3d_vqa_iter_25.caffemodel 3 16 1 videos_output_prefix.txt fc8