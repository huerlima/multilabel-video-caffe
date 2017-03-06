#ifndef CAFFE_IM2VOXEL_DATA_LAYER_HPP_
#define CAFFE_IM2VOXEL_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class Im2VoxelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit Im2VoxelDataLayer(const LayerParameter& param);
  virtual ~Im2VoxelDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Im2VoxelDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
  vector< vector<int> > label_list_;
  vector< vector<int> > sequence_list_;
  int batch_prefetch_cnt_;
  Blob<Dtype> last_voxel_;

};

}  // namespace caffe

#endif  // CAFFE_IM2VOXEL_DATA_LAYER_HPP_
