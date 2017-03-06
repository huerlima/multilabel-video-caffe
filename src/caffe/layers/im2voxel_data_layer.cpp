#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/im2voxel_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
Im2VoxelDataLayer<Dtype>::Im2VoxelDataLayer(const LayerParameter& param)
: BasePrefetchingDataLayer<Dtype>(param),
  reader_(param) {
    LOG(INFO) << "starting im2voxel" << std::endl;
}

template <typename Dtype>
Im2VoxelDataLayer<Dtype>::~Im2VoxelDataLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void Im2VoxelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    const int batch_size = this->layer_param_.data_param().batch_size();
    const int temporal_length = this->layer_param_.im2voxel_data_param().temporal_length();

    LOG(INFO) << "im2voxel layersetup" << std::endl;
    // Read a data point, and use it to initialize the top blob.
    Datum& datum = *(reader_.full().peek());

    LOG(INFO) << "im2voxel layersetup2" << std::endl;
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    this->transformed_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    vector<int> voxel_shape(5);
    voxel_shape[0] = batch_size;
    voxel_shape[1] = top_shape[1];
    voxel_shape[2] = temporal_length;
    voxel_shape[3] = top_shape[2];
    voxel_shape[4] = top_shape[3];

    top[0]->Reshape(voxel_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(voxel_shape);
    }

    voxel_shape[0] = 1;
    this->last_voxel_.Reshape(voxel_shape);

    //read in labels
    const string labelSource = this->layer_param_.im2voxel_data_param().labelsource();
    LOG(INFO) << "Opening label file " << labelSource;
    std::ifstream infile(labelSource.c_str());
    int count = 0;
    string filename, label;
    label_list_.clear();
    while (infile >> filename >> label) {
    	std::istringstream iss(label);
    	std::string l;
    	vector<int> ls;
    	//for (int i=0; i < numLabels; ++i) {
    	//LOG(INFO) << label;
    	while (!iss.eof()) {
    		std::getline(iss,l,',');
    		ls.push_back(atoi(l.c_str()));
    	}
    	label_list_.push_back(ls);
    	//if (count == 50){
    	//    return;
    	//}
//    	shuffle_index_.push_back(count);
    	count++;
    }


    LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();
    // label
    if (this->output_labels_) {
        vector<int> label_shape(5, batch_size);
        label_shape[0] = batch_size;
        label_shape[1] = label_list_[0].size();
        label_shape[2] = 1;
        label_shape[3] = 1;
        label_shape[4] = 1;

        top[1]->Reshape(label_shape);
        LOG(INFO) << "reshaped label ";
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            LOG(INFO) << "fetching label ";
            this->prefetch_[i].label_.Reshape(label_shape);
        }
        LOG(INFO) << "fetched label ";

    }
    batch_prefetch_cnt_ = 0;
}

// This function is called on prefetch thread
template<typename Dtype>
void Im2VoxelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;

    //LOG(INFO) << "im2voxel load_batch" << std::endl;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    // TODO:
     // implement start of sequence
     // implement boundary conditions / padding
     // implement fixed cropping for whole sequence
     // read in text file for labels
     // implement that we keep voxelSize images in a local cache and take the first images from there



    // Reshape according to the first datum of each batch
    // on single input batches allows for inputs of varying dimension.
    const int batch_size = this->layer_param_.data_param().batch_size();
    const int temporal_length = this->layer_param_.im2voxel_data_param().temporal_length();
    Datum& datum = *(reader_.full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    //partVoxelSize represents the voxel size with a single image missing
    int imSize = top_shape[1] * top_shape[2] * top_shape[3];
    int partVoxelSize = (temporal_length - 1) * imSize;
    int fullVoxelSize = partVoxelSize + imSize;
    int repeatImagesStart = (temporal_length - 1)/2; //number of times the first image in a sequence gets copied
    int repeatImagesEnd = temporal_length - 1 - repeatImagesStart; //number of times the last image in a sequence gets copied

    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size and temporal_length.
    vector<int> voxel_shape(5);
    voxel_shape[0] = batch_size;
    voxel_shape[1] = top_shape[1];
    voxel_shape[2] = temporal_length;
    voxel_shape[3] = top_shape[2];
    voxel_shape[4] = top_shape[3];

    batch->data_.Reshape(voxel_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

    Dtype* last_data = last_voxel_.mutable_cpu_data();

    if (this->output_labels_) {
        top_label = batch->label_.mutable_cpu_data();
    }
    int sequenceStart = 1; // 0 means no sequence start.

    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a datum
        Datum& datum = *(reader_.full().pop("Waiting for data"));
        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply data transformations (mirror, scale, crop...)
        int offset = batch->data_.offset(item_id);
        int framesTillSeqEnd = -1;
//        for (int i = 1; i <= repeatImagesEnd; ++i) {
//        	if (sequence_list_[batch_prefetch_cnt_+][i]
//        }

        //distinguish if we are at sequence start or in the middle of a sequence
        if (sequenceStart != 0) {

            //read in first voxel (in a sequence): full temporal length
            for (int voxel_part_id = 0; voxel_part_id < temporal_length; ++voxel_part_id) {
                // get a datum
            	Datum& datum = *(reader_.full().pop("Waiting for data"));
                int currOffset = imSize * voxel_part_id;
                this->transformed_data_.set_cpu_data(top_data + offset + currOffset);
                this->data_transformer_->Transform(datum, &(this->transformed_data_));
                reader_.free().push(const_cast<Datum*>(&datum));

                // repeat first image
                if (voxel_part_id == 0) {
                	for (int rep = 1; rep <= repeatImagesStart; ++rep){
                		memcpy(top_data + offset + imSize * rep, top_data + offset, imSize);
                		++voxel_part_id;
                	}
                }

            }
            sequenceStart = 0;

        } else {
        	//no sequence has started here

            DLOG(INFO) << "item_id " << item_id << " offset " << offset << " partVoxelSize " << partVoxelSize << " fullVoxelSize " << fullVoxelSize<< std::endl;
            // copy the preceding temporal_length -1 images into the new location
            memcpy(top_data + offset, last_data,partVoxelSize);
            // add the current image
            //memcpy(top_data + offset + partVoxelSize, &datum, top_shape[1] * top_shape[2] * top_shape[3]);
            //LOG(INFO) << "item_id " ;
            this->transformed_data_.set_cpu_data(top_data + offset + partVoxelSize);
            //LOG(INFO) << "item_id " ;
            this->data_transformer_->Transform(datum, &(this->transformed_data_));

        }

        //last data stores the last images of a voxel (transformed), but not the first image of the last voxel
        memcpy(last_data, top_data + offset + imSize, partVoxelSize);

        // Copy label.
        //if (this->output_labels_) {
        DLOG(INFO) << "fetching label for item_id " << item_id << " batch " << batch_prefetch_cnt_ << " = "<< label_list_[batch_prefetch_cnt_][0] << std::endl;
        for (int i=0; i<label_list_[0].size(); ++i){
        	top_label[item_id *  label_list_[0].size() + i] = label_list_[batch_prefetch_cnt_][i];
        	//LOG(INFO) << "fetching label for item_id " << item_id << " labeli " << i << " = "<< datum.label(i) << std::endl;
        }
        //}
        trans_time += timer.MicroSeconds();

        reader_.free().push(const_cast<Datum*>(&datum));
        batch_prefetch_cnt_++;
        if (batch_prefetch_cnt_ >= label_list_.size()) {
             // We have reached the end. Restart from the first.
             LOG(INFO) << "Restarting label prefetching from start.";
             batch_prefetch_cnt_ = 0;
        }
    }






    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(Im2VoxelDataLayer);
REGISTER_LAYER_CLASS(Im2VoxelData);

}  // namespace caffe
