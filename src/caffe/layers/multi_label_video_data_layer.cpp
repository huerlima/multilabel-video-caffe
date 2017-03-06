#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/multi_label_video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiLabelVideoDataLayer<Dtype>::~MultiLabelVideoDataLayer<Dtype>() {
	this->StopInternalThread();
}

template <typename Dtype>
void MultiLabelVideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>&
		bottom, const vector<Blob<Dtype>*>& top) {
	const int new_length = this->layer_param_.multi_label_video_data_param().new_length();
	const int new_height = this->layer_param_.multi_label_video_data_param().new_height();
	const int new_width  = this->layer_param_.multi_label_video_data_param().new_width();
	const bool is_color  = this->layer_param_.multi_label_video_data_param().is_color();
	string root_folder = this->layer_param_.multi_label_video_data_param().root_folder();

	CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
					"new_height and new_width to be set at the same time.";
	// Read the file with filenames and labels
	const string& source = this->layer_param_.multi_label_video_data_param().source();
	LOG(INFO) << "Opening file " << source;

	std::ifstream infile(source.c_str());
	string filename;
	int frame_num;
    int count = 0;
    string multilabel;


	LOG(INFO) << " here we go " << std::endl;
	while (infile >> filename >> frame_num >> multilabel) {
		triplet video_and_label;
		video_and_label.first = filename;
		video_and_label.second = frame_num;
		//file_list_.push_back(filename);
		//start_frm_list_.push_back(start_frm);
		std::istringstream iss(multilabel);
		std::string l;
		vector<int> ls;
		//LOG(INFO) << "reading label for lines_id_read " << count <<" ";
		while (!iss.eof()) {
			std::getline(iss,l,',');
			//LOG(INFO) <<  l.c_str() << "," ;
			ls.push_back(atoi(l.c_str()));
		}
		video_and_label.third = ls;

		lines_.push_back(video_and_label);
		//LOG(INFO)  << std::endl;
		//if (count == 50){
		//    return;
		//}
		//shuffle_index_.push_back(count);
		count++;
	}

	if (this->layer_param_.multi_label_video_data_param().shuffle()) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleVideos();
	}
	LOG(INFO) << "A total of " << lines_.size() << " video chunks.";

	lines_id_ = 0;
	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.multi_label_video_data_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand() %
				this->layer_param_.multi_label_video_data_param().rand_skip();
		LOG(INFO) << "Skipping first " << skip << " data points.";
		CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
		lines_id_ = skip;
	}
	// Read a video clip, and use it to initialize the top blob.
	std::vector<cv::Mat> cv_imgs;
	bool read_video_result = ReadVideoToCVMat(root_folder +
			lines_[lines_id_].first,
			lines_[lines_id_].second,
			new_length, new_height, new_width,
			is_color,
			&cv_imgs);
	CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
			" at frame " << lines_[lines_id_].second << ".";
	CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
			lines_[lines_id_].first <<
			" at frame " <<
			lines_[lines_id_].second <<
			" correctly.";
	// Use data_transformer to infer the expected blob shape from a cv_image.
	const bool is_video = true;
	vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
			is_video);
	this->transformed_data_.Reshape(top_shape);
	// Reshape prefetch_data and top[0] according to the batch_size.
	const int batch_size = this->layer_param_.multi_label_video_data_param().batch_size();
	CHECK_GT(batch_size, 0) << "Positive batch size required";
	top_shape[0] = batch_size;
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(top_shape);
	}
	top[0]->Reshape(top_shape);

	LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
			<< top[0]->shape(1) << "," << top[0]->shape(2) << ","
			<< top[0]->shape(3) << "," << top[0]->shape(4);
	// label

    vector<int> label_shape(5);
    LOG(INFO) << "    count=" << count << " lines_[0].third.size()" << lines_[0].third.size();
    label_shape[0] = batch_size;
    label_shape[1] = lines_[0].third.size();
    label_shape[2] = 1;
    label_shape[3] = 1;
    label_shape[4] = 1;

    //top[1]->Reshape(newshape);
    //prefetch_label_.reset(new Blob<Dtype>(newshape));

	//vector<int> label_shape(1, batch_size);
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].label_.Reshape(label_shape);
	}


}

template <typename Dtype>
void MultiLabelVideoDataLayer<Dtype>::ShuffleVideos() {
	caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void MultiLabelVideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());
	MultiLabelVideoDataParameter multi_label_video_data_param = this->layer_param_.multi_label_video_data_param();
	const int batch_size = multi_label_video_data_param.batch_size();
	const int new_length = multi_label_video_data_param.new_length();
	const int new_height = multi_label_video_data_param.new_height();
	const int new_width = multi_label_video_data_param.new_width();
	const bool is_color = multi_label_video_data_param.is_color();
	string root_folder = multi_label_video_data_param.root_folder();

	// Reshape according to the first image of each batch
	// on single input batches allows for inputs of varying dimension.
	std::vector<cv::Mat> cv_imgs;
	bool read_video_result = ReadVideoToCVMat(root_folder +
			lines_[lines_id_].first,
			lines_[lines_id_].second,
			new_length, new_height, new_width,
			is_color,
			&cv_imgs);
	CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
			" at frame " << lines_[lines_id_].second << ".";
	CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
			lines_[lines_id_].first <<
			" at frame " <<
			lines_[lines_id_].second <<
			" correctly.";
	// Use data_transformer to infer the expected blob shape from a cv_imgs.
	bool is_video = true;
	vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
			is_video);
	this->transformed_data_.Reshape(top_shape);
	// Reshape batch according to the batch_size.
	top_shape[0] = batch_size;
	batch->data_.Reshape(top_shape);

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_label = batch->label_.mutable_cpu_data();

	// datum scales
	const int lines_size = lines_.size();
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		// get a blob
		timer.Start();
		CHECK_GT(lines_size, lines_id_);
		std::vector<cv::Mat> cv_imgs;
		bool read_video_result = ReadVideoToCVMat(root_folder +
				lines_[lines_id_].first,
				lines_[lines_id_].second,
				new_length, new_height,
				new_width, is_color, &cv_imgs);
		CHECK(read_video_result) << "Could not load " << lines_[lines_id_].first <<
				" at frame " << lines_[lines_id_].second << ".";
		CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
				lines_[lines_id_].first <<
				" at frame " <<
				lines_[lines_id_].second <<
				" correctly.";
		read_time += timer.MicroSeconds();
		timer.Start();
		// Apply transformations (mirror, crop...) to the image
		int offset = batch->data_.offset(item_id);
		this->transformed_data_.set_cpu_data(prefetch_data + offset);
		const bool is_video = true;
		this->data_transformer_->Transform(cv_imgs, &(this->transformed_data_),
				is_video);
		trans_time += timer.MicroSeconds();

		//prefetch_label[item_id] = lines_[lines_id_].third;

		for (int i=0; i< lines_[lines_id_].third.size(); ++i){
			prefetch_label[item_id *  lines_[lines_id_].third.size() + i] = lines_[lines_id_].third[i];
			//LOG(INFO) << "fetching label for lines_id_ " << lines_id_ << " labeli " << i << " = "<< lines_[lines_id_].third[i] << std::endl;
		}

		// go to the next iter
		lines_id_++;
		if (lines_id_ >= lines_size) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.multi_label_video_data_param().shuffle()) {
				ShuffleVideos();
			}
		}
	}
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiLabelVideoDataLayer);
REGISTER_LAYER_CLASS(MultiLabelVideoData);

}  // namespace caffe
#endif  // USE_OPENCV
