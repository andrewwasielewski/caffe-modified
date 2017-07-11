/*
 * CaffeAPI.cpp
 *
 *  Created on: Dec 29, 2016
 *      Author: z003arva
 */

#include "CaffeAPI.h"

Caffe_API::Caffe_API() {
}

Caffe_API::~Caffe_API() {
}

void Caffe_API::readNetwork(const string& proto_file,const string& trained_file){
//	ReadNetParamsFromTextFileOrDie(proto_file, &paramText_);
//	ReadNetParamsFromBinaryFileOrDie(trained_file, &paramBinary_);
	net_.reset(new Net<float>(proto_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);
	vector<Blob<float>*> input_layer = net_->input_blobs();
	inputShape_ = input_layer[0]->shape();
}


void Caffe_API::inputData(float* data,const string& blob_name){
	vector<Blob<float>*> input_layer = net_->input_blobs();
	float* layerdata = input_layer[0]->mutable_cpu_data();
	for (int i = 0; i < input_layer[0]->count(); ++i) {
		layerdata[i] = data[i];
	}
}

void Caffe_API::readTestDataFromBinFile(const char* datafile,const string& blob_name){
	shared_ptr<Blob<float> > input_blob = net_->blob_by_name(blob_name);
	float* inputdata = input_blob->mutable_cpu_data();
	std::fstream datain;
	datain.open(datafile,std::ios::in|std::ios::binary);
	CHECK(datain.is_open())<<"Cannot open Data file\n";
	datain.read((char*)inputdata,input_blob->count()*sizeof(float));
	datain.close();
}

void Caffe_API::inputData(float*** data,const string& blob_name,bool transpose){
	shared_ptr<Blob<float> > input_blob = net_->blob_by_name(blob_name);
	float* inputdata = input_blob->mutable_cpu_data();
	int counter = 0;
	if(transpose){
		for (int x = 0; x < inputShape_[2]; ++x) {
			for (int y = 0; y < inputShape_[3]; ++y) {
				for (int z = 0; z < inputShape_[4]; ++z){
					inputdata[counter++] = data[z][y][x];
				}
			}
		}
	}else{
		for (int z = 0; z < inputShape_[4]; ++z) {
			for (int y = 0; y < inputShape_[3]; ++y) {
				for (int x = 0; x < inputShape_[2]; ++x) {
					inputdata[counter++] = data[z][y][x];
				}
			}
		}
	}

}

void Caffe_API::outputData(vector<float>& data,const string& blob_name){
	shared_ptr<Blob<float> > output_blob = net_->blob_by_name(blob_name);
	const float* begin_i = output_blob->cpu_data();
	const float* end_i = begin_i + output_blob->count();
	data = vector<float>(begin_i,end_i);
}

void Caffe_API::run(){
	net_->Forward();
}
