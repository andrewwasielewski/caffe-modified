/*
 * CaffeAPI.h
 *
 *  Created on: Dec 29, 2016
 *      Author: z003arva
 */

#ifndef CAFFE_API_CAFFEAPI_H_
#define CAFFE_API_CAFFEAPI_H_

#include <caffe\caffe.hpp>
#include <string>
#include <vector>
#include <memory>

using namespace caffe;
using std::string;
using std::vector;

class Caffe_API {
public:
	Caffe_API();
	void setMode(bool usegpu){
		if(usegpu)
			Caffe::set_mode(Caffe::GPU);
		else
			Caffe::set_mode(Caffe::CPU);
	}
	void readNetwork(const string& proto_file,const string& trained_file);
	void resetNet(){net_.reset();}
	void inputData(float *data,const string& blob_name);
	void inputData(float ***data,const string& blob_name,bool transpose = false);
	void outputData(vector<float>& data,const string& blob_name);
	void readTestDataFromBinFile(const char* datafile,const string& blob_name);
	void run();
	virtual ~Caffe_API();
protected:
	shared_ptr<Net<float> > net_;
	//NetParameter paramText_,paramBinary_;
	vector<int> inputShape_;
};

#endif /* CAFFE_API_CAFFEAPI_H_ */
