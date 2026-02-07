#pragma once
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>

using namespace Eigen;
using namespace std;
class MLP
{
private:
	vector<size_t> sizes;
	vector <RowVectorXf> bTrue;
	RowVectorXf nTrue;
	vector<RowVectorXf> neuron;
	vector<RowVectorXi> neuronStencil;
	vector<MatrixXf> weight;
	vector<RowVectorXf> bias;
	vector<RowVectorXf> deltaNeuron;
	vector<MatrixXf> deltaWeight;
	vector<RowVectorXf> deltaBias;
	vector<MatrixXf> deltaWeightAdd;
	vector<RowVectorXf> deltaBiasAdd;
	vector<RowVectorXf> batch;
	float Loss;
	float lr;
	vector<float> dataBase;
	size_t batchSize;

	RowVectorXf(*hidActiv)(RowVectorXf);
	RowVectorXf(*outActiv)(RowVectorXf);
	float (*lossFunc)(RowVectorXf, RowVectorXf);

	RowVectorXf(*dHidActiv)(RowVectorXf);
	RowVectorXf(*deltaOut)(RowVectorXf, RowVectorXf);

	MatrixXf dW(RowVectorXf neuro, RowVectorXf delta);
	void backClear();
	void forward();
	void backprop();
	void update();
	void init();
public:
	MLP(
		vector<size_t> sizes,
		RowVectorXf(*hidActiv)(RowVectorXf),
		RowVectorXf(*outActiv)(RowVectorXf),
		float (*lossFunc)(RowVectorXf, RowVectorXf),
		RowVectorXf(*dHidActiv)(RowVectorXf),
		RowVectorXf(*deltaOut)(RowVectorXf, RowVectorXf),
		float lr
	);
	void setDB_CSV(const string& fileName);
	void learn(float error, size_t batchSize);
	void test(RowVectorXf input);
	RowVectorXf answer();
};


