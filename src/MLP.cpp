#include "MLP.h"

MatrixXf MLP::dW(RowVectorXf neuro, RowVectorXf delta) {
	return neuro.transpose() * delta;
}
void MLP::forward() {

	for (size_t j = 0; j < sizes.size() - 1; ++j) {
		neuron[j + 1] = neuron[j] * weight[j] + bias[j];
		if (j == sizes.size() - 2)
			neuron[j + 1] = outActiv(neuron[j + 1]);
		else
			neuron[j + 1] = (hidActiv(neuron[j + 1])).array()* neuronStencil[j + 1].cast<float>().array();
	}
}
void MLP::backprop() {
	Loss += lossFunc(nTrue, neuron[neuron.size() - 1]);

	deltaNeuron[sizes.size() - 1] = deltaOut(nTrue, neuron[sizes.size() - 1]);

	for (size_t i = sizes.size() - 1; i > 0; --i) {
		deltaWeight[i - 1] = dW(neuron[i - 1], deltaNeuron[i]);
		deltaWeightAdd[i - 1] += deltaWeight[i - 1];
		deltaBias[i - 1] = deltaNeuron[i];
		deltaBiasAdd[i - 1] += deltaBias[i - 1];
		deltaNeuron[i - 1] = ((weight[i - 1] * deltaNeuron[i].transpose()).transpose()).array() * dHidActiv(neuron[i - 1]).array();
	}
}
void MLP::update() {
	for (size_t i = 0; i < weight.size(); ++i) {
		weight[i] = weight[i] - lr * deltaWeightAdd[i];
		bias[i] = bias[i] - lr * deltaBiasAdd[i];
	}
	Loss /= batchSize;
}
void MLP::init() {
	for (size_t i = 0; i < sizes.size(); ++i) {
		RowVectorXf v = RowVectorXf::Ones(sizes[i]);
		RowVectorXi vi = RowVectorXi::Ones(sizes[i]);
		neuron.push_back(v);
		neuronStencil.push_back(vi);
		deltaNeuron.push_back(v);
	}

	for (size_t i = 1; i < sizes.size(); ++i) {
		RowVectorXf v = RowVectorXf::Zero(sizes[i]);
		bias.push_back(v);
		deltaBias.push_back(VectorXf::Zero(v.size()));
		deltaBiasAdd.push_back(VectorXf::Zero(v.size()));
	}

	random_device rd;
	mt19937 gen(rd());

	for (size_t i = 0; i < sizes.size() - 1; ++i) {
		float std_dev = sqrt(2.0f / sizes[i]);
		normal_distribution<float> dist(0.0f, std_dev);
		MatrixXf m(sizes[i], sizes[i + 1]);

		for (size_t j = 0; j < sizes[i]; ++j)
			for (size_t k = 0; k < sizes[i + 1]; ++k)
				m(j, k) = dist(gen);

		weight.push_back(m);
		deltaWeight.push_back(MatrixXf::Zero(sizes[i], sizes[i + 1]));
		deltaWeightAdd.push_back(MatrixXf::Zero(sizes[i], sizes[i + 1]));
	}
	nTrue = RowVectorXf::Zero(neuron[neuron.size() - 1].size());
}

void MLP::backClear() {
	for (size_t i = 0; i < deltaWeightAdd.size(); ++i)
		deltaWeightAdd[i].setZero();
	for (size_t i = 0; i < deltaBiasAdd.size(); ++i)
		deltaBiasAdd[i].setZero();
	Loss = 0;
}

MLP::MLP(
	vector<size_t> sizes,
	RowVectorXf(*hidActiv)(RowVectorXf),
	RowVectorXf(*outActiv)(RowVectorXf),
	float (*lossFunc)(RowVectorXf, RowVectorXf),
	RowVectorXf(*dHidActiv)(RowVectorXf),
	RowVectorXf(*deltaOut)(RowVectorXf, RowVectorXf),
	float lr = 0.01
) : sizes{ sizes },
hidActiv{ hidActiv },
outActiv{ outActiv },
lossFunc{ lossFunc },
dHidActiv{ dHidActiv },
deltaOut{ deltaOut },
lr{ lr }
{
	init();
}

void MLP::setDB_CSV(const string& fileName) {
	ifstream file(fileName);
	string line;
	dataBase.clear();

	while (getline(file, line)) {
		stringstream ss(line);
		string cell;
		while (getline(ss, cell, ',')) {
			dataBase.push_back(stof(cell));
		}
		std::cout << "loaded from database: " << dataBase.size() / (785 * 600) << "%\033[H";
	}
	std::cout << "\033[2J\033[H";
}

// MLP training method
void MLP::learn(float error, size_t batchSize) {
	this->batchSize = batchSize;
	size_t maxEpochs = 2000;
	size_t epoch = 0;
	size_t numOfExample = 0;
	size_t exampleSize = neuron[0].size() + 1;
	size_t sizeOfExamples = (size_t)(dataBase.size() / exampleSize);
	size_t rundomDatabasePtr = 0;

	for (size_t i = 0; i < batchSize; ++i) {
		batch.push_back(neuron[0]);
	}

	for (size_t i = 0; i < batchSize; ++i)
		bTrue.push_back(RowVectorXf::Zero(nTrue.size()));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution d(0.8);
	std::uniform_int_distribution<> dist(0, sizeOfExamples - 1);

	// main learning cycle
	while (numOfExample < sizeOfExamples)
	{


		for (size_t i = 0; i < batchSize; ++i) {
			rundomDatabasePtr = dist(gen);
			for (size_t j = 0; j < bTrue[i].size(); ++j)
			{
				if (j == static_cast<size_t>(dataBase[rundomDatabasePtr * exampleSize]))
					bTrue[i](j) = 1.0f;
				else
					bTrue[i](j) = 0.0f;
			}
			for (size_t j = 0; j < neuron[0].size(); ++j)
			{
				batch[i](j) = dataBase[(rundomDatabasePtr * exampleSize + 1) + j] / 255.f;
			}
			++numOfExample;
			if (numOfExample >= sizeOfExamples) break;
		}
		
		for (size_t i = 0; i < neuron.size(); ++i) {
			for (size_t j = 0; j < neuron.size(); ++j) {
				neuronStencil[i](j) = d(gen)?1:0;
			}
		}

		do
		{
			backClear();
			for (size_t i = 0; i < batchSize; ++i)
			{
				neuron[0] = batch[i];
				nTrue = bTrue[i];
				forward();
				backprop();
			}
			update();
			++epoch;
		} while ((Loss > error) && (epoch < maxEpochs));
		epoch = 0;
		std::cout << "\033[2J\033[H";
		std::cout << (float)(numOfExample + 1) * neuron[0].size() / dataBase.size() * 100 << "%\n";
	}
	for (size_t i = 0; i < neuron.size(); ++i) {
		for (size_t j = 0; j < neuron.size(); ++j) {
			neuronStencil[i](j) = 1;
		}
	}
}

void MLP::test(RowVectorXf input) {
	neuron[0] = input;
	forward();
}

RowVectorXf MLP::answer() {
	return neuron[neuron.size() - 1];
}