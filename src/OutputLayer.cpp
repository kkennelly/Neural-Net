
#include "../inc/OutputLayer.h"

// implement activation function on all values
vector<double> OutputLayer::sigmoid(vector<double> values) {

	vector<double> temp;

	for (auto val : values) {
		temp.push_back(1 / (1 + exp(-val)));
	}

	return temp;
}

// fill weights matrix with all random doubles between 0 and 1
void OutputLayer::generateInitialWeights(int numLayerNodes, int numPrevLayerNodes) {

	vector< vector<double> > matrix;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(0.0, 1.0);


	for (int i = 0; i < numPrevLayerNodes; i++) {
		vector<double> temp;

		for (int j = 0; j < numLayerNodes; j++) {
			temp.push_back(dis(gen));
		}

		matrix.push_back(temp);
	}

	weights = matrix;
}
