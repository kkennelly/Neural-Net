#include "../inc/HiddenLayer.h"
#include  <random>
#include <time.h>

// implement activation function on all values
vector<double> HiddenLayer::sigmoid(vector<double> values, double gain) {
	vector<double> temp;

	for (auto val : values) {
		temp.push_back(1 / (1 + exp(-val/gain)));
	}

	return temp;
}

// fill weights matrix with all random doubles between 0 and 1
void HiddenLayer::generateInitialWeights(int numOfLayerNodes, int numOfPrevLayerNodes) {
	vector< vector<double> > matrix;

//	std::default_random_engine generator;
//  std::uniform_real_distribution<double> distribution(0.00000000001,1.0);

	for (int i = 0; i < numOfPrevLayerNodes; i++) {
		vector<double> temp;

		for (int j = 0; j < numOfLayerNodes; j++) {
//			temp.push_back(distribution(generator));
            temp.push_back((double) rand()/RAND_MAX - 0.5);
		}

		matrix.push_back(temp);
	}

	weights = matrix;
}