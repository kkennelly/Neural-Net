
#include "../inc/InputLayer.h"

// we don't want this layer to execute the activation function,
// so we will just return the input values
vector<double> InputLayer::sigmoid(vector<double> values, double gain) {

	return values;
}

// input layer weights should be all 1
void InputLayer::generateInitialWeights(int numOfLayerNodes, int numOfPrevLayerNodes) {

	vector< vector<double> > matrix;

	for (int i = 0; i < numOfPrevLayerNodes; i++) {
		vector<double> temp;

		for (int j = 0; j < numOfLayerNodes; j++) {
			temp.push_back(1);
		}

		matrix.push_back(temp);
	}

	weights = matrix;
}

vector<double> InputLayer::dotProduct(vector<double> values) {
	return values;
}
