#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "Layer.h"

class InputLayer : public Layer {

public:
	InputLayer(int numLayerNodes, int numPrevLayerNodes) : Layer(numLayerNodes, numPrevLayerNodes) {
		generateInitialWeights(numLayerNodes, numPrevLayerNodes);
	}

protected:
	vector<double> sigmoid(vector<double> values);
	void generateInitialWeights(int numLayerNodes, int numPrevLayerNodes);
	
	// overridden. This function will just return values because weights are all 1
	// we can save some time.
	vector<double> dotProduct(vector<double> values);

};

#endif
