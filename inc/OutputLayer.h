#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include "Layer.h"

class OutputLayer : public Layer {

public:
	OutputLayer(int numLayerNodes, int numPrevLayerNodes) : Layer(numLayerNodes, numPrevLayerNodes) {
		generateInitialWeights(numLayerNodes, numPrevLayerNodes);
	}

protected:
	vector<double> sigmoid(vector<double> values);
	void generateInitialWeights(int numLayerNodes, int numPrevLayerNodes);

};

#endif
