#ifndef HIDDEN_LAYER_H
#define HIDDEN_LAYER_H

#include "Layer.h"

class HiddenLayer : public Layer {

public:
	HiddenLayer(int numLayerNodes, int numPrevLayerNodes) : Layer(numLayerNodes, numPrevLayerNodes) {
		generateInitialWeights(numLayerNodes, numPrevLayerNodes);
	}

protected:
	vector<double> sigmoid(vector<double> values, double gain);
	void generateInitialWeights(int numLayerNodes, int numPrevLayerNodes);

    int getId(){
        return 2;
    }

};

#endif
