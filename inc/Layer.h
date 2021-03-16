#ifndef LAYER_H
#define LAYER_H

#include <random>
#include <vector>
#include <iostream>
using namespace std;


class Layer {

public:
	Layer(int numLayerNodes, int numPrevLayerNodes) {

	}

	~Layer(){

	}

	vector<double> process(vector<double> incoming_values) {
		vector<double> temp = dotProduct(incoming_values, transpose(weights));
		prevOutput = sigmoid(temp);
		return prevOutput;
	}
	vector<double> backPropogate(vector<double> error, vector<double> prevLayerOutputs, double learningRate);
	vector<double> prevOutput;


protected:
	vector< vector<double> > weights;

	virtual vector<double> sigmoid(vector<double> values) = 0;
	virtual void generateInitialWeights(int numLayerNodes, int numPrevLayerNodes) = 0;
	virtual int getId() = 0;

	vector<double> dotProduct(vector<double> values, vector< vector<double> > matrix);
	vector< vector<double> > transpose();
	vector <vector<double> > transpose(vector< vector<double> > matrix);
};

#endif
