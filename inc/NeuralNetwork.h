#pragma once

#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#define DEFAULT_LEARNING	0.3
#define DEFAULT_GAIN        1.0

#include <random>
#include <vector>
#include "Layer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"


using namespace std;

class NeuralNetwork {

public:
	NeuralNetwork(int numOfLayers, int nodesOfInput, int nodesOfHidden, int nodesOfOutput);
	~NeuralNetwork();

	void setLearning(double learning);
	double getLearning();

	void setSigmoidGain(double input);
	double getSigmoidGain();

	void trainingProcess();

	vector<double> query(vector<double> inputs);

	void setTrainingData(vector< vector<double> > expected, vector< vector<double> > inputs){
		expectedResults = expected;
		trainingInputs = inputs;	
	}

	// TODO: Move back to private variable
    vector <Layer*> layers;
    vector< vector<double> > expectedResults;
    vector< vector<double> > trainingInputs;

private:
	int numLayers;
	double learningRate = DEFAULT_LEARNING;
	double sigmoidGain = DEFAULT_GAIN;



	vector< vector<double> > initializeWeights(int height, int width);
	void train(vector<double> input, vector<double> target);

};

#endif
