#include "../inc/NeuralNetwork.h"
#include <iostream>


/* **************************************************************** */


NeuralNetwork::NeuralNetwork(int numOfLayers, int nodesOfInput, int nodesOfHidden, int nodesOfOutput) : 
	numLayers(numOfLayers) {

	// push back input layer
	layers.push_back(new InputLayer(nodesOfInput, nodesOfInput));

	// push back all (layers - 2) amount of hidden layers
	for (int i = 0; i < numLayers - 2; i++) {
		if (i == 0)
			layers.push_back(new HiddenLayer(nodesOfHidden, nodesOfInput));
		else
			layers.push_back(new HiddenLayer(nodesOfHidden, nodesOfHidden));
	}

	// push back output layer
	layers.push_back(new OutputLayer(nodesOfOutput, nodesOfHidden));

}

/* **************************************************************** */


NeuralNetwork::~NeuralNetwork() {
	for (Layer* layer : layers) {
		delete layer;
	}
};

/* **************************************************************** */

void NeuralNetwork::setLearning(double learning) {
	learningRate = learning;
}

double NeuralNetwork::getLearning() {
	return learningRate;
}
/* **************************************************************** */

void NeuralNetwork::setSigmoidGain(double input) {
    sigmoidGain = input;
}

double NeuralNetwork::getSigmoidGain() {
    return sigmoidGain;
}

/* **************************************************************** */

void NeuralNetwork::trainingProcess() {

	cout << "Starting to train network..." << endl;
	
	for(int i = 0; i < trainingInputs.size(); i++) 
	{
	    cout << "Training picture " << i << endl;
		train(trainingInputs.at(i), expectedResults.at(i));
	}
}

/* **************************************************************** */

void NeuralNetwork::train(vector<double> input, vector<double> target) {

	// run through the network
	vector<double> actual = query(input);


	// find our initial errors
	vector<double> errors;
	for (int i = 0; i < target.size(); i++) {
		errors.push_back(target.at(i) - actual.at(i));
	}

    vector<double> error_total;
    for(int i = 0; i < errors.size(); i++){
        error_total.push_back(0.5 * pow(errors.at(i), 2));
    }

	// back propogate errors - we do not need to call for input layer
//	cout << "Backpropogating errors..." << endl;
	for (int i = layers.size() - 1 ; i > 0; i--) {
		errors = layers.at(i)->backPropogate(errors, layers.at(i - 1)->prevOutput, learningRate);
	}
}

/* **************************************************************** */

vector<double> NeuralNetwork::query(vector<double> inputs) {

	// run through all layers
	vector<double> temp = layers.at(0)->process(inputs, sigmoidGain);
	for (int i = 1; i < layers.size(); i++) {
		temp = layers.at(i)->process(temp, sigmoidGain);
	}

	// return output layer output
	return temp;
}

/* **************************************************************** */






