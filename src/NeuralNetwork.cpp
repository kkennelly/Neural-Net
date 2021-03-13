#include "../inc/NeuralNetwork.h"
#include <iostream>


/* **************************************************************** */


NeuralNetwork::NeuralNetwork(int numOfLayers, int nodesOfInput, int nodesOfHidden, int nodesOfOutput) : 
	numLayers(numOfLayers) {

	// push back input layer
	layers.push_back(new InputLayer(nodesOfInput, nodesOfInput));

	// push back all (layers - 2) amount of hidden layers
	for (int i = 0; i < numLayers - 2; i++) {
		if (i = 0)
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

void NeuralNetwork::trainingProcess() {

	cout << "Starting to train network..." << endl;
	
	for(int i = 0; i < trainingInputs.size(); i++) 
	{
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
	
	// find average of the errors
	int average;
	for (int i = 0; i < errors.size(); i++) {
		average  = average + errors.at(i);
	}
	average = average / 10;
	cout << "Average error for nodes: " << average << endl;

	// back propogate errors - we do not need to call for input layer
	cout << "Backpropogating errors..." << endl;
	for (int i = layers.size() - 1 ; i > 0; i--) {
		cout << "Working on layer " << i << endl;
		errors = layers.at(i)->backPropogate(errors, layers.at(i - 1)->prevOutput, learningRate);
	}
}

/* **************************************************************** */

vector<double> NeuralNetwork::query(vector<double> inputs) {

	// run through all layers
	vector<double> temp = layers.at(0)->process(inputs);
	for (int i = 1; i != layers.size(); i++) {
		temp = layers.at(i)->process(temp);
	}

	// return output layer output
	return temp;
}

/* **************************************************************** */






