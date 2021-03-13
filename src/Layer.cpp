#include "../inc/Layer.h"


vector<double> Layer::dotProduct(vector<double> values) {
	return dotProduct(values, weights);
}

vector<double> Layer::dotProduct(vector<double> values, vector< vector<double> > matrix) {
	vector<double> temp;
	double running = 0;

	for (int i = 0; i != matrix[0].size(); i++) {
		running = 0;
		for (int j = 0; j != values.size(); j++) {
			running += matrix[i][j] * values[j];
		}

		temp.push_back(running);
	}

	return temp;
}

vector< vector<double> > Layer::transpose() {

	return transpose(weights);
}

vector <vector<double> > Layer::transpose(vector< vector<double> > matrix) {
	// initialize new matrix
	vector< vector<double> > transposed;

	for (int col = 0; col != matrix.size(); col++) {
		vector <double> newCol;
		for (int row = 0; row != matrix[0].size(); row++) {
			newCol.push_back(matrix[row][col]);
		}
		transposed.push_back(newCol);
	}

	return transposed;
}

vector<double> Layer::backPropogate(vector<double> error, vector<double> prevLayerOutputs, double learningRate) {

	cout << "Getting sum of weights..." << endl;
	// get sum of weights
	vector<double> sums;
	double sum = 0.0;
	for (int col = 0; col < weights[0].size(); col++) {
		sum = 0.0;
		for (int row = 0; row < weights.size(); row++) {
			sum = sum + weights.at(row).at(col);
		}
		sums.push_back(sum);
	}
	
	cout << "Finding error across weights..." << endl;
	
	// generate matrix containing all the weight contribution information
	vector< vector<double> > errorAcrossWeights;
	for (int row = 0; row < weights.size(); row++) {
		vector<double> rowInfo;
		for (int col = 0; col < weights[0].size(); col++) {
			rowInfo.push_back(weights.at(row).at(col) / sums.at(col));
		}
		errorAcrossWeights.push_back(rowInfo);
	}	

	// change in weight = learning rate * error (at node) * sigmoid (output of node) * (1 - sigmoid (output of node)) * (output of prev node)

	vector<double> weightTimesPrevLayer = dotProduct(prevLayerOutputs, transpose(weights));
	vector<double> sigmoid = this->sigmoid(weightTimesPrevLayer);

	cout << "Changing weights..." << endl;
	
	cout << "Error size: " << error.size() << endl;
	cout << "Weights height: " << weights.size() << endl;
	cout << "Weights length: " << weights[0].size() << endl;
	cout << "Sigmoid size: " << sigmoid.size() << endl;
	cout << "Prev Layer Outputs: " << prevLayerOutputs.size() << endl;
	
	for (int row = 0; row < weights.size(); row++) {
		double change = learningRate * error.at(	row) * sigmoid.at(row) * (1 - sigmoid.at(row)) * prevLayerOutputs.at(row);
		for (int col = 0; col < weights[0].size(); col++) {
			weights.at(row).at(col) = weights.at(row).at(col) + change;
		}
	}

	// return previous layer's error information for their nodes
	return dotProduct(error, errorAcrossWeights);
}
