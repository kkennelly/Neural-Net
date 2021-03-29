#include "../inc/Layer.h"
#include <typeinfo>
#include "../inc/InputLayer.h"


vector< vector<double> > Layer::transpose() {

	return transpose(weights);
}

vector <vector<double> > Layer::transpose(vector< vector<double> > matrix) {
	// initialize new matrix
	vector< vector<double> > transposed;

	for (int col = 0; col != matrix[0].size(); col++) {
		vector <double> newCol;
		for (int row = 0; row != matrix.size(); row++) {
			newCol.push_back(matrix[row][col]);
		}
		transposed.push_back(newCol);
	}

	return transposed;
}

vector<double> Layer::backPropogate(vector<double> error, vector<double> prevLayerOutputs, double learningRate) {

    vector<double> deltas;
    for (int i = 0; i < weights[0].size(); i++)
    {
        deltas.push_back(-error.at(i) * prevOutput.at(i) * (1 - prevOutput.at(i)));
    }

    for (int row = 0; row < weights.size(); row++)
    {
        //update weights
        for (int col = 0; col < weights[row].size(); col++) {
            weights.at(row).at(col) -= learningRate * prevLayerOutputs.at(row) * deltas.at(col);
        }
    }

    return (dotProduct(error, weights));
}

vector<double> Layer::dotProduct(vector<double> values, vector< vector<double> > matrix) {

    if (this->getId() == 1)
        return values;

    vector<double> temp;
    double running = 0;

    for (int row = 0; row < matrix.size(); row++) {
        running = 0;
        for (int col = 0; col < matrix[row].size(); col++) {
            running += matrix.at(row).at(col) * values.at(col);
        }
        temp.push_back(running);
    }

    return temp;
}

