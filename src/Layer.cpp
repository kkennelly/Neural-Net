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

    vector<double> passBackError;
    double sum = 0;
    for (int row = 0; row < weights.size(); row++){
        sum = 0;
        for (int col = 0; col < weights[0].size(); col++){
            sum += error.at(col) * prevOutput.at(col) * (1 - prevOutput.at(col)) * weights.at(row).at(col);
        }
        passBackError.push_back(sum);
    }

//    cout << "Changing Weights..." << endl;
    for(int row = 0; row < weights.size(); row++){
        for (int col = 0; col < weights[0].size(); col++){
            double change = error.at(col) * prevOutput.at(col) * (1 - prevOutput.at(col)) * prevLayerOutputs.at(row) * learningRate;
            weights.at(row).at(col) -= change;
        }
    }

	// return previous layer's error information for their nodes
	return passBackError;
}

vector<double> Layer::dotProduct(vector<double> values, vector< vector<double> > matrix) {

    if (this->getId() == 1)
        return values;

    vector<double> temp;
    double running = 0;

    for (int row = 0; row < matrix.size(); row++) {
        running = 0;
        for (int col = 0; col < matrix[0].size(); col++) {
            running += matrix.at(row).at(col) * values.at(col);
        }

        temp.push_back(running);
    }

    return temp;
}
