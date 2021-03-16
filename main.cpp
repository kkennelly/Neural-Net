#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void prepareData(string fileName);
vector<double> constructTargetMatrix(int num, int max);

vector< vector<double> > inputs;
vector <vector <double> > answerInfo;

int main() {

	cout << "Creating neural network." << endl;
	// initialize -  set number of nodes and layers
	NeuralNetwork* net = new NeuralNetwork(3, 784, 100, 10);
	net->setLearning(0.3);

	// train 
	prepareData("/home/kate/Neural Nets/Network/data/mnist_train.csv");
    net->setTrainingData(answerInfo, inputs);
	net->trainingProcess();

	// query
	vector<double> output;
	prepareData("/home/kate/Neural Nets/Network/data/mnist_test.csv");
	for(int i = 0; i < 2; i++){
	    output = net->query(inputs.at(i));
	}

	for(int i = 0; i < output.size(); i++){
	    cout << output.at(i) << endl;
	}

}

void prepareData(string fileName){

	cout << "Preparing training data..." << endl;
	
	// get in data
	ifstream fin(fileName);

	if(!fin.is_open())
		throw runtime_error("Could not open file");
	
	string line, temp, word;
	inputs.clear();
	vector<double> answers;
		
	while (fin.good() && fin.peek() != EOF) {
		vector<double> entry;
		
		getline(fin, line);
		stringstream s(line);
		
		// get the correct answer put in our vector
		getline(s, word, ',');
		answers.push_back(stod(word));
		
		while(getline(s, word, ',')) {
			entry.push_back(stod(word));
		}
		
		// get all the image information into a vector of vectors
		inputs.push_back(entry);
	}
	
	fin.close();
	
	
	// check to make sure we are getting info right
	for (int i = 0; i < 20; i++) {
		cout << answers.at(i) << " ";
	}
	cout << endl;
	
	// change inputs from 0-225 to 0.1-1.0
	for (int i = 0; i < inputs.size(); i++){
		for(int j = 0; j < inputs.at(i).size(); j++) {
			inputs.at(i).at(j) = inputs.at(i).at(j) / 255 * 0.99 + 0.01;
		}
	}
	
	// get vertices that will be our expected response
	answerInfo.clear();
	for(int i = 0; i < answers.size(); i++){
		answerInfo.push_back(constructTargetMatrix((int)answers.at(i), 10));
	}
	
	// set the vectors in the network class
//	net->setTrainingData(answerInfo, inputs);
	
	
}

vector<double> constructTargetMatrix(int num, int max) {
	
	vector<double> answer;
	
	for(int i = 0; i < max; i++)
	{
		if (i == num)
			answer.push_back(0.99);
		else
			answer.push_back(0.01);
	}
	
	return answer;
}


