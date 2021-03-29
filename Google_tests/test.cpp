#define protected public
#include "../inc/InputLayer.h"
#include "../src/InputLayer.cpp"
#include "../inc/OutputLayer.h"
#include "../src/OutputLayer.cpp"
#include "../inc/HiddenLayer.h"
#include "../src/HiddenLayer.cpp"
#include "../inc/Layer.h"
#include "../src/Layer.cpp"
#include "../inc/NeuralNetwork.h"
#include "../src/NeuralNetwork.cpp"

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
    EXPECT_GE((VAL), (MIN));           \
    EXPECT_LE((VAL), (MAX))

#include <gtest/gtest.h>

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

TEST(LayerConstruction, InputLayerTest) {
	Layer* test = new InputLayer(5,5);

	EXPECT_EQ(5, test->weights.size());
	EXPECT_EQ(5, test->weights[0].size());

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			EXPECT_EQ(1, test->weights.at(0).at(0));
		}
	}

	delete test;
}

TEST(LayerConstruction, OutputLayerTest) {
	Layer* test = new OutputLayer(5,5);

	EXPECT_EQ(5, test->weights.size());
	EXPECT_EQ(5, test->weights[0].size());

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			EXPECT_IN_RANGE(test->weights.at(0).at(0), -0.5, 0.5);
		}
	}

	delete test;
}

TEST(LayerConstruction, HiddenLayerTest) {
	Layer* test = new HiddenLayer(5, 5);

	EXPECT_EQ(5, test->weights.size());
	EXPECT_EQ(5, test->weights[0].size());

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			EXPECT_IN_RANGE(test->weights.at(0).at(0), -0.5, 0.5);
		}
	}

	delete test;
}

TEST(SigmoidFunction, InputLayerTest) {
	Layer* test = new InputLayer(3, 3);

	vector<double> testVector{ 0.020142, 0.000125, 0.982312 };
	vector<double> result = test->sigmoid(testVector, 1);

	for (int i = 0; i < 3; i++) {
		EXPECT_EQ(testVector.at(i), result.at(i));
	}

	delete test;
}

TEST(SigmoidFunction, OutputLayerTest) {
	Layer* test = new OutputLayer(3,3);

	vector<double> testVector{ 0.020142, 0.000125, 0.982312 };
	vector<double> result = test->sigmoid(testVector, 1);

	EXPECT_NEAR(0.50503532, result.at(0), 0.00001);
	EXPECT_NEAR(0.50003124, result.at(1), 0.00001);
	EXPECT_NEAR(0.72756672, result.at(2), 0.00001);

	delete test;
}

TEST(SigmoidFunction, HiddenLayerTest) {
	Layer* test = new HiddenLayer(3,3);

	vector<double> testVector{ 0.020142, 0.000125, 0.982312 };
	vector<double> result = test->sigmoid(testVector, 1);

	EXPECT_NEAR(0.50503532, result.at(0), 0.00001);
	EXPECT_NEAR(0.50003124, result.at(1), 0.00001);
	EXPECT_NEAR(0.72756672, result.at(2), 0.00001);

	delete test;
}

TEST(DotProductFunction, InputLayerTest) {
	InputLayer* test = new InputLayer(3,3);

	vector<double> testVector{ 0.020142, 0.000125, 0.982312 };
	vector<double> result = test->dotProduct(testVector);

	for (int i = 0; i < 3; i++) {
		EXPECT_EQ(testVector.at(i), result.at(i));
	}
	
	delete test;
}

TEST(DotProductFunction, HiddenAndOutputLayerTest) {
	OutputLayer* test = new OutputLayer(3,3);

	vector<vector<double>> testWeights;
	vector<double> row1{ 0.032654, 0.168462, 0.164215 };
	vector<double> row2{ 0.165465, 0.165421, 0.894325 };
	vector<double> row3{ 0.000000, 0.054159, 0.546545 };
	testWeights.push_back(row1);
	testWeights.push_back(row2);
	testWeights.push_back(row3);
	test->weights = testWeights;

	vector<double> testVector{ 0.020142, 0.000125, 0.982312 };
	vector<double> result = test->dotProduct(testVector, test->weights);

	EXPECT_NEAR(0.1619891, result.at(0), 0.000001);
	EXPECT_NEAR(0.8818596, result.at(1), 0.000001);
	EXPECT_NEAR(0.5368844, result.at(2), 0.000001);

	delete test;
}

TEST(TransposeFunction, AllLayersTest) {
	OutputLayer* test = new OutputLayer(3,3);

	vector<vector<double>> testWeights;
	vector<double> row1{ 0.032654, 0.168462, 0.164215 };
	vector<double> row2{ 0.165465, 0.165421, 0.894325 };
	vector<double> row3{ 0.000000, 0.054159, 0.546545 };
	testWeights.push_back(row1);
	testWeights.push_back(row2);
	testWeights.push_back(row3);
	test->weights = testWeights;

	vector<vector<double>> result = test->transpose();

	EXPECT_EQ(0.032654, result.at(0).at(0));
	EXPECT_EQ(0.165465, result.at(0).at(1));
	EXPECT_EQ(0.000000, result.at(0).at(2));

	EXPECT_EQ(0.168462, result.at(1).at(0));
	EXPECT_EQ(0.165421, result.at(1).at(1));
	EXPECT_EQ(0.054159, result.at(1).at(2));

	EXPECT_EQ(0.164215, result.at(2).at(0));
	EXPECT_EQ(0.894325, result.at(2).at(1));
	EXPECT_EQ(0.546545, result.at(2).at(2));

	delete test;
}

TEST(BackPropagateFunction, Test1) {
	NeuralNetwork *net = new NeuralNetwork(3, 3, 2, 2);

	vector<double> firstrow1 {0.1, 0.2};
	vector<double> firstrow2 {0.3, 0.4};
	vector<double> firstrow3 {0.5, 0.6};
	vector< vector <double> > firstWeights;
	firstWeights.push_back(firstrow1);
	firstWeights.push_back(firstrow2);
	firstWeights.push_back(firstrow3);
	net->layers[1]->weights = firstWeights;

	vector<double> secrow1{0.7, 0.8};
	vector<double> secrow2{0.9, 0.1};
    vector< vector <double> > secondWeights;
    secondWeights.push_back(secrow1);
    secondWeights.push_back(secrow2);
    net-> layers[2]->weights = secondWeights;

    net->setSigmoidGain(1.0);
    net->setLearning(0.1);

    vector<double> input {1, 4, 5};

    net->query(input);

    vector <double> hiddenOutput = net->layers[1]->prevOutput;

    EXPECT_NEAR(0.978118, hiddenOutput.at(0), 0.01);
    EXPECT_NEAR(0.991837, hiddenOutput.at(1), 0.01);

    vector <double> outputOutput = net->layers[2]->prevOutput;

    EXPECT_NEAR(0.828826, outputOutput.at(0), 0.0001);
    EXPECT_NEAR(0.707169, outputOutput.at(1), 0.0001);

    vector<double> target {0.1, 0.05};

    vector<double> outputOneError {target.at(0) - outputOutput.at(0), target.at(1) - outputOutput.at(1)};
    vector<double> hiddenError = net->layers[2]->backPropogate(outputOneError, hiddenOutput, net->getLearning());

    EXPECT_NEAR(0.7 - 0.1 * 0.101138, net->layers[2]->weights.at(0).at(0), 0.01);
    EXPECT_NEAR(0.8 - 0.1 * 0.133109, net->layers[2]->weights.at(0).at(1), 0.01);
    EXPECT_NEAR(0.9 - 0.1 * 0.102557, net->layers[2]->weights.at(1).at(0), 0.01);
    EXPECT_NEAR(0.1 - 0.1 * 0.134976, net->layers[2]->weights.at(1).at(1), 0.01);

}



#undef protected
