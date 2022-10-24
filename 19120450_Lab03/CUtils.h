#pragma once

#include "opencv2/opencv.hpp"
# include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define PI 3.14159265358979323846
#define e 0.0000000000001

class CUtils
{
public:
	vector<float> gaussKernel(int size, float sigma);
	pair<vector<float>, vector<float>> sobelKernel();
	float computeConvolution(vector<float> kernel, vector<uchar> data);
	vector<uchar> createEmptyVector(int size, uchar value);
};

