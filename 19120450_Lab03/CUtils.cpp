#include "CUtils.h"

vector<float> CUtils::gaussKernel(int size, float sigma)
{
	int halfSize = size / 2;
	vector<float> kernel;

	for (int x = -halfSize; x <= halfSize; x++)
	{
		for (int y = -halfSize; y <= halfSize; y++)
		{
			kernel.push_back((1.0 / (2.0 * PI * sigma * sigma)) * exp(-((x * x * 1.0 + y * y * 1.0) / (2.0 * sigma * sigma))));
		}
	}
	return kernel;
}

pair<vector<float>, vector<float>> CUtils::sobelKernel()
{
	vector<float>kernelX;
	kernelX.push_back(-1);
	kernelX.push_back(0);
	kernelX.push_back(1);
	kernelX.push_back(-2);
	kernelX.push_back(0);
	kernelX.push_back(2);
	kernelX.push_back(-1);
	kernelX.push_back(0);
	kernelX.push_back(1);

	vector<float>kernelY;
	kernelY.push_back(1);
	kernelY.push_back(2);
	kernelY.push_back(1);
	kernelY.push_back(0);
	kernelY.push_back(0);
	kernelY.push_back(0);
	kernelY.push_back(-1);
	kernelY.push_back(-2);
	kernelY.push_back(-1);

	return make_pair(kernelX, kernelY);
}

float CUtils::computeConvolution(vector<float> kernel, vector<uchar> data)
{
	float result = 0;
	for (int i = 0; i < data.size(); i++)
	{
		result += data[i] * kernel[i];
	}
	return result;;
}

vector<uchar> CUtils::createEmptyVector(int size, uchar value)
{
	vector<uchar> data;
	for (int i = 0; i < size; i++)
	{
		data.push_back(value);
	}
	return data;
}
