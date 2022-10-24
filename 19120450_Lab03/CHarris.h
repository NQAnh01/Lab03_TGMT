#pragma once
#include "CUtils.h"

class CHarris
{
public:
	int gaussianBlur(const Mat& src, Mat& dst, int size, float sigma);
	pair<Mat, Mat> computeGradients(const Mat& src);

	Mat detectHarris(const Mat& src, float k, float thres);
	pair<Mat, float> computeReponseMatrix(const Mat& src, pair<Mat, Mat> gradients, float k, float thres);
	vector<pair<int, int>> coordOfLocalMaxima(const Mat& reponseMatrix, float maxReponse, float thres);
	Mat mapCornerToImage(const Mat& src, vector<pair<int, int>> cornerCoord);
};

