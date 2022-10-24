#pragma once
#include "CHarris.h"
class CBlob
{
public:
	pair<uchar, Point> localMax(vector<pair<uchar, Point>> source);
	Mat detectBlob(const Mat& img);
	
};

