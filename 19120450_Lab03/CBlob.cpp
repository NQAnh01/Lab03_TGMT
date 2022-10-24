#include "CBlob.h"

pair<uchar, Point> CBlob::localMax(vector<pair<uchar, Point>> source)
{
	pair<uchar, Point> max = source[0];
	for (int i = 1; i < source.size(); i++)
	{
		if (source[i].first > max.first)
		{
			max = source[i];
		}
	}
	return max;
}

Mat CBlob::detectBlob(const Mat& img)
{
	if (!img.data) {
		return Mat();
	}
	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

	vector<Mat> scaleSpace;
	Mat scaleImg, normImg;
	double k = 1.414;
	double sigma = 1.0;

	for (int i = 0; i <= 9; i++)
	{
		Laplacian(img_gray, scaleImg, CV_16S, 3, pow(k, i) * sigma);
		normalize(scaleImg, normImg, 0, 255, NORM_MINMAX, CV_8UC1);
		scaleSpace.push_back(normImg);
	}
	//Scale - Space size of 5x5

	int width = img_gray.cols;
	int height = img_gray.rows;
	int widthStep = img_gray.step[0];
	int nChannel = img_gray.step[1];

	vector<Vec3d> blobPoint;
	for (int y = 1; y < height - 2; y += 2)
	{
		for (int x = 1; x < width - 2; x += 2)
		{
			double max = 0.0;
			int maxIndex = 0;
			double localMax;
			Point maxLocate, localMaxLocate;

			for (int i = 0; i < scaleSpace.size(); i++)
			{

				minMaxLoc(scaleSpace[i](Range(y - 1, y + 2), Range(x - 1, x + 2)), NULL, &localMax, NULL, &localMaxLocate);
				if (localMax > max)
				{
					max = localMax;
					maxIndex = i;
					maxLocate = localMaxLocate;
				}
			}

			if (max > 200)
			{
				blobPoint.push_back({ (double)x + maxLocate.x,(double)y + maxLocate.y,pow(k,maxIndex * sigma) });
			}

		}
	}

	Mat dst = img.clone();
	for (int i = 0; i < blobPoint.size(); i++)
	{
		Point p;
		p.y = blobPoint[i][1];
		p.x = blobPoint[i][0];
		circle(dst, p, blobPoint[i][2] * 1.414, Scalar(0, 0, 255), 0);
	}

	return dst;
}



