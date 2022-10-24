#include "CHarris.h"
CUtils util;
//khử nhiễu bằng gauss
int CHarris::gaussianBlur(const Mat& src, Mat& dst, int size, float sigma)
{
	
	if (!src.data)
	{
		return 0;
	}

	int width = src.cols;
	int height = src.rows;
	int widthStep = src.step[0];
	int nChannel = src.step[1];

	dst = Mat(height, width, src.type());

	uchar* pData = (uchar*)src.data;
	uchar* dstData = (uchar*)dst.data;
	
	vector<float> kernel = util.gaussKernel(size, sigma);

	for (int y = 2; y < height - 2; y++)
	{
		for (int x = 2; x < width - 2; x++)
		{
			if (nChannel == 1) {
				vector<uchar>data = util.createEmptyVector(kernel.size(), 0);
				float value;

				data[0] = (uchar)pData[(y - 2) * widthStep + (x - 2) * nChannel];
				data[1] = (uchar)pData[(y - 2) * widthStep + (x - 1) * nChannel];
				data[2] = (uchar)pData[(y - 2) * widthStep + (x)*nChannel];
				data[3] = (uchar)pData[(y - 2) * widthStep + (x + 1) * nChannel];
				data[4] = (uchar)pData[(y - 2) * widthStep + (x + 2) * nChannel];

				data[5] = (uchar)pData[(y - 1) * widthStep + (x - 2) * nChannel];
				data[6] = (uchar)pData[(y - 1) * widthStep + (x - 1) * nChannel];
				data[7] = (uchar)pData[(y - 1) * widthStep + (x)*nChannel];
				data[8] = (uchar)pData[(y - 1) * widthStep + (x + 1) * nChannel];
				data[9] = (uchar)pData[(y - 1) * widthStep + (x + 2) * nChannel];

				data[10] = (uchar)pData[(y)*widthStep + (x - 2) * nChannel];
				data[11] = (uchar)pData[(y)*widthStep + (x - 1) * nChannel];
				data[12] = (uchar)pData[(y)*widthStep + (x)*nChannel];
				data[13] = (uchar)pData[(y)*widthStep + (x + 1) * nChannel];
				data[14] = (uchar)pData[(y)*widthStep + (x + 2) * nChannel];

				data[15] = (uchar)pData[(y + 1) * widthStep + (x - 2) * nChannel];
				data[16] = (uchar)pData[(y + 1) * widthStep + (x - 1) * nChannel];
				data[17] = (uchar)pData[(y + 1) * widthStep + (x)*nChannel];
				data[18] = (uchar)pData[(y + 1) * widthStep + (x + 1) * nChannel];
				data[19] = (uchar)pData[(y + 1) * widthStep + (x + 2) * nChannel];

				data[20] = (uchar)pData[(y + 2) * widthStep + (x - 2) * nChannel];
				data[21] = (uchar)pData[(y + 2) * widthStep + (x - 1) * nChannel];
				data[22] = (uchar)pData[(y + 2) * widthStep + (x)*nChannel];
				data[23] = (uchar)pData[(y + 2) * widthStep + (x + 1) * nChannel];
				data[24] = (uchar)pData[(y + 2) * widthStep + (x + 2) * nChannel];


				value = util.computeConvolution(kernel, data);
				dstData[y * widthStep + x * nChannel] = saturate_cast<uchar>(value);

			}
		}
	}

}

//tính gradient đạo hàm theo 2 hướng x, y bằng sobel
pair<Mat, Mat> CHarris::computeGradients(const Mat& src)
{
	int width = src.cols;
	int height = src.rows;
	int widthStep = src.step[0];
	int nChannel = src.step[1];

	Mat gradientX = Mat(height, width, src.type());
	Mat gradientY = Mat(height, width, src.type());

	uchar* pData = (uchar*)src.data;
	uchar* xData = (uchar*)gradientX.data;
	uchar* yData = (uchar*)gradientY.data;
	//3x3
	pair<vector<float>, vector<float>>kernel = util.sobelKernel();

	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			if (nChannel == 1) {
				vector<uchar>data = util.createEmptyVector(kernel.first.size(), 0);
				float Gx, Gy, angle;
				data[0] = (uchar)pData[(y - 1) * widthStep + (x - 1) * nChannel];
				data[1] = (uchar)pData[(y - 1) * widthStep + x * nChannel];
				data[2] = (uchar)pData[(y - 1) * widthStep + (x + 1) * nChannel];

				data[3] = (uchar)pData[y * widthStep + (x - 1) * nChannel];
				data[4] = (uchar)pData[y * widthStep + x * nChannel];
				data[5] = (uchar)pData[y * widthStep + (x + 1) * nChannel];

				data[6] = (uchar)pData[(y + 1) * widthStep + (x - 1) * nChannel];
				data[7] = (uchar)pData[(y + 1) * widthStep + x * nChannel];
				data[8] = (uchar)pData[(y + 1) * widthStep + (x + 1) * nChannel];

				Gx = util.computeConvolution(kernel.first, data);
				Gy = util.computeConvolution(kernel.second, data);

				xData[y * widthStep + x * nChannel] = saturate_cast<uchar>(Gx);
				yData[y * widthStep + x * nChannel] = saturate_cast<uchar>(Gy);
			}

		}
	}


	return make_pair(gradientX, gradientY);
}

Mat CHarris::detectHarris(const Mat& src, float k, float thres)
{
	if (!src.data)
	{
		return Mat();
	}
	Mat src_gray;
	//chuyển từ RGB sang grayscale
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	int width = src_gray.cols;
	int height = src_gray.rows;
	int widthStep = src_gray.step[0];
	int nChannel = src_gray.step[1];
	//khử nhiễu 
	Mat blurImage;
	if (!gaussianBlur(src_gray, blurImage, 5, 1.0))
	{
		return Mat();
	}
	//đạo hàm theo x, y bằng sobel
	pair<Mat, Mat>gradients = computeGradients(blurImage);
	//tính độ tương quan giữa các đạo hàm và tính corner response
	pair<Mat, float> reponseMatrix = computeReponseMatrix(blurImage, gradients, k, thres);
	//Threshhold và lấy giá trị max ở mỗi vùng. Ở bước này sau khi threshhold, chúng ta sẽ lấy 1 cửa sổ (3x3) trượt qua từng pixel của R và lấy giá trị max ở mỗi vùng.
	vector<pair<int, int>> cornerCoord = coordOfLocalMaxima(reponseMatrix.first, reponseMatrix.second, thres);
	Mat dst = mapCornerToImage(src, cornerCoord);
	return dst;
}

//tính độ tương quan giữa các đạo hàm và tính corner response
pair<Mat, float> CHarris::computeReponseMatrix(const Mat& src, pair<Mat, Mat> gradients, float k, float thres)
{
	int width = src.cols;
	int height = src.rows;
	int widthStep = src.step[0];
	int nChannel = src.step[1];
	int halfKernel = 5 / 2;

	uchar* gradientXData = (uchar*)gradients.first.data;
	uchar* gradientYData = (uchar*)gradients.second.data;

	vector<float> window = util.gaussKernel(5, 1.0);
	Mat reponseMatrix = Mat::zeros(height, width, CV_32FC1);
	float* reponseData = (float*)reponseMatrix.data;

	int reponseChannel = reponseMatrix.step[1];
	float det, trace, matrixXX, matrixYY, matrixXY, reponse, maxReponse = 0.0;

	for (int y = halfKernel; y < height - halfKernel; y++)
	{
		for (int x = halfKernel; x < width - halfKernel; x++)
		{
			matrixXX = 0.0; matrixYY = 0.0; matrixXY = 0.0;

			vector<float> Gx;
			vector<float> Gy;

			for (int i = -halfKernel; i <= halfKernel; i++)
			{
				for (int j = -halfKernel; j <= halfKernel; j++)
				{
					Gx.push_back(gradientXData[(y + i) * widthStep + (x + j) * nChannel]);
					Gy.push_back(gradientYData[(y + i) * widthStep + (x + j) * nChannel]);
				}
			}
			for (int i = 0; i < window.size(); i++)
			{
				matrixXX += (Gx[i] * window[i]) * (Gx[i] * window[i]);
				matrixYY += (Gy[i] * window[i]) * (Gy[i] * window[i]);
				matrixXY += (Gx[i] * window[i]) * (Gy[i] * window[i]);
			}

			det = matrixXX * matrixYY - matrixXY * matrixXY;
			trace = matrixXX + matrixYY;

			reponse = det - k * (trace * trace);


			reponseMatrix.at<float>(y, x) = reponse;

			if (maxReponse < reponse)
			{
				maxReponse = reponse;
			}
		}
	}

	return make_pair(reponseMatrix, maxReponse);
}

//Threshhold và lấy giá trị max ở mỗi vùng. Ở bước này sau khi threshhold, chúng ta sẽ lấy 1 cửa sổ (3x3) trượt qua từng pixel của R và lấy giá trị max ở mỗi vùng.
vector<pair<int, int>> CHarris::coordOfLocalMaxima(const Mat& reponseMatrix, float maxReponse, float thres)
{
	vector<pair<int, int>> cornerCoord;

	int width = reponseMatrix.cols;
	int height = reponseMatrix.rows;
	int widthStep = reponseMatrix.step[0];
	int nChannel = reponseMatrix.step[1];
	int halfSize = 5 / 2;
	//ngưỡng
	thres = (thres / 255.0) * maxReponse;

	for (int y = halfSize; y < height - halfSize; y++)
	{
		for (int x = halfSize; x < width - halfSize; x++)
		{

			float max = reponseMatrix.at<float>(y, x);
			pair<int, int> maxCoord = make_pair(y, x);
			for (int i = -halfSize; i <= halfSize; i++)
			{
				for (int j = -halfSize; j <= halfSize; j++)
				{
					/*Get coord of local Maxima*/
					float value = reponseMatrix.at<float>(y + i, x + j);
					float test = value - max;
					if (test > e)
					{
						max = value;
						maxCoord = make_pair((y + i), (x + j));
					}

				}
			}
			//lớn hơn ngưỡng thì lấy
			if (max >= thres) {
				cornerCoord.push_back(maxCoord);
			}
		}
	}

	return cornerCoord;
}

Mat CHarris::mapCornerToImage(const Mat& src, vector<pair<int, int>> cornerCoord)
{
	int width = src.cols;
	int height = src.rows;
	sort(cornerCoord.begin(), cornerCoord.end());
	cornerCoord.erase(unique(cornerCoord.begin(), cornerCoord.end()));

	for (int i = 0; i < cornerCoord.size(); i++)
	{
		circle(src, Point(cornerCoord[i].second, cornerCoord[i].first), 3, Scalar(255, 0, 0), 1.2, 8, 0);
	}

	return src;
}
