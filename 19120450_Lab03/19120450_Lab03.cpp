// 19120450_Lab03.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "CBlob.h"
#include "CDOG.h"
#include "CHarris.h"


int main(int argc, char* argv[])
{
    if (argc < 3) {
        cout << "ket thuc!";
        return -1;
    }
    string maLenh = argv[2];
    Mat src, dst;
    src = imread(argv[1], 1);
    if (atoi(maLenh.c_str()) == 1) {
        CHarris hr;
        string k = argv[3], thres = argv[4];
        dst = hr.detectHarris(src, float(atoi(k.c_str())), float(atoi(thres.c_str())));
    }
    else if (atoi(maLenh.c_str()) == 3) {
        CBlob bl;
        dst = bl.detectBlob(src);
    }
    else if (atoi(maLenh.c_str()) == 4) {
        CDOG dg;
        dst = dg.detectDOG(src);
    }
    if (!dst.data) {
        cout << "fail!";
        return -1;
    }
    namedWindow("dstImg", WINDOW_AUTOSIZE);
    imshow("dstImg", dst);
    waitKey(0);
    return 0;
}

