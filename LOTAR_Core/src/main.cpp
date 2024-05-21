#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "Tester.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
//! [includes]

int main()
{
    std::string test_name = "DebugUsualTest";
    std::vector<std::string> trainPaths;
    trainPaths.push_back("E:\\museum_dataset\\fast_test\\train");
    std::vector<std::string> testPaths;
    testPaths.push_back("E:\\museum_dataset\\fast_test\\test");
    Tester tester(test_name, trainPaths, testPaths);
    tester.TestConcatDescr();
    //std::cout << cv::getBuildInformation();
    return 0;
}