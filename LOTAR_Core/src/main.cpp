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
    std::string test_name = "Small_Test_pictures_1_";
    std::vector<std::string> trainPaths;
    trainPaths.push_back("E:\\museum_dataset\\pictures\\train");
    std::vector<std::string> testPaths;
    testPaths.push_back("E:\\museum_dataset\\pictures\\test");
    Tester tester(test_name, trainPaths, testPaths);
    tester.Test();
    tester.TestConcatDescr();
    tester.TestUnited();
    //std::cout << cv::getBuildInformation();
    return 0;
}