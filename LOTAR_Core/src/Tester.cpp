#include "Tester.h";
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <ctime> 


struct Tester::Exhibit
{
	std::vector<cv::Mat> descriptors;
	std::string name;
};
struct Tester::UnitedExhibit
{
	cv::Mat descriptor;
	std::string name;
};
struct Tester::TestUnit
{
	cv::Mat descriptor;
	std::string realName, predictedName;
	bool isPredictRight() const { return realName == predictedName; }
};
enum Tester::DescriptorsType
{
	ORB,
	SIFT,
	KAZE,
	AKAZE,
	BRISK
};

Tester::Tester(std::string& _testName, std::vector<std::string>& _trainPaths, 
	std::vector<std::string>& _testPaths, float _detectionThreshold):
	detectionThreshold(_detectionThreshold)
{
	testName = _testName;
	trainPaths = _trainPaths;
	testPaths = _testPaths;
}

void Tester::Test()
{
	try
	{
		bool pathsStatus = checkPathsValid();
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what();
		return;
	}
	std::cout << "Paths is valid\n";
	
	std::cout << "Start ORB testing\n";
	test(DescriptorsType::ORB);
	std::cout << "Finish ORB testing\n";

	std::cout << "Start BRISK testing\n";
	test(DescriptorsType::BRISK);
	std::cout << "Finish BRISK testing\n";

	std::cout << "Start AKAZE testing\n";
	test(DescriptorsType::AKAZE);
	std::cout << "Finish AKAZE testing\n";
}

bool Tester::checkPathsValid()
{
	namespace fs = std::filesystem;

	if (trainPaths.size() == 0 || testPaths.size() == 0)
		throw std::exception("train and test datasets shouldn't be empty\n");

	for (const auto& stringPath : trainPaths)
	{
		fs::path trainPath = fs::path(stringPath);
		if (!fs::exists(trainPath))
			return false;
	}

	for (const auto& stringPath : testPaths)
	{
		fs::path testPath = fs::path(stringPath);
		if (!fs::exists(testPath))
			return false;
	}

	return true;
}


void Tester::test(DescriptorsType descrType)
{
	namespace fs = std::filesystem;
	
	std::vector<Exhibit> exhibits;
	std::vector<TestUnit> testUnits;
	std::string descrTypeString;
	if (descrType == DescriptorsType::ORB)
		descrTypeString = "_ORB_log_";
	else if (descrType == DescriptorsType::SIFT)
		descrTypeString = "_SIFT_log_";
	else if (descrType == DescriptorsType::KAZE)
		descrTypeString = "_KAZE_log_";
	else if (descrType == DescriptorsType::AKAZE)
		descrTypeString = "_AKAZE_log_";
	else if (descrType == DescriptorsType::BRISK)
		descrTypeString = "_BRISK_log_";
	else
	{
		std::cout << "Incorrect description type im method test\n";
	}
	auto time_moment = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	auto date = std::ctime(&time_moment);
	std::string log_file_name = std::string("E:\\LotarCORE\\Logs\\") + testName +
		descrTypeString + std::string(date) + std::string(".txt");
	std::string::iterator it = std::remove(log_file_name.begin(), log_file_name.end(), '\n');
	log_file_name.erase(it, log_file_name.end());
	std::replace(log_file_name.begin(), log_file_name.end(), ' ', '_');
	std::replace(log_file_name.begin(), log_file_name.end(), ':', '_');
	log_file_name[1] = ':';
	std::ofstream fout(log_file_name);
	if (!fout.is_open())
	{
		std::cout << "Can not open log file\n";
		return;
	}

	std::cout << "Start load train data\n";
	//reading train data
	float train_time = readTrainData(exhibits, descrType);
	fout << std::string("Ready train exhibits time: ") + std::to_string(train_time) + std::string(" ms\n");
	std::cout << "Finish load train data\n";

	std::cout << "Start load test data\n";
	//reading test data
	float test_time = readTestData(testUnits, descrType);
	fout << std::string("Ready test units time: ") + std::to_string(test_time) + std::string(" ms\n");
	std::cout << "Start load test data\n";


	std::cout << "Start matching\n";
	//matching exhibits and every test units
	float match_time = 0;
	for (auto& test_unit : testUnits)
	{
		auto start_train_time = std::chrono::system_clock::now();
		std::pair<float, Exhibit> bestMatch = {0, Exhibit()};
		for (auto& exhibit : exhibits)
		{
			float bestCoeff = match(exhibit, test_unit, descrType);
			if (bestCoeff > bestMatch.first)
				bestMatch = { bestCoeff, exhibit };
		}
		test_unit.predictedName = bestMatch.second.name;
		auto end_train_time = std::chrono::system_clock::now();
		std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
		match_time += dura.count();
		std::cout << test_unit.realName << " has been matched\n";
	}
	fout << std::string("Matching time: ") + std::to_string(match_time) + std::string(" ms\n");
	std::cout << "Finish matching\n";

	//Writing log info to log file
	logInfo(testUnits, fout);
	fout.close();
}

float Tester::match(Exhibit& exhibit, TestUnit& test_unit, DescriptorsType descrType)
{
	float bestMatchCoeff = 0.0f;
	cv::Ptr<cv::DescriptorMatcher> matcher;
	if (descrType == DescriptorsType::ORB || descrType == DescriptorsType::AKAZE || 
	descrType == DescriptorsType::BRISK)
		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
	else
		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_L1);
	std::vector< std::vector<cv::DMatch> > knn_matches;
	const int8_t k = 2;
	for (const auto& descr : exhibit.descriptors)
	{
		matcher->knnMatch(descr, test_unit.descriptor, knn_matches, k);
		float good_matches = 0;;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < detectionThreshold * knn_matches[i][1].distance)
			{
				++good_matches;
			}
		}
		if (knn_matches.size() == 0)
			continue;
		float matchCoeff = good_matches / (float)knn_matches.size();
		if (matchCoeff > bestMatchCoeff)
			bestMatchCoeff = matchCoeff;
	}
	return bestMatchCoeff;
}

void Tester::logInfo(std::vector<TestUnit>& test_units, std::ofstream& fout)
{
	size_t counter = 1;
	int right_predictions = 0;
	for (const auto& test_unit : test_units)
	{
		std::string log_row = std::to_string(counter) + std::string(".\n");
		log_row += "\tReal name: " + test_unit.realName + "\n";
		log_row += "\tPredicted name: " + test_unit.predictedName + "\n";
		if (test_unit.isPredictRight())
		{
			++right_predictions;
			log_row += "\tPrediction was RIGHT\n";
		}
		else
			log_row += "\tPrediction was FALSE\n";
		fout << log_row;
		++counter;
	}
	fout << "\n\n\n";
	fout << "Right predictions: " << right_predictions << "\n" << "Right predictions ratio: "
		 << (float)right_predictions / (float)test_units.size() << "\n";
}


float Tester::readTrainData(std::vector<Exhibit>& exhibits, DescriptorsType descrType)
{
	namespace fs = std::filesystem;
	float train_time = 0;
	cv::Ptr<cv::Feature2D> detector;
	if (descrType == DescriptorsType::ORB)
		detector = cv::ORB::create();
	else if (descrType == DescriptorsType::SIFT)
		detector = cv::SIFT::create();
	else if (descrType == DescriptorsType::KAZE)
		detector = cv::KAZE::create();
	else if (descrType == DescriptorsType::AKAZE)
		detector = cv::AKAZE::create();
	else if (descrType == DescriptorsType::BRISK)
		detector = cv::BRISK::create();
	for (const auto& strPath : trainPaths)
	{
		fs::path path(strPath);
		for (const auto& dir : fs::directory_iterator(path))
		{
			if (!dir.is_directory())
				throw std::exception("invalid path to train files");
			std::vector<cv::Mat> descriptors;
			std::string name = dir.path().filename().string();
			//run over images of exhibit
			for (const auto& file : fs::directory_iterator(dir))
			{
				if (!file.is_regular_file())
					throw std::exception("invalid path to train files");
				cv::Mat img = cv::imread(file.path().string());
				cv::Mat grayImg;
				cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
				auto start_train_time = std::chrono::system_clock::now();
				
				std::vector<cv::KeyPoint> kp;
				cv::Mat descriptor;
				detector->detectAndCompute(grayImg, cv::noArray(), kp, descriptor);
				descriptors.push_back(descriptor);
				auto end_train_time = std::chrono::system_clock::now();
				std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
				train_time += dura.count();
			}
			//create exhibit object and add it to exhibits vector
			Exhibit exhibit{ descriptors, name };
			exhibits.push_back(exhibit);
			std::cout << "Train object " << name << " loaded\n";
		}
	}
	return train_time;
}
float Tester::readTestData(std::vector<TestUnit>& testUnits, DescriptorsType descrType)
{
	namespace fs = std::filesystem;
	float test_time = 0;
	cv::Ptr<cv::Feature2D> detector;
	if (descrType == DescriptorsType::ORB)
		detector = cv::ORB::create();
	else if (descrType == DescriptorsType::SIFT)
		detector = cv::SIFT::create();
	else if (descrType == DescriptorsType::KAZE)
		detector = cv::KAZE::create();
	else if (descrType == DescriptorsType::AKAZE)
		detector = cv::AKAZE::create();
	else if (descrType == DescriptorsType::BRISK)
		detector = cv::BRISK::create();
	for (const auto& strPath : testPaths)
	{
		fs::path path(strPath);
		for (const auto& dir : fs::directory_iterator(path))
		{
			if (!dir.is_directory())
				throw std::exception("invalid path to test files");
			cv::Mat descriptor;
			std::string name = dir.path().filename().string();
			//run over testUnits in directory
			for (const auto& file : fs::directory_iterator(dir))
			{
				if (!file.is_regular_file())
					throw std::exception("invalid path to test files");
				cv::Mat img = cv::imread(file.path().string());
				cv::Mat grayImg;
				cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
				auto start_train_time = std::chrono::system_clock::now();
				
				std::vector<cv::KeyPoint> kp;
				cv::Mat descriptor;
				detector->detectAndCompute(grayImg, cv::noArray(), kp, descriptor);
				auto end_train_time = std::chrono::system_clock::now();
				std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
				test_time += dura.count();
				//create testUnit and write data;
				TestUnit test_unit;
				test_unit.descriptor = descriptor;
				test_unit.realName = name;
				testUnits.push_back(test_unit);
			}
			std::cout << "Test object " << name << " loaded\n";
		}
	}
	return test_time;
}



float Tester::readUnitedTrainData(std::vector<UnitedExhibit>& exhibits, DescriptorsType descrType)
{
	namespace fs = std::filesystem;
	float train_time = 0;
	cv::Ptr<cv::Feature2D> detector;
	if (descrType == DescriptorsType::ORB)
		detector = cv::ORB::create();
	else if (descrType == DescriptorsType::SIFT)
		detector = cv::SIFT::create();
	else if (descrType == DescriptorsType::KAZE)
		detector = cv::KAZE::create();
	else if (descrType == DescriptorsType::AKAZE)
		detector = cv::AKAZE::create();
	else if (descrType == DescriptorsType::BRISK)
		detector = cv::BRISK::create();
	for (const auto& strPath : trainPaths)
	{
		fs::path path(strPath);
		for (const auto& dir : fs::directory_iterator(path))
		{
			if (!dir.is_directory())
				throw std::exception("invalid path to train files");
			std::string name = dir.path().filename().string();
			//run over images of exhibit
			UnitedExhibit exhibit;
			cv::Size exhibitSize;
			cv::Mat unitedImg, currentImg;
			bool isFirstImgRead = false;
			for (const auto& file : fs::directory_iterator(dir))
			{
				if (!file.is_regular_file())
					throw std::exception("invalid path to train files");
				cv::Mat currentImg = cv::imread(file.path().string());
				auto start_train_time = std::chrono::system_clock::now();
				if (!isFirstImgRead)
				{
					unitedImg = currentImg;
					exhibitSize = currentImg.size();
					isFirstImgRead = true;
				}
				else
				{
					cv::Mat resizedImg;
					cv::resize(currentImg, resizedImg, exhibitSize);
					cv::Mat concatImg;
					cv::hconcat(unitedImg, resizedImg, concatImg);
					unitedImg = concatImg;
				}
				//detector->detectAndCompute(grayImg, cv::noArray(), kp, descriptor);
				auto end_train_time = std::chrono::system_clock::now();
				std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
				train_time += dura.count();
			}
			std::vector<cv::KeyPoint> kp;
			cv::Mat descriptor;
			cv::Mat grayImg;
			auto start_train_time = std::chrono::system_clock::now();
			cv::cvtColor(unitedImg, grayImg, cv::COLOR_BGR2GRAY);
			detector->detectAndCompute(grayImg, cv::noArray(), kp, descriptor);
			exhibit.name = name;
			exhibit.descriptor = descriptor;
			exhibits.push_back(exhibit);
			auto end_train_time = std::chrono::system_clock::now();
			std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
			train_time += dura.count();
			//create exhibit object and add it to exhibits vector
			//UnitedExhibit exhibit{ descriptors, name };
			//exhibits.push_back(exhibit);
			std::cout << "Train object " << name << " loaded\n";
		}
	}
	return train_time;
}


float Tester::matchUnited(UnitedExhibit& exhibit, TestUnit& test_unit, DescriptorsType descrType)
{
	float bestMatchCoeff = 0.0f;
	cv::Ptr<cv::DescriptorMatcher> matcher;
	if (descrType == DescriptorsType::ORB || descrType == DescriptorsType::AKAZE ||
											 descrType == DescriptorsType::BRISK)
		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
	else
		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_L1);
	std::vector< std::vector<cv::DMatch> > knn_matches;
	const int8_t k = 2;
	matcher->knnMatch(exhibit.descriptor, test_unit.descriptor, knn_matches, k);
	float good_matches = 0;;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < detectionThreshold * knn_matches[i][1].distance)
		{
			++good_matches;
		}
	}
	if (knn_matches.size() == 0)
		return 0.0f;
	float matchCoeff = good_matches / (float)knn_matches.size();
	if (matchCoeff > bestMatchCoeff)
		bestMatchCoeff = matchCoeff;
	return bestMatchCoeff;
}

void Tester::testUnited(DescriptorsType descrType)
{
	namespace fs = std::filesystem;

	std::vector<UnitedExhibit> exhibits;
	std::vector<TestUnit> testUnits;
	std::string descrTypeString;
	if (descrType == DescriptorsType::ORB)
		descrTypeString = "_ORB_log_United_";
	else if (descrType == DescriptorsType::SIFT)
		descrTypeString = "_SIFT_log_United_";
	else if (descrType == DescriptorsType::KAZE)
		descrTypeString = "_KAZE_log_United_";
	else if (descrType == DescriptorsType::AKAZE)
		descrTypeString = "_AKAZE_log_United_";
	else
	{
		std::cout << "Incorrect description type im method test\n";
	}
	auto time_moment = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	auto date = std::ctime(&time_moment);
	std::string log_file_name = std::string("E:\\LotarCORE\\Logs\\") + testName +
		descrTypeString + std::string(date) + std::string(".txt");
	std::string::iterator it = std::remove(log_file_name.begin(), log_file_name.end(), '\n');
	log_file_name.erase(it, log_file_name.end());
	std::replace(log_file_name.begin(), log_file_name.end(), ' ', '_');
	std::replace(log_file_name.begin(), log_file_name.end(), ':', '_');
	log_file_name[1] = ':';
	std::ofstream fout(log_file_name);
	if (!fout.is_open())
	{
		std::cout << "Can not open log file\n";
		return;
	}

	std::cout << "Start load train data\n";
	//reading train data
	float train_time = readUnitedTrainData(exhibits, descrType);
	fout << std::string("Ready train exhibits time: ") + std::to_string(train_time) + std::string(" ms\n");
	std::cout << "Finish load train data\n";

	std::cout << "Start load test data\n";
	//reading test data
	float test_time = readTestData(testUnits, descrType);
	fout << std::string("Ready test units time: ") + std::to_string(test_time) + std::string(" ms\n");
	std::cout << "Start load test data\n";


	std::cout << "Start matching\n";
	//matching exhibits and every test units
	float match_time = 0;
	for (auto& test_unit : testUnits)
	{
		auto start_train_time = std::chrono::system_clock::now();
		std::pair<float, UnitedExhibit> bestMatch = { 0, UnitedExhibit() };
		for (auto& exhibit : exhibits)
		{
			float bestCoeff = matchUnited(exhibit, test_unit, descrType);
			if (bestCoeff > bestMatch.first)
				bestMatch = { bestCoeff, exhibit };
		}
		test_unit.predictedName = bestMatch.second.name;
		auto end_train_time = std::chrono::system_clock::now();
		std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
		match_time += dura.count();
		std::cout << test_unit.realName << " has been matched\n";
	}
	fout << std::string("Matching time: ") + std::to_string(match_time) + std::string(" ms\n");
	std::cout << "Finish matching\n";

	//Writing log info to log file
	logInfo(testUnits, fout);
	fout.close();
}



void Tester::TestUnited()
{
	try
	{
		bool pathsStatus = checkPathsValid();
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what();
		return;
	}
	std::cout << "Paths is valid\n";
	std::cout << "Start united testing\n";
	std::cout << "Start ORB testing\n";
	testUnited(DescriptorsType::ORB);
	std::cout << "Finish ORB testing\n";

	std::cout << "Start SIFT testing\n";
	//test(DescriptorsType::SIFT);
	std::cout << "Finish SIFT testing\n";

	std::cout << "Start AKAZE testing\n";
	//test(DescriptorsType::AKAZE);
	std::cout << "Finish AKAZE testing\n";
}



void Tester::testConcatDescr(DescriptorsType descrType)
{
	namespace fs = std::filesystem;

	std::vector<UnitedExhibit> exhibits;
	std::vector<TestUnit> testUnits;
	std::string descrTypeString;
	if (descrType == DescriptorsType::ORB)
		descrTypeString = "_ORB_log_United_";
	else if (descrType == DescriptorsType::SIFT)
		descrTypeString = "_SIFT_log_United_";
	else if (descrType == DescriptorsType::KAZE)
		descrTypeString = "_KAZE_log_United_";
	else if (descrType == DescriptorsType::AKAZE)
		descrTypeString = "_AKAZE_log_United_";
	else
	{
		std::cout << "Incorrect description type im method test\n";
	}
	auto time_moment = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	auto date = std::ctime(&time_moment);
	std::string log_file_name = std::string("E:\\LotarCORE\\Logs\\") + testName +
		descrTypeString + std::string(date) + std::string(".txt");
	std::string::iterator it = std::remove(log_file_name.begin(), log_file_name.end(), '\n');
	log_file_name.erase(it, log_file_name.end());
	std::replace(log_file_name.begin(), log_file_name.end(), ' ', '_');
	std::replace(log_file_name.begin(), log_file_name.end(), ':', '_');
	log_file_name[1] = ':';
	std::ofstream fout(log_file_name);
	if (!fout.is_open())
	{
		std::cout << "Can not open log file\n";
		return;
	}

	std::cout << "Start load train data\n";
	//reading train data
	float train_time = readConcatDescrTrainData(exhibits, descrType);
	fout << std::string("Ready train exhibits time: ") + std::to_string(train_time) + std::string(" ms\n");
	std::cout << "Finish load train data\n";

	std::cout << "Start load test data\n";
	//reading test data
	float test_time = readTestData(testUnits, descrType);
	fout << std::string("Ready test units time: ") + std::to_string(test_time) + std::string(" ms\n");
	std::cout << "Start load test data\n";


	std::cout << "Start matching\n";
	//matching exhibits and every test units
	float match_time = 0;
	for (auto& test_unit : testUnits)
	{
		auto start_train_time = std::chrono::system_clock::now();
		std::pair<float, UnitedExhibit> bestMatch = { 0, UnitedExhibit() };
		for (auto& exhibit : exhibits)
		{
			float bestCoeff = matchUnited(exhibit, test_unit, descrType);
			if (bestCoeff > bestMatch.first)
				bestMatch = { bestCoeff, exhibit };
		}
		test_unit.predictedName = bestMatch.second.name;
		auto end_train_time = std::chrono::system_clock::now();
		std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
		match_time += dura.count();
		std::cout << test_unit.realName << " has been matched\n";
	}
	fout << std::string("Matching time: ") + std::to_string(match_time) + std::string(" ms\n");
	std::cout << "Finish matching\n";

	//Writing log info to log file
	logInfo(testUnits, fout);
	fout.close();
}
float Tester::readConcatDescrTrainData(std::vector<UnitedExhibit>& exhibits, DescriptorsType descrType)
{
	namespace fs = std::filesystem;
	float train_time = 0;
	cv::Ptr<cv::Feature2D> detector;
	if (descrType == DescriptorsType::ORB)
		detector = cv::ORB::create();
	else if (descrType == DescriptorsType::SIFT)
		detector = cv::SIFT::create();
	else if (descrType == DescriptorsType::KAZE)
		detector = cv::KAZE::create();
	else if (descrType == DescriptorsType::AKAZE)
		detector = cv::AKAZE::create();
	else if (descrType == DescriptorsType::BRISK)
		detector = cv::BRISK::create();
	for (const auto& strPath : trainPaths)
	{
		fs::path path(strPath);
		for (const auto& dir : fs::directory_iterator(path))
		{
			if (!dir.is_directory())
				throw std::exception("invalid path to train files");
			std::string name = dir.path().filename().string();
			//run over images of exhibit
			UnitedExhibit exhibit;
			cv::Mat unitedDescr, currentDescr;
			bool isFirstImgRead = false;
			for (const auto& file : fs::directory_iterator(dir))
			{
				if (!file.is_regular_file())
					throw std::exception("invalid path to train files");
				cv::Mat currentImg = cv::imread(file.path().string());
				cv::Mat grayImg;
				cv::cvtColor(currentImg, grayImg, cv::COLOR_BGR2GRAY);
				auto start_train_time = std::chrono::system_clock::now();
				if (!isFirstImgRead)
				{
					std::vector<cv::KeyPoint> kp;
					cv::Mat descriptor;
					detector->detectAndCompute(grayImg, cv::noArray(), kp, descriptor);
					unitedDescr = descriptor;
					isFirstImgRead = true;
				}
				else
				{
					std::vector<cv::KeyPoint> kp;
					cv::Mat descriptor, concatDescr;
					detector->detectAndCompute(grayImg, cv::noArray(), kp, descriptor);
					
					cv::vconcat(unitedDescr, descriptor, concatDescr);
					unitedDescr = concatDescr;
				}
				auto end_train_time = std::chrono::system_clock::now();
				std::chrono::duration<float, std::milli> dura = end_train_time - start_train_time;
				train_time += dura.count();
			}
			exhibit.name = name;
			exhibit.descriptor = unitedDescr;
			exhibits.push_back(exhibit);
			//create exhibit object and add it to exhibits vector
			//UnitedExhibit exhibit{ descriptors, name };
			//exhibits.push_back(exhibit);
			std::cout << "Train object " << name << " loaded\n";
		}
	}
	return train_time;
}


void Tester::TestConcatDescr()
{
	try
	{
		bool pathsStatus = checkPathsValid();
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what();
		return;
	}
	std::cout << "Paths is valid\n";
	std::cout << "Start united testing\n";
	std::cout << "Start ORB testing\n";
	testConcatDescr(DescriptorsType::ORB);
	std::cout << "Finish ORB testing\n";

	std::cout << "Start SIFT testing\n";
	//test(DescriptorsType::SIFT);
	std::cout << "Finish SIFT testing\n";

	std::cout << "Start AKAZE testing\n";
	//test(DescriptorsType::AKAZE);
	std::cout << "Finish AKAZE testing\n";
}