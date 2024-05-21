#pragma once
#include <string>
#include <vector>


namespace cv
{
	class Mat;
}

class Tester
{

	std::vector<std::string> trainPaths, testPaths;
	const float detectionThreshold;
	std::string testName;
	struct Exhibit;
	struct TestUnit;
	enum DescriptorsType;

	struct UnitedExhibit;

public:
	Tester(std::string& _testName, std::vector<std::string>& _trainPaths,
		std::vector<std::string>& _testPaths, float _detectionThreshold = 0.7f);


	void Test();
	void TestUnited();
	void TestConcatDescr();

protected:
	bool checkPathsValid();
	
	void test(DescriptorsType descrType);
	float readTrainData(std::vector<Exhibit>& exhibits, DescriptorsType descrType);
	float readTestData(std::vector<TestUnit>& testUnits, DescriptorsType descrType);


	void testUnited(DescriptorsType descrType);
	float readUnitedTrainData(std::vector<UnitedExhibit>& exhibits, DescriptorsType descrType);

	void testConcatDescr(DescriptorsType descrType);
	float readConcatDescrTrainData(std::vector<UnitedExhibit>& exhibits, DescriptorsType descrType);

	float match(Exhibit& exhibit, TestUnit& test_unit, DescriptorsType descrType);
	float matchUnited(UnitedExhibit& exhibit, TestUnit& test_unit, DescriptorsType descrType);

	void logInfo(std::vector<TestUnit>& test_units, std::ofstream& fout);
};
