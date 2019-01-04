/*
*	@author:Cong
*	@day:2018/6/13
*	@environment:Visual Studio 2013 + OpenCV 3.4.2
*/
#include <iostream>
#include "stdlib.h"
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>


#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"


//#define IMPROVEMENT
//#define ORIGIN


//TRAIN_NUM + TEST_TRAIN_NUM ���ܴ���10
//ѵ���õı�ǩ�Լ�ÿ����ǩ��Ӧ��ͼ����
#define LABEL_NUM 40
#define TRAIN_NUM 8

//�����õ��õı�ǩ�Լ�ÿ����ǩ��Ӧ��ͼ����
#define TEST_LABEL_NUM 40
#define TEST_TRAIN_NUM 2
//#define LABEL_NUM 1
//#define TRAIN_NUM 2

#define EigenFace_NUM 20

//���ַ��෽����DIS��ֱ������С�ľ����Ӧ�ı�ǩ��KNN����������㷨
#define DIS
//#define KNN


using namespace std;
using namespace cv;
using namespace Eigen;

int main()
{

	vector<Mat> TrainSet;
	vector<int> TrainLabel;
	//ѵ����ͼ������
	int TrainSetSize = LABEL_NUM * TRAIN_NUM;
	string TrainFile;
	//����Olivette�о�ʵ���ҵ�ORL�������ݿ⣬�����ݼ�������ͼƬ�Ĵ�С����һ����
	//��ǰ9��ͼƬ����������һ������ʶ��
	for (int i = 1; i <= LABEL_NUM; i += 1)
	{
		for (int j = 1; j <= TRAIN_NUM; j += 1)
		{
			TrainFile = "att_faces\\";
			TrainFile += "s";
			//LABEL_NUM < 100
			if (i > 9)
			{
				TrainFile += char(i / 10 + 48);
				TrainFile += char(i % 10 + 48);
			}
			else
				TrainFile += char(i + 48);

			TrainFile += "\\";
			//TRAIN_NUM < 100
			if (j > 9)
			{
				TrainFile += char(j / 10 + 48);
				TrainFile += char(j % 10 + 48);
			}
			else
				TrainFile += char(j + 48);
			TrainFile += ".bmp";
			Mat temp = imread(TrainFile, CV_LOAD_IMAGE_GRAYSCALE);
			TrainSet.push_back(temp);
			TrainLabel.push_back(i);
		}
	}
	
	int row = TrainSet[0].rows;
	int col = TrainSet[0].cols;
	//S��signed integer��
	//Mat TrainAverageImage = Mat::zeros(row, col, CV_8UC1);//CV_32FC1);
	//int temp = 0;
	//�����ֵ�����Ϊһ����
	Mat TrainAverageImage = Mat::zeros(row, col, CV_32FC1);

	for (int i = 0; i < row; i += 1)
	{
		for (int j = 0; j < col; j += 1)
		{
			for (int k = 0; k < TrainSetSize; k += 1)
			{
				//cout << TrainSet[k].at<uchar>(i, j) << endl;
				//cout << (int)(TrainSet[k].at<uchar>(i, j) - '0') << endl;
				//��unsigned charת����int�����ۼ�����
				TrainAverageImage.at<float>(i, j) += (int)(TrainSet[k].at<uchar>(i, j));
				//temp += (int)(TrainSet[k].at<uchar>(i, j));

			}
			//TrainAverageImage.at<uchar>(i, j) = temp / TrainSetSize;
			//temp = 0;
			TrainAverageImage.at<float>(i, j) /= (float)TrainSetSize;

		}
	}
	//�˴����ֱ���Ը�����������ʾ�Ļ��������⣬�����ھ����е���ֵ��û�������
	//imshow("TrainAverageImage", TrainAverageImage.t());
	//waitKey(0);

	//ƫ��
	MatrixXf TrainDeviation(row * col, TrainSetSize);
	TrainDeviation.fill(0.0f);
	for (int k = 0; k < TrainSetSize; k += 1)
	{
		for (int i = 0; i < row; i += 1)
		{
			for (int j = 0; j < col; j += 1)
			{
				TrainDeviation(i * col + j, k) = (int)(TrainSet[k].at<uchar>(i, j)) * 1.0f - TrainAverageImage.at<float>(i, j);
			}
		}
	}
#ifdef ORIGIN
	////�����ǰ��ն������Э�������
	////������������Э��������ά��ά����ǳ��Ĵ��ڱ������нӽ�10000*10000
	////������Э��������Ŀ����Ϊ�����������ֵ�Լ�����������
	////�����������һ�ֽⷨ��������Բο�https://blog.csdn.net/qq_16936725/article/details/51761685
	////����Э�������
	MatrixXf TrainCovariance(row * col, row * col);
	TrainCovariance.fill(0.0f);
	TrainCovariance = TrainDeviation * TrainDeviation.adjoint();
	
#endif

#ifndef ORIGIN
	//����Ϊ����Э������������ֵ�Լ�������������һ�ַ���
	//������Բο�https://blog.csdn.net/qq_16936725/article/details/51761685
	MatrixXf TrainDeviationTDeviation(TrainSetSize, TrainSetSize);
	TrainDeviationTDeviation.fill(0.0f);

	TrainDeviationTDeviation = TrainDeviation.adjoint() * TrainDeviation;

#endif
	cout <<"�������ֵ�Լ���������"<< endl;
	
	//������ֵ�Լ���������
	//ע�����������󷨵���������������ֵ��ά����ͬ
	//��eigen������������ֵ�Լ��������������õ�����һ��������������ֵ�������������ɵľ���
	//�������������ÿһ������������
#ifdef ORIGIN
	EigenSolver<MatrixXf> es(TrainCovariance);
	MatrixXf TrainEigenVectors = es.pseudoEigenvectors();
#endif

#ifndef ORIGIN
	EigenSolver<MatrixXf> es(TrainDeviationTDeviation);
	MatrixXf TrainDTDEigenVectors = es.pseudoEigenvectors();
#endif

	MatrixXcf TrainEigenValues = es.eigenvalues();


	//float temp = TrainEigenValues(0, 0).real() * TrainEigenValues(0, 0).real() + TrainEigenValues(0, 0).imag() * TrainEigenValues(0, 0).imag();
	//TrainAimEigenValues.push_back(temp);
	//TrainIndex.push_back(0);

	//��EigenFace_NUM������������ǰEigenFace_NUM�������ֵ��Ӧ������������
	vector<float> TrainAimEigenValues;
	vector<int> TrainIndex;
	for (int i = 0; i < TrainEigenValues.rows(); i += 1)
	{
		float temp = TrainEigenValues(i, 0).real() * TrainEigenValues(i, 0).real() + TrainEigenValues(i, 0).imag() * TrainEigenValues(i, 0).imag();
		int j = 0;
		//����,�Ӵ�С����
		//�ҵ��ò����λ�ã��˴�����Խ�磬�ٽ�ʱj < TrainAimEigenValues.size()�������Ὣ����ġ���·������
		for (j = 0; j < TrainAimEigenValues.size() && temp < TrainAimEigenValues[j]; j += 1);
		if (TrainAimEigenValues.size() < EigenFace_NUM)
		{
			//��ʱһ����Ҫ����
			TrainAimEigenValues.insert(TrainAimEigenValues.begin() + j, temp);
			TrainIndex.insert(TrainIndex.begin() + j, i);
			//������Ϊ132
			//vector<int> a;
			//a.push_back(1);
			//a.push_back(2);
			//a.insert(a.begin() + 1, 3);
			//for (int i = 0; i < a.size(); i += 1)
			//	cout << a[i] << endl;
			//TrainAimEigenValues.push_back(temp);
			//TrainIndex.push_back(i);
		}
		else
		{
			//��ʱ��һ����Ҫ����
			if (j < TrainAimEigenValues.size())
			{
				//��ʱ��Ҫ����
				TrainAimEigenValues.insert(TrainAimEigenValues.begin() + j, temp);
				TrainIndex.insert(TrainIndex.begin() + j, i);
				//��������һ��Ԫ�أ���Ҫ����С���Ǹ��޳���
				TrainAimEigenValues.pop_back();
				TrainIndex.pop_back();
			}
			else
			{
				//��ʱӦ����j = TrainAimEigenValues.size()����Ҫ��һ���ж�
				if (temp > TrainAimEigenValues[j - 1])
				{
					//��ʱ��Ҫ����
					//��ʱ��Ҫ����
					TrainAimEigenValues.insert(TrainAimEigenValues.begin() + j, temp);
					TrainIndex.insert(TrainIndex.begin() + j, i);
					//��������һ��Ԫ�أ���Ҫ����С���Ǹ��޳���
					TrainAimEigenValues.pop_back();
					TrainIndex.pop_back();
				}
				//else //��ʱ����Ҫ���룬��Ϊ����С���Ǹ����ֻ�С
			}

		}
	}


#ifndef ORIGIN
	MatrixXf TrainEigenVectors(row * col, EigenFace_NUM);
	TrainEigenVectors.fill(0.0f);
	for (int i = 0; i < EigenFace_NUM; i += 1)
	{
		TrainEigenVectors.col(i) = TrainDeviation * TrainDTDEigenVectors.col(TrainIndex[i]);
	}

	for (int i = 0; i < EigenFace_NUM; i += 1)
	{
		TrainIndex[i] = i;
	}
#endif

	////���ҵ�����������������
	////�������ļ��ļ���
	//ofstream DataEigenFace("EigenFace\\EigenFace.txt");
	//for (int i = 0; i < EigenFace_NUM; i += 1)
	//{
	//	for (int j = 0; j < row * col; j += 1)
	//	{
	//		DataEigenFace << TrainEigenVectors(j, TrainIndex[i]) << "    ";
	//	}
	//	DataEigenFace << endl;
	//}
	//DataEigenFace.close();
	////�����ͼƬ
	//for (int i = 0; i < EigenFace_NUM; i += 1)
	//{
	//	string EigenFaceName = "EigenFace\\EigenFace";
	//	Mat temp = Mat::zeros(row, col, CV_8UC1);
	//	for (int j = 0; j < row; j += 1)
	//	{
	//		for (int k = 0; k < col; k += 1)
	//		{
	//			temp.at<uchar>(j, k) = TrainEigenVectors(j * col + k, TrainIndex[i]);
	//		}
	//	}
	//	//EigenFace_NUM���ᳬ��100Ҳ��i���ᳬ��100
	//	//�������100�������������⣬�����100�ԱʼǱ���Ч�ʣ���֪���ܵ�ʲôʱ������ܵ�������
	//	if (i > 9)
	//	{
	//		EigenFaceName += char(i / 10 + 48);
	//		EigenFaceName += char(i % 10 + 48);
	//	}
	//	else
	//		EigenFaceName += char(i + 48);
	//	EigenFaceName += ".bmp";
	//	imwrite(EigenFaceName, temp);
	//}

	
	//����ʶ����������ڵķ���
	//��ȡ���Լ�
	vector<Mat> TestSet;
	vector<int> TestLabel;
	string TestFile;
	for (int i = 1; i <= TEST_LABEL_NUM; i += 1)
	{
		for (int j = 1; j <= TEST_TRAIN_NUM; j += 1)
		{
			TestFile = "att_faces\\";
			TestFile += "s";
			//LABEL_NUM < 100
			if (i > 9)
			{
				TestFile += char(i / 10 + 48);
				TestFile += char(i % 10 + 48);
			}
			else
				TestFile += char(i + 48);

			TestFile += "\\";
			//TRAIN_NUM < 100
			if ((j + TRAIN_NUM) > 9)
			{
				TestFile += char((j + TRAIN_NUM) / 10 + 48);
				TestFile += char((j + TRAIN_NUM) % 10 + 48);
			}
			else
				TestFile += char((j + TRAIN_NUM) + 48);
			TestFile += ".bmp";

			Mat temp = imread(TestFile, CV_LOAD_IMAGE_GRAYSCALE);
			TestSet.push_back(temp);
			TestLabel.push_back(i);
		}
	}

	//ѵ�����ݼ��е�ͼ�����������ϵ�ͶӰ�õ�һ��Ȩ������
	//���Ȩ��������һ���������������ھ���TrainWeight��ÿһ����
	MatrixXf TrainWeight(EigenFace_NUM, TrainSetSize);
	TrainWeight = TrainEigenVectors.adjoint() * TrainDeviation;
	
	//����ÿ�������TEST_TRAIN_NUM��ͼƬ��Ϊ����
	int TestSetSize = TEST_LABEL_NUM * TEST_TRAIN_NUM;
	//�������ݼ��е�ͼ�����������ϵ�ͶӰ�õ�һ��Ȩ������
	//���Ȩ��������һ���������������ھ���TestWeight��ÿһ����
	MatrixXf TestWeight(EigenFace_NUM, TestSetSize);
	
	MatrixXf TestDeviation(row * col, TestSetSize);
	TestDeviation.fill(0.0f);
	for (int k = 0; k < TestSetSize; k += 1)
	{
		for (int i = 0; i < row; i += 1)
		{
			for (int j = 0; j < col; j += 1)
			{
				TestDeviation(i * col + j, k) = (int)(TestSet[k].at<uchar>(i, j)) * 1.0f - TrainAverageImage.at<float>(i, j);
			}
		}
	}
	TestWeight = TrainEigenVectors.adjoint() * TestDeviation;
	
	MatrixXf Deviation(EigenFace_NUM, 1);
	MatrixXf Distance(1, 1);
#ifdef DIS
	VectorXd MinDistance(TestSetSize);
	VectorXd MinDistanceIndex(TestSetSize);

	//����С�ľ��븳��ֵ
	for (int i = 0; i < TestSetSize; i += 1)
	{
		Deviation = TestWeight.col(i) - TrainWeight.col(0);
		//cout << "size" << TestWeight.rows() << "*" << TrainWeight.cols() << endl;
		//cout << "size" << Deviation.rows() << "*" << Deviation.cols() << endl;
		Distance = Deviation.adjoint() * Deviation;
		//cout << "size" << Distance.rows() << "*" << Distance.cols() << endl;
		if (Distance.rows() != 1 || Distance.cols() != 1)
		{
			cout << "error" << endl;
			return 1;
		}
		MinDistance(i) = Distance(0, 0);
		MinDistanceIndex(i) = TrainLabel[0];
	}


	for (int i = 0; i < TestSetSize; i += 1)
	{
		for (int j = 1; j < TrainSetSize; j += 1)
		{
			Deviation = TestWeight.col(i) - TrainWeight.col(j);
			Distance = Deviation.adjoint() * Deviation;
			//cout << "size" << Distance.rows() << "*" << Distance.cols() << endl;
			if (Distance.rows() != 1 || Distance.cols() != 1)
			{
				cout << "error" << endl;
				return 1;
			}
			if (MinDistance(i) > Distance(0, 0))
			{
				MinDistance(i) = Distance(0, 0);
				MinDistanceIndex(i) = TrainLabel[j];
			}
		}
	}
#endif 
#ifdef KNN
#define KNN_K 3

	MatrixXf MinDistance(TestSetSize, KNN_K);
	MatrixXi MinDistanceIndex(TestSetSize, KNN_K);
	
	//������㷨��ΪѰ��ǰKNN_K���֮�󣬲���Ҫ����ֵ��
	////����С�ľ��븳��ֵ
	//for (int i = 0; i < TestSetSize; i += 1)
	//{
	//	Deviation = TestWeight.col(i) - TrainWeight.col(0);
	//	//cout << "size" << TestWeight.rows() << "*" << TrainWeight.cols() << endl;
	//	//cout << "size" << Deviation.rows() << "*" << Deviation.cols() << endl;
	//	Distance = Deviation.adjoint() * Deviation;
	//	//cout << "size" << Distance.rows() << "*" << Distance.cols() << endl;
	//	if (Distance.rows() != 1 || Distance.cols() != 1)
	//	{
	//		cout << "error" << endl;
	//		return 1;
	//	}
	//	MinDistance(i, 0) = Distance(0, 0);
	//	MinDistanceIndex(i, 0) = TrainLabel[0];
	//}

	//Ѱ��ǰKNN_KС�ľ���
	for (int i = 0; i < TestSetSize; i += 1)
	{
		int KNNNUM = 0;
		for (int j = 1; j < TrainSetSize; j += 1)
		{
			Deviation = TestWeight.col(i) - TrainWeight.col(j);
			Distance = Deviation.adjoint() * Deviation;
			//cout << "size" << Distance.rows() << "*" << Distance.cols() << endl;
			if (Distance.rows() != 1 || Distance.cols() != 1)
			{
				cout << "error" << endl;
				return 1;
			}
			//ÿ��ͼƬ��ǰKNN_K��С����
			if (KNNNUM < KNN_K)
			{
				MinDistance(i, KNNNUM) = Distance(0, 0);
				MinDistanceIndex(i, KNNNUM) = TrainLabel[j];
				//������Ҫ���մ�С����
				int k = 0;
				for (k = KNNNUM - 1; k >= 0; k -= 1)
				{
					//����¼���ı�֮ǰ��ҪС,��ô���н���
					if (MinDistance(i, k) > MinDistance(i, k + 1))
					{
						float temp = MinDistance(i, k);
						MinDistance(i, k) = MinDistance(i, k + 1);
						MinDistance(i, k + 1) = temp;

						int tempIndex = MinDistanceIndex(i, k);
						MinDistanceIndex(i, k) = MinDistanceIndex(i, k + 1);
						MinDistanceIndex(i, k + 1) = tempIndex;
					}
				}

				//if (MinDistance(i) > Distance(0, 0))
				//{
				//	MinDistance(i) = Distance(0, 0);
				//	MinDistanceIndex(i) = TrainLabel[j];
				//}

				KNNNUM += 1;
				//cout << KNNNUM << endl;
			}
			else
			{
				//��ʱ���Ѿ��洢���ˣ����µľ���ʱ����Ҫ�Ⱥ����ıȣ�֮���ڽ�������
				if (MinDistance(i, KNN_K - 1) > Distance(0, 0))
				{
					//�µľ�������ľ���ҪС�����µľ��븲����ȥ
					MinDistance(i, KNN_K - 1) = Distance(0, 0);
					MinDistanceIndex(i, KNN_K - 1) = TrainLabel[j];
					//֮���մ�С����,�����˼·��KNNNUM < KNN_K�����µĳ�����һ�µ�
					int k = 0;
					for (k = (KNN_K - 1) - 1; k >= 0; k -= 1)
					{
						//����¼���ı�֮ǰ��ҪС,��ô���н���
						if (MinDistance(i, k) > MinDistance(i, k + 1))
						{
							float temp = MinDistance(i, k);
							MinDistance(i, k) = MinDistance(i, k + 1);
							MinDistance(i, k + 1) = temp;

							int tempIndex = MinDistanceIndex(i, k);
							MinDistanceIndex(i, k) = MinDistanceIndex(i, k + 1);
							MinDistanceIndex(i, k + 1) = tempIndex;
						}
					}
				}
			}
		}
	}
	
	//������ԣ����������Ƿ���ȷ
	//cout << "ǰ" << KNN_K << "С����\n" << MinDistance << endl;

	//����������ͶƱ��ÿ����ǩ�ĳ�ʼƱ����Ϊ0��������Ϊÿ����ǩ���ٻ�Ͷ���Լ����������ÿ����ǩ��Ʊ������Ϊ1
	MatrixXi MinDistanceIndexVote(TestSetSize, KNN_K);
	MinDistanceIndexVote.fill(0);
	for (int i = 0; i < TestSetSize; i += 1)
	{
		for (int j = 0; j < KNN_K; j += 1)
		{
			for (int k = 0; k < KNN_K; k += 1)
			{
				if (MinDistanceIndex(i, j) == MinDistanceIndex(i, k))
				{
					MinDistanceIndexVote(i, j) += 1;
				}
			}
		}
	}

	//������ԣ�����ͶƱ�Ƿ���ȷ
	//cout << "��ǩ��\n" << MinDistanceIndex << endl;
	//cout << "ͶƱ���\n" << MinDistanceIndexVote << endl;

	//��Ʊ����ߵı�ǩ�����������ǩ��Ʊ����ͬ�ı�ǩ���Կ�ǰ�ı�ǩ������ľ����С��Ϊ׼��������MinDistanceIndex(i, 0)��
	for (int i = 0; i < TestSetSize; i += 1)
	{
		int tempMax = MinDistanceIndexVote(i, 0);
		int tempLabel = MinDistanceIndex(i, 0);
		for (int j = 0; j < KNN_K; j += 1)
		{
			if (tempMax < MinDistanceIndexVote(i, j))
			{
				tempMax = MinDistanceIndexVote(i, j);
				tempLabel = MinDistanceIndex(i, j);
			}
		}
		MinDistanceIndex(i, 0) = tempLabel;
	}
#endif
	int correct = 0;
	for (int i = 0; i < TestSetSize; i += 1)
	{
		if (MinDistanceIndex(i, 0) == TestLabel[i])
			correct += 1;
		cout << "Ԥ����" << MinDistanceIndex(i, 0) << "\tʵ�ʽ��" << TestLabel[i] << endl;
	}
	float rate = correct * 1.0f / TestSetSize * 100;
	cout << "��ȷ��" << rate << "%" << endl;


	//����ƽ����
	Mat aveface;
	TrainAverageImage.convertTo(aveface, CV_8UC1);
	imwrite("TrainAverageImage.jpg", aveface);

	//����������
	for (int i = 0; i < EigenFace_NUM; i += 1)
	{
		string EigenFaceName = "EigenFace\\EigenFace";
		Mat temp = Mat::zeros(row, col, CV_8UC1);
		for (int j = 0; j < row; j += 1)
		{
			for (int k = 0; k < col; k += 1)
			{
				temp.at<uchar>(j, k) = TrainEigenVectors(j * col + k, TrainIndex[i]);
			}
		}
		//EigenFace_NUM���ᳬ��100Ҳ��i���ᳬ��100
		//�������100�������������⣬�����100�ԱʼǱ���Ч�ʣ���֪���ܵ�ʲôʱ������ܵ�������
		if (i > 9)
		{
			EigenFaceName += char(i / 10 + 48);
			EigenFaceName += char(i % 10 + 48);
		}
		else
			EigenFaceName += char(i + 48);
		EigenFaceName += ".bmp";
		imwrite(EigenFaceName, temp);
	}


	system("pause");


	/*	
	Mat temp = Mat::zeros(row, col, CV_8UC1);
	for (int i = 0; i < row; i += 1)
	{
		for (int j = 0; j < col; j += 1)
		{
			temp.at<uchar>(i, j) = TrainAverageImage.at<float>(i,j);
		}
	}
	imshow("111", temp);
	waitKey(0);
	*/
	

	


	//int predictRight = 0, predictWrong = 0;
	//int modelChoose;
	//Ptr<FaceRecognizer> model;
	//model = createEigenFaceRecognizer(80);
	//model->train(images, labels);
	//system("pause");
	//while (1)
	//{
	//	predictRight = 0;
	//	predictWrong = 0;
	//	model.release();
	//	cout << "choose model:" << endl;
	//	cout << "1 EigenFace" << endl;
	//	cout << "2 Fisher" << endl;
	//	cout << "3 LBPH" << endl;
	//	cin >> modelChoose;
	//	switch (modelChoose)
	//	{
	//	case 1:model = createEigenFaceRecognizer(); break;
	//	case 2:model = createFisherFaceRecognizer(); break;
	//	case 3:model = createLBPHFaceRecognizer(); break;
	//	default:model = createLBPHFaceRecognizer(); break;
	//	}
	//	model->train(images, labels);

	//	for (int i = 1; i <= 40; i++)
	//	{

	//		TrainFile = "att_faces/";
	//		////itoa(i, tempdir, 10);
	//		TrainFile += "s";
	//		TrainFile += tempdir;
	//		TrainFile += "/";
	//		TrainFile += "10.bmp";
	//		cout << TrainFile << endl;
	//		Mat image = imread(TrainFile, CV_LOAD_IMAGE_GRAYSCALE);
	//		int predict = model->predict(image);
	//		if (predict == i)predictRight++;
	//		else predictWrong++;
	//		cout << predict << endl;
	//	}
	//	cout << "right:" << predictRight << endl;
	//	cout << "wrong:" << predictWrong << endl;
	//}

	return 0;
}