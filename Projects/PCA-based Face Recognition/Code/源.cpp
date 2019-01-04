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


//TRAIN_NUM + TEST_TRAIN_NUM 不能大于10
//训练用的标签以及每个标签对应的图像数
#define LABEL_NUM 40
#define TRAIN_NUM 8

//测试用的用的标签以及每个标签对应的图像数
#define TEST_LABEL_NUM 40
#define TEST_TRAIN_NUM 2
//#define LABEL_NUM 1
//#define TRAIN_NUM 2

#define EigenFace_NUM 20

//两种分类方法，DIS是直接找最小的距离对应的标签，KNN就是最近邻算法
#define DIS
//#define KNN


using namespace std;
using namespace cv;
using namespace Eigen;

int main()
{

	vector<Mat> TrainSet;
	vector<int> TrainLabel;
	//训练用图像总数
	int TrainSetSize = LABEL_NUM * TRAIN_NUM;
	string TrainFile;
	//来自Olivette研究实验室的ORL人脸数据库，该数据集中所有图片的大小都是一样的
	//将前9张图片读入程序，最后一张用于识别
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
	//S（signed integer）
	//Mat TrainAverageImage = Mat::zeros(row, col, CV_8UC1);//CV_32FC1);
	//int temp = 0;
	//计算均值，结果为一矩阵
	Mat TrainAverageImage = Mat::zeros(row, col, CV_32FC1);

	for (int i = 0; i < row; i += 1)
	{
		for (int j = 0; j < col; j += 1)
		{
			for (int k = 0; k < TrainSetSize; k += 1)
			{
				//cout << TrainSet[k].at<uchar>(i, j) << endl;
				//cout << (int)(TrainSet[k].at<uchar>(i, j) - '0') << endl;
				//将unsigned char转换成int，并累加起来
				TrainAverageImage.at<float>(i, j) += (int)(TrainSet[k].at<uchar>(i, j));
				//temp += (int)(TrainSet[k].at<uchar>(i, j));

			}
			//TrainAverageImage.at<uchar>(i, j) = temp / TrainSetSize;
			//temp = 0;
			TrainAverageImage.at<float>(i, j) /= (float)TrainSetSize;

		}
	}
	//此处如果直接以浮点数矩阵显示的话会有问题，但是在矩阵中的数值是没有问题的
	//imshow("TrainAverageImage", TrainAverageImage.t());
	//waitKey(0);

	//偏差
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
	////以下是按照定义求解协方差矩阵
	////这样的问题是协方差矩阵的维数维数会非常的大，在本程序中接近10000*10000
	////而计算协方差矩阵的目的是为了求解其特征值以及特征向量，
	////因此这里有另一种解法，详情可以参考https://blog.csdn.net/qq_16936725/article/details/51761685
	////计算协方差矩阵
	MatrixXf TrainCovariance(row * col, row * col);
	TrainCovariance.fill(0.0f);
	TrainCovariance = TrainDeviation * TrainDeviation.adjoint();
	
#endif

#ifndef ORIGIN
	//以下为计算协方差矩阵的特征值以及特征向量的另一种方法
	//详情可以参考https://blog.csdn.net/qq_16936725/article/details/51761685
	MatrixXf TrainDeviationTDeviation(TrainSetSize, TrainSetSize);
	TrainDeviationTDeviation.fill(0.0f);

	TrainDeviationTDeviation = TrainDeviation.adjoint() * TrainDeviation;

#endif
	cout <<"求解特征值以及特征向量"<< endl;
	
	//求特征值以及特征向量
	//注意这里两种求法的特征向量与特征值的维数不同
	//用eigen函数计算特征值以及特征向量，最后得到的是一个矩阵，是由特征值与特征向量构成的矩阵
	//特征向量矩阵的每一列是特征向量
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

	//找EigenFace_NUM个特征脸（找前EigenFace_NUM大的特征值对应的特征向量）
	vector<float> TrainAimEigenValues;
	vector<int> TrainIndex;
	for (int i = 0; i < TrainEigenValues.rows(); i += 1)
	{
		float temp = TrainEigenValues(i, 0).real() * TrainEigenValues(i, 0).real() + TrainEigenValues(i, 0).imag() * TrainEigenValues(i, 0).imag();
		int j = 0;
		//排序,从大到小排序
		//找到该插入的位置（此处不会越界，临界时j < TrainAimEigenValues.size()不成立会将后面的“短路掉”）
		for (j = 0; j < TrainAimEigenValues.size() && temp < TrainAimEigenValues[j]; j += 1);
		if (TrainAimEigenValues.size() < EigenFace_NUM)
		{
			//此时一定需要插入
			TrainAimEigenValues.insert(TrainAimEigenValues.begin() + j, temp);
			TrainIndex.insert(TrainIndex.begin() + j, i);
			//输出结果为132
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
			//此时不一定需要插入
			if (j < TrainAimEigenValues.size())
			{
				//此时需要插入
				TrainAimEigenValues.insert(TrainAimEigenValues.begin() + j, temp);
				TrainIndex.insert(TrainIndex.begin() + j, i);
				//插入后多了一个元素，需要将最小的那个剔除掉
				TrainAimEigenValues.pop_back();
				TrainIndex.pop_back();
			}
			else
			{
				//这时应当是j = TrainAimEigenValues.size()，需要进一步判断
				if (temp > TrainAimEigenValues[j - 1])
				{
					//此时需要插入
					//此时需要插入
					TrainAimEigenValues.insert(TrainAimEigenValues.begin() + j, temp);
					TrainIndex.insert(TrainIndex.begin() + j, i);
					//插入后多了一个元素，需要将最小的那个剔除掉
					TrainAimEigenValues.pop_back();
					TrainIndex.pop_back();
				}
				//else //此时不需要插入，因为比最小的那个数字还小
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

	////将找到的特征脸保存起来
	////保存在文件文件中
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
	////保存成图片
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
	//	//EigenFace_NUM不会超过100也即i不会超过100
	//	//如果超过100，这里会出错，此外，如果是100以笔记本的效率，不知道跑到什么时候才能跑到这里了
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

	
	//人脸识别，利用最近邻的方法
	//读取测试集
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

	//训练数据集中的图像在特征脸上的投影得到一个权重向量
	//这个权重向量是一个列向量，保存在矩阵TrainWeight的每一列中
	MatrixXf TrainWeight(EigenFace_NUM, TrainSetSize);
	TrainWeight = TrainEigenVectors.adjoint() * TrainDeviation;
	
	//这里每个类别用TEST_TRAIN_NUM张图片作为测试
	int TestSetSize = TEST_LABEL_NUM * TEST_TRAIN_NUM;
	//测试数据集中的图像在特征脸上的投影得到一个权重向量
	//这个权重向量是一个列向量，保存在矩阵TestWeight的每一列中
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

	//给最小的距离赋初值
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
	
	//后面的算法改为寻找前KNN_K大的之后，不需要赋初值了
	////给最小的距离赋初值
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

	//寻找前KNN_K小的距离
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
			//每个图片找前KNN_K个小的数
			if (KNNNUM < KNN_K)
			{
				MinDistance(i, KNNNUM) = Distance(0, 0);
				MinDistanceIndex(i, KNNNUM) = TrainLabel[j];
				//这里需要按照大小排序
				int k = 0;
				for (k = KNNNUM - 1; k >= 0; k -= 1)
				{
					//如果新加入的比之前的要小,那么进行交换
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
				//这时候已经存储满了，有新的距离时，需要先和最大的比，之后在进行排序
				if (MinDistance(i, KNN_K - 1) > Distance(0, 0))
				{
					//新的距离比最大的距离要小，将新的距离覆盖上去
					MinDistance(i, KNN_K - 1) = Distance(0, 0);
					MinDistanceIndex(i, KNN_K - 1) = TrainLabel[j];
					//之后按照大小排序,这里的思路和KNNNUM < KNN_K条件下的程序是一致的
					int k = 0;
					for (k = (KNN_K - 1) - 1; k >= 0; k -= 1)
					{
						//如果新加入的比之前的要小,那么进行交换
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
	
	//输出测试，看看排序是否正确
	//cout << "前" << KNN_K << "小距离\n" << MinDistance << endl;

	//接下来进行投票，每个标签的初始票数都为0，但是因为每个标签至少会投给自己，所以最后每个标签的票数至少为1
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

	//输出测试，看看投票是否正确
	//cout << "标签号\n" << MinDistanceIndex << endl;
	//cout << "投票结果\n" << MinDistanceIndexVote << endl;

	//以票数最高的标签，当做分类标签，票数相同的标签，以靠前的标签（代表的距离更小）为准，保存在MinDistanceIndex(i, 0)中
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
		cout << "预测结果" << MinDistanceIndex(i, 0) << "\t实际结果" << TestLabel[i] << endl;
	}
	float rate = correct * 1.0f / TestSetSize * 100;
	cout << "正确率" << rate << "%" << endl;


	//保存平均脸
	Mat aveface;
	TrainAverageImage.convertTo(aveface, CV_8UC1);
	imwrite("TrainAverageImage.jpg", aveface);

	//保存特征脸
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
		//EigenFace_NUM不会超过100也即i不会超过100
		//如果超过100，这里会出错，此外，如果是100以笔记本的效率，不知道跑到什么时候才能跑到这里了
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