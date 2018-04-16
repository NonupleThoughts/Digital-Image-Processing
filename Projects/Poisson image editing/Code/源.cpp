/*
*	@author:Cong
*	@day:2018/4/14
*	@environment:Visual Studio 2013 + OpenCV 2.4.9
*/
#include <cv.h>
#include <highgui.h>
#include "Poisson_Image_editing.h"
//#include <stdlib.h>
#include <vector>
using namespace std;
using namespace cv;
//ROI(region of interest)

/*
*����Poisson Image Editingʵ��Dota2.jpg��Burning.jpg���޷�ƴ��
*i.e.��Burning.jpg�޷���뵽Dota2.jpg��
*/

int main()
{
	Mat Burning, Dota2, Joint;
	Burning = imread("pic\\Burning.jpg");
	Dota2 = imread("pic\\Dota2.jpg");
	Joint = imread("pic\\Dota2.jpg");
	cout << "Burning size" << Burning.rows << "*" << Burning.cols << endl;
	cout << "Dota2 size" << Dota2.rows << "*" << Dota2.cols << endl;

	////���,Dota2���Ƹ�Joint
	//Mat temppp;
	//Dota2.copyTo(Joint);
	//imshow("dasd", Joint);

	//��Dota2��src��ͼ�ж������Ȥ������ע�����Ȥ�Ĵ�С����Burningͼ�Ĵ�С
	Rect ROI;
	ROI.x = 140;
	ROI.y = 800;
	ROI.height = Burning.rows;
	ROI.width = Burning.cols;
	//ֱ�ӽ�����ͼƬ����ƴ��
	for (int i = 0; i < Burning.rows; i += 1)
	{
		for (int j = 0; j < Burning.cols; j += 1)
		{
			Joint.at<Vec3b>(ROI.x + i, ROI.y + j) = Burning.at<Vec3b>(i, j);
		}
	}
	//imshow("Burningԭͼ", Burning);
	//imshow("Dota2ԭͼ", Dota2);
	//imshow("ֱ��ƴ��ͼ", Joint);
	//��Burningͼ���ݶ�
	Mat Burning_GradX, Burning_GradY;
	GetGradientX(Burning, Burning_GradX);
	GetGradientY(Burning, Burning_GradY);
	//��Dota2ͼ���ݶ�
	Mat Dota2_GradX, Dota2_GradY;
	GetGradientX(Dota2, Dota2_GradX);
	GetGradientY(Dota2, Dota2_GradY);
	//��Burningͼ���ݶȸ��ǵ�Dota2ͼ���ݶ���
	//�ⲽ������Ϊ�˷����������
	Mat Cover_GradX, Cover_GradY;
	Cover(Dota2_GradX, Burning_GradX, Cover_GradX, ROI);
	Cover(Dota2_GradY, Burning_GradY, Cover_GradY, ROI);
	//imshow("AIM_GradX", Cover_GradX);
	//imshow("AIM_GradY", Cover_GradY);
	Mat Div;
	GetDivergence(Cover_GradX, Cover_GradY, Div);
	//imshow("Div", Div);
	
	//����Poisson Image Editing���ĵĹ�ʽ��
	//��ROI�����У�������˹�������������ص����õĽ������ԭͼ������Ӧλ�õ��ݶȵ�ɢ�ȣ�ROI�ı߽�ȡ����ͼ��src���ı߽�
	//��������������Щ���⣬���ǲ�Ҫ������Щϸ����
	//���ڶ�ά��ɢ�����ݣ���������˹���ӵ����������ô��ʾ
	//result(m,n) = -4f(m,n) + f(m + 1, n) + f(m - 1, n) + f(m, n + 1) + f(m, n - 1)
	//���ǾͿ��Խ���ʽ��ת��ΪAx=b�ķ�����
	//����A��һ�����͵�ϡ�����ÿһ�з���Ԫ�ز�����������и��Ӹ�Ч�Ĵ洢��ʽ������û��ϸ����ֻ������OpenCV��Mat������
	//����ѧ�Ͽ���ʹ�ø�˹�����������ſ˱ȵ���(Jacobi method)�����������⣬�����ȹ����Ϻ��˽⵽����Ч�ʸ��ߣ����������ѡ��ʹ�ú���������������


	//������ɫͨ������ɢ��(BGR)
	Mat Joint_R, Joint_G, Joint_B;
	vector<Mat> Joint_channels;
	split(Joint, Joint_channels);
	Joint_B = Joint_channels.at(0);
	Joint_G = Joint_channels.at(1);
	Joint_R = Joint_channels.at(2);
	//imshow("Joint", Joint);
	//waitKey(0);

	Mat Div_R, Div_G, Div_B;
	vector<Mat> Div_channels;
	split(Div, Div_channels);
	Div_B = Div_channels.at(0);
	Div_G = Div_channels.at(1);
	Div_R = Div_channels.at(2);

	Sparse_matrix A_R, A_G, A_B;
	/*
	//����Sparse_matrix
	A_R.create(2, 5);
	A_R.insert(0, 0, 1);
	A_R.insert(0, 2, 1);
	A_R.insert(1, 2, 1);
	A_R.insert(1, 4, 5);

	cout << A_R.at(0, 0) << endl;
	cout << A_R.at(0, 1) << endl;
	cout << A_R.at(0, 2) << endl;
	cout << A_R.at(1, 2) << endl;
	cout << A_R.at(1, 4) << endl;
	*/
	Mat b_R, b_G, b_B;
	calcA_b(Joint_R, Div_R, A_R, b_R, ROI);
	calcA_b(Joint_G, Div_G, A_G, b_G, ROI);
	calcA_b(Joint_B, Div_B, A_B, b_B, ROI);

	//ע��Jacobi Method�������result����������������ʾ��ʱ��Ҫת��һ��
	Mat result_col_R, result_col_G, result_col_B;
	int iteration_limit = 10;
	Solve_Jacobi(A_R, b_R, result_col_R, iteration_limit);
	Solve_Jacobi(A_G, b_G, result_col_G, iteration_limit);
	Solve_Jacobi(A_B, b_B, result_col_B, iteration_limit);

	Mat result_R, result_G, result_B;
	convert(result_col_R, result_R, ROI.height, ROI.width);
	convert(result_col_G, result_G, ROI.height, ROI.width);
	convert(result_col_B, result_B, ROI.height, ROI.width);

	vector<Mat> result_channels;
	result_channels.push_back(result_B);
	result_channels.push_back(result_G);
	result_channels.push_back(result_R);

	Mat result(Dota2.rows, Dota2.cols, CV_8UC3);
	Mat result_final;
	Dota2.copyTo(result_final);
	merge(result_channels, result);

	for (int i = 0; i < Burning.rows; i += 1)
	{
		for (int j = 0; j < Burning.cols; j += 1)
		{
			result_final.at<Vec3b>(ROI.x + i, ROI.y + j) = result.at<Vec3b>(i, j);
		}
	}
	imshow("Joint", Joint);
	imshow("result", result_final);
	imwrite("pic\\Joint.jpg", Joint);
	imwrite("pic\\result.jpg", result_final);
	
	//imshow("Div_B", Div_B);
	//imshow("Div_B", Div_B);

	
	////�ں�
	//Mat RGB(Dota2.rows, Dota2.cols, CV_8UC3);
	//Mat RGB1(Dota2.rows, Dota2.cols, CV_8UC1,Scalar(0));
	//tempsadasd.push_back(RGB1);
	//tempsadasd.push_back(Div_G);
	//tempsadasd.push_back(RGB1);
	//merge(tempsadasd, RGB);
	//imshow("DotaB", RGB);
	
	cvWaitKey(0);
	return 0;
}
