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
*利用Poisson Image Editing实现Dota2.jpg与Burning.jpg的无缝拼接
*i.e.将Burning.jpg无缝插入到Dota2.jpg中
*/

int main()
{
	Mat Burning, Dota2, Joint;
	Burning = imread("pic\\Burning.jpg");
	Dota2 = imread("pic\\Dota2.jpg");
	Joint = imread("pic\\Dota2.jpg");
	cout << "Burning size" << Burning.rows << "*" << Burning.cols << endl;
	cout << "Dota2 size" << Dota2.rows << "*" << Dota2.cols << endl;

	////深拷贝,Dota2复制给Joint
	//Mat temppp;
	//Dota2.copyTo(Joint);
	//imshow("dasd", Joint);

	//在Dota2（src）图中定义感兴趣的区域，注意感兴趣的大小就是Burning图的大小
	Rect ROI;
	ROI.x = 140;
	ROI.y = 800;
	ROI.height = Burning.rows;
	ROI.width = Burning.cols;
	//直接将两个图片进行拼接
	for (int i = 0; i < Burning.rows; i += 1)
	{
		for (int j = 0; j < Burning.cols; j += 1)
		{
			Joint.at<Vec3b>(ROI.x + i, ROI.y + j) = Burning.at<Vec3b>(i, j);
		}
	}
	//imshow("Burning原图", Burning);
	//imshow("Dota2原图", Dota2);
	//imshow("直接拼接图", Joint);
	//求Burning图像梯度
	Mat Burning_GradX, Burning_GradY;
	GetGradientX(Burning, Burning_GradX);
	GetGradientY(Burning, Burning_GradY);
	//求Dota2图像梯度
	Mat Dota2_GradX, Dota2_GradY;
	GetGradientX(Dota2, Dota2_GradX);
	GetGradientY(Dota2, Dota2_GradY);
	//将Burning图的梯度覆盖到Dota2图的梯度上
	//这步操作是为了方便后续处理
	Mat Cover_GradX, Cover_GradY;
	Cover(Dota2_GradX, Burning_GradX, Cover_GradX, ROI);
	Cover(Dota2_GradY, Burning_GradY, Cover_GradY, ROI);
	//imshow("AIM_GradX", Cover_GradX);
	//imshow("AIM_GradY", Cover_GradY);
	Mat Div;
	GetDivergence(Cover_GradX, Cover_GradY, Div);
	//imshow("Div", Div);
	
	//根据Poisson Image Editing论文的公式四
	//即ROI区域中，拉普拉斯算子与其中像素点作用的结果等于原图像在相应位置的梯度的散度，ROI的边界取背景图像（src）的边界
	//以上描述可能有些问题，但是不要在意这些细节了
	//对于二维离散的数据（）拉普拉斯算子的运算可以这么表示
	//result(m,n) = -4f(m,n) + f(m + 1, n) + f(m - 1, n) + f(m, n + 1) + f(m, n - 1)
	//我们就可以将公式四转换为Ax=b的方程组
	//其中A是一个大型的稀疏矩阵，每一行非零元素不超过五个，有更加高效的存储方式，这里没有细究，只是利用OpenCV的Mat来保存
	//在数学上可以使用高斯迭代或者是雅克比迭代(Jacobi method)来求解这个问题，查阅先关资料后了解到后者效率更高，所以这里就选择使用后者来求解这个问题


	//按照颜色通道分离散度(BGR)
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
	//测试Sparse_matrix
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

	//注意Jacobi Method解出来的result都是列向量，在显示的时候还要转换一下
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

	
	////融合
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
