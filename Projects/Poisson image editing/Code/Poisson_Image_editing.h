#ifndef Poission_image_editing
#define Poission_image_editing
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

//一个非常简陋的稀疏矩阵
class Sparse_matrix
{
private:
	Mat m;
	int max;
	int dimension;
public:
	/*
	sparse_matrix(int row, int num)//, int type)
	{
		m.create(row, 2 * num + 1, CV_32SC1);
		max = num;
		m.setTo(0);
	}
	*/
	void create(int row, int num)
	{
		m.create(row, 2 * num + 1, CV_32SC1);
		max = num;
		dimension = row;
		m.setTo(0);
	}
	void insert(int row, int col, int data)
	{
		if (data == 0)
			return;
		m.at<int>(row, 0) += 1;
		if (m.at<int>(row, 0) > max)
			return;
		m.at<int>(row, m.at<int>(row, 0)) = col;
		m.at<int>(row, m.at<int>(row, 0) + max) = data;
		//cout << m.at<int>(row, 0) << endl;
		
	}
	int at(int row, int col)
	{
		for (int i = 1; i <= max; i += 1)
		{
			if (m.at<int>(row, i) == col)
			{
				return (m.at<int>(row, i + max));
			}
		}
		return 0;
	}
	int size()
	{
		return dimension;
	}
};

//求梯度
void GetGradientX(Mat& Image, Mat& GradX);
void GetGradientY(Mat& Image, Mat& GradY);

//覆盖
void Cover(Mat& BackGround, Mat& ForeGround, Mat& result, Rect& ROI);

//求解散度
void GetDivergence(Mat& GradX, Mat& GradY, Mat& Div);

//计算Ax=b中的A与b
void calcA_b(Mat& Src, Mat& Div, Sparse_matrix& A, Mat& b, Rect ROI);

//利用Jacobi method 求解Ax = b
void Solve_Jacobi(Sparse_matrix& A, Mat& b, Mat& result, int iteration_limit);

//将列向量转换成row * col的矩阵
void convert(Mat& col_vector, Mat& result, int row, int col);

#endif