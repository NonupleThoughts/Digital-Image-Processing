#ifndef Poission_image_editing
#define Poission_image_editing
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

//һ���ǳ���ª��ϡ�����
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

//���ݶ�
void GetGradientX(Mat& Image, Mat& GradX);
void GetGradientY(Mat& Image, Mat& GradY);

//����
void Cover(Mat& BackGround, Mat& ForeGround, Mat& result, Rect& ROI);

//���ɢ��
void GetDivergence(Mat& GradX, Mat& GradY, Mat& Div);

//����Ax=b�е�A��b
void calcA_b(Mat& Src, Mat& Div, Sparse_matrix& A, Mat& b, Rect ROI);

//����Jacobi method ���Ax = b
void Solve_Jacobi(Sparse_matrix& A, Mat& b, Mat& result, int iteration_limit);

//��������ת����row * col�ľ���
void convert(Mat& col_vector, Mat& result, int row, int col);

#endif