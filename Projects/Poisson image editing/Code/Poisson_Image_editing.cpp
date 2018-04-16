#include "Poisson_Image_editing.h"
#include "stdlib.h"

//求解Image水平方向的梯度
void GetGradientX(Mat& Image, Mat& GradX)
{
	Mat kernel = Mat::zeros(1, 3, CV_8S);
	kernel.at<char>(0, 2) = 1;
	kernel.at<char>(0, 1) = -1;
	GradX.create(Image.rows, Image.cols, CV_32FC3);
	filter2D(Image, GradX, CV_32FC3, kernel);
	/*
	for (int i = 0; i < GradX.rows; i += 1)
	{
		for (int j = 0; j < GradX.cols; j += 1)
		{
			Vec3f temp = GradX.at<Vec3f>(i, j);
			if (temp[0] != temp[1] || temp[0] != temp[2])
				cout << GradX.at<Vec3f>(i, j) << endl;

		}
	}
	*/
	//imshow("Gradx", GradX);	
}

//求解Image竖直方向的梯度
void GetGradientY(Mat& Image, Mat& GradY)
{
	Mat kernel = Mat::zeros(3, 1, CV_8S);
	kernel.at<char>(2, 0) = 1;
	kernel.at<char>(1, 0) = -1;
	GradY.create(Image.rows, Image.cols, CV_32FC3);
	filter2D(Image, GradY, CV_32FC3, kernel);
}

//覆盖
void Cover(Mat& BackGround, Mat& ForeGround, Mat& result, Rect& ROI)
{
	result.create(BackGround.rows, BackGround.cols, BackGround.type());
	result = BackGround.clone();
	for (int i = 0; i < ForeGround.rows; i += 1)
	{
		for (int j = 0; j < ForeGround.cols; j += 1)
		{
			result.at<Vec3f>(ROI.x + i, ROI.y + j) = ForeGround.at<Vec3f>(i, j);
		}
	}
	//imshow("BackGround", BackGround);
	//imshow("result", result);
	//waitKey(0);

}

//求散度
void GetDivergence(Mat& GradX, Mat& GradY, Mat& Div)
{
	//按照散度的公式来求
	Mat kernel_X = Mat::zeros(1, 3, CV_8S);
	kernel_X.at<char>(0, 0) = -1;
	kernel_X.at<char>(0, 1) = 1;
	filter2D(GradX, GradX, CV_32FC3, kernel_X);


	Mat kernel_Y = Mat::zeros(3, 1, CV_8S);
	kernel_Y.at<char>(0, 0) = -1;
	kernel_Y.at<char>(1, 0) = 1;
	filter2D(GradY, GradY, CV_32FC3, kernel_Y);

	Div = GradX + GradY;
}

//计算Ax=b中的A与b
void calcA_b(Mat& Src, Mat& Div, Sparse_matrix& A, Mat& b, Rect ROI)
{
	int dimension = ROI.height * ROI.width;
	//这里选择的拉普拉斯算子是四邻域的那个，见前面注释，A最多有5个非零元素
	A.create(dimension, 5);
	b.create(dimension, 1, CV_32FC1);
	//在该函数中，所有矩阵起始点都是（0,0）
	int A_row = 0;
	for (int row = 0; row < ROI.height; row += 1)
	{
		for (int col = 0; col < ROI.width; col += 1)
		{
			b.at<float>(row * ROI.width + col, 0) = Div.at<float>(ROI.x + row, ROI.y + col);

			A.insert(A_row, (row * ROI.width + col), -4);
			if (row > 0)
				A.insert(A_row, ((row - 1) * ROI.width + col), 1);
			if(row < ROI.height - 1)
				A.insert(A_row, ((row + 1) * ROI.width + col), 1);
			if (col > 0)
				A.insert(A_row, (row * ROI.width + col - 1), 1);
			if (col < ROI.width - 1)
				A.insert(A_row, (row * ROI.width + col + 1), 1);
			A_row += 1;
			if (row == 0 || row == ROI.height - 1 || col == 0 || col == ROI.width - 1)	//边界
			{
				if (row == 0)
				{
					if (col == 0)
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row - 1, ROI.y + col) - Src.at<uchar>(ROI.x + row, ROI.y + col - 1));
					}
					else if (col == ROI.width - 1)
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row - 1, ROI.y + col) - Src.at<uchar>(ROI.x + row, ROI.y + col + 1));
					}
					else
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row - 1, ROI.y + col));
					}
				}
				else if (row == ROI.height - 1)
				{
					if (col == 0)
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row + 1, ROI.y + col) - Src.at<uchar>(ROI.x + row, ROI.y + col - 1));
					}
					else if (col == ROI.width - 1)
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row + 1, ROI.y + col) - Src.at<uchar>(ROI.x + row, ROI.y + col + 1));
					}
					else
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row + 1, ROI.y + col));
					}
				}
				else
				{
					if (col == 0)
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row, ROI.y + col - 1));
					}
					else if (col == ROI.width - 1)
					{
						b.at<float>(row * ROI.width + col, 0) += (-Src.at<uchar>(ROI.x + row, ROI.y + col + 1));
					}
				}
			}
		}
	}
	//cout << A_row - dimension << endl;
	//system("pause");
	//cout << A.size() << endl;
}

//利用Jacobi method 求解Ax = b
void Solve_Jacobi(Sparse_matrix& A, Mat& b, Mat& result, int iteration_limit = 100)
{
	//iteration_limit是一个迭代上限
	int iteration_count = 0;
	int dimension = b.rows;
	Mat result_pre(dimension, 1, CV_32FC1);
	result_pre.setTo(0.0f);
	result.create(dimension, 1, CV_32FC1);
	for (iteration_count = 0; iteration_count < iteration_limit; iteration_count += 1)
	{
		for (int i = 0; i < dimension; i += 1)
		{
			float temp = 0;
			for (int j = 0; j < dimension; j += 1)
			{
				if (A.at(i, j) != 0 && i != j)
				{
					temp += ((float)A.at(i, j) * result_pre.at<float>(j, 0));
					//if ((float)A.at(i, j) != 0)
					//	cout << result_pre.at<float>(j, 0) << endl;
				}
			}
			//cout << temp << endl;

			//确定A的时候保证了A的对角线元素全不为零
			result.at<float>(i, 0) = (b.at<float>(i, 0) - temp) / (float)A.at(i, i);
			//cout << b.at<float>(i, 0) << endl;
			//cout << b.at<float>(i, 0) - temp << endl;
			//cout << (float)A.at(i, i) << endl;
			//cout << result.at<float>(i, 0) << endl;
			
			//result.copyTo(result_pre);
			//iteration_count += 1;
			//if (iteration_limit > 0 && iteration_count > iteration_limit)
			//	break;
		}
		result.copyTo(result_pre);
	}
	cout << iteration_count << endl;
}

//将列向量转换成row * col的矩阵
void convert(Mat& col_vector, Mat& result, int row, int col)
{
	result.create(row, col, CV_8UC1);
	for (int i = 0; i < row; i += 1)
	{
		for (int j = 0; j < col; j += 1)
		{
			int pos = i * col + j;
			if (col_vector.at<float>(pos, 0) > 255)
				result.at<uchar>(i, j) = 255;
			else if (col_vector.at<float>(pos, 0) < 0)
				result.at<uchar>(i, j) = 0;
			else
				result.at<uchar>(i, j) = (uchar)(col_vector.at<float>(pos, 0));
		}
	}
}