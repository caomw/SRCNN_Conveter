#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat SRCNN(Mat lIm, Mat biases_conv1, Mat biases_conv2, Mat biases_conv3, \
	Mat weights_conv1, vector<Mat> weights_conv2, Mat weights_conv3);
int main()
{
	int up_scale = 2;
	//Initialize All Parameter
	Mat biases_conv1, biases_conv2, biases_conv3, weights_conv1, weights_conv3;
	vector<Mat> weights_conv2(32);
	Mat im, lIm, hIm;

	//ModelPath
	string ModelPath = "Data";
	//GetModel
#pragma region Get Model
	FileStorage fs1(ModelPath + "/biases_conv1.xml", FileStorage::READ);
	fs1["biases_conv1"] >> biases_conv1;
	fs1.release();
	FileStorage fs2("Data/biases_conv2.xml", FileStorage::READ);
	fs2["biases_conv2"] >> biases_conv2;
	fs2.release();
	FileStorage fs3("Data/biases_conv3.xml", FileStorage::READ);
	fs3["biases_conv3"] >> biases_conv3;
	fs3.release();
	FileStorage fs4("Data/weights_conv1.xml", FileStorage::READ);
	fs4["weights_conv1"] >> weights_conv1;
	fs4.release();
	FileStorage fs5_1("Data/weights_conv2_1.xml", FileStorage::READ);
	fs5_1["weights_conv2_1"] >> weights_conv2[0];
	fs5_1.release();
	FileStorage fs5_2("Data/weights_conv2_2.xml", FileStorage::READ);
	fs5_2["weights_conv2_2"] >> weights_conv2[1];
	fs5_2.release();
	FileStorage fs5_3("Data/weights_conv2_3.xml", FileStorage::READ);
	fs5_3["weights_conv2_3"] >> weights_conv2[2];
	fs5_3.release();
	FileStorage fs5_4("Data/weights_conv2_4.xml", FileStorage::READ);
	fs5_4["weights_conv2_4"] >> weights_conv2[3];
	fs5_4.release();
	FileStorage fs5_5("Data/weights_conv2_5.xml", FileStorage::READ);
	fs5_5["weights_conv2_5"] >> weights_conv2[4];
	fs5_5.release();
	FileStorage fs5_6("Data/weights_conv2_6.xml", FileStorage::READ);
	fs5_6["weights_conv2_6"] >> weights_conv2[5];
	fs5_6.release();
	FileStorage fs5_7("Data/weights_conv2_7.xml", FileStorage::READ);
	fs5_7["weights_conv2_7"] >> weights_conv2[6];
	fs5_7.release();
	FileStorage fs5_8("Data/weights_conv2_8.xml", FileStorage::READ);
	fs5_8["weights_conv2_8"] >> weights_conv2[7];
	fs5_8.release();
	FileStorage fs5_9("Data/weights_conv2_9.xml", FileStorage::READ);
	fs5_9["weights_conv2_9"] >> weights_conv2[8];
	fs5_9.release();
	FileStorage fs5_10("Data/weights_conv2_10.xml", FileStorage::READ);
	fs5_10["weights_conv2_10"] >> weights_conv2[9];
	fs5_10.release();
	FileStorage fs5_11("Data/weights_conv2_11.xml", FileStorage::READ);
	fs5_11["weights_conv2_11"] >> weights_conv2[10];
	fs5_11.release();
	FileStorage fs5_12("Data/weights_conv2_12.xml", FileStorage::READ);
	fs5_12["weights_conv2_12"] >> weights_conv2[11];
	fs5_12.release();
	FileStorage fs5_13("Data/weights_conv2_13.xml", FileStorage::READ);
	fs5_13["weights_conv2_13"] >> weights_conv2[12];
	fs5_13.release();
	FileStorage fs5_14("Data/weights_conv2_14.xml", FileStorage::READ);
	fs5_14["weights_conv2_14"] >> weights_conv2[13];
	fs5_14.release();
	FileStorage fs5_15("Data/weights_conv2_15.xml", FileStorage::READ);
	fs5_15["weights_conv2_15"] >> weights_conv2[14];
	fs5_15.release();
	FileStorage fs5_16("Data/weights_conv2_16.xml", FileStorage::READ);
	fs5_16["weights_conv2_16"] >> weights_conv2[15];
	fs5_16.release();
	FileStorage fs5_17("Data/weights_conv2_17.xml", FileStorage::READ);
	fs5_17["weights_conv2_17"] >> weights_conv2[16];
	fs5_17.release();
	FileStorage fs5_18("Data/weights_conv2_18.xml", FileStorage::READ);
	fs5_18["weights_conv2_18"] >> weights_conv2[17];
	fs5_18.release();
	FileStorage fs5_19("Data/weights_conv2_19.xml", FileStorage::READ);
	fs5_19["weights_conv2_19"] >> weights_conv2[18];
	fs5_19.release();
	FileStorage fs5_20("Data/weights_conv2_20.xml", FileStorage::READ);
	fs5_20["weights_conv2_20"] >> weights_conv2[19];
	fs5_20.release();
	FileStorage fs5_21("Data/weights_conv2_21.xml", FileStorage::READ);
	fs5_21["weights_conv2_21"] >> weights_conv2[20];
	fs5_21.release();
	FileStorage fs5_22("Data/weights_conv2_22.xml", FileStorage::READ);
	fs5_22["weights_conv2_22"] >> weights_conv2[21];
	fs5_22.release();
	FileStorage fs5_23("Data/weights_conv2_23.xml", FileStorage::READ);
	fs5_23["weights_conv2_23"] >> weights_conv2[22];
	fs5_23.release();
	FileStorage fs5_24("Data/weights_conv2_24.xml", FileStorage::READ);
	fs5_24["weights_conv2_24"] >> weights_conv2[23];
	fs5_24.release();
	FileStorage fs5_25("Data/weights_conv2_25.xml", FileStorage::READ);
	fs5_25["weights_conv2_25"] >> weights_conv2[24];
	fs5_25.release();
	FileStorage fs5_26("Data/weights_conv2_26.xml", FileStorage::READ);
	fs5_26["weights_conv2_26"] >> weights_conv2[25];
	fs5_26.release();
	FileStorage fs5_27("Data/weights_conv2_27.xml", FileStorage::READ);
	fs5_27["weights_conv2_27"] >> weights_conv2[26];
	fs5_27.release();
	FileStorage fs5_28("Data/weights_conv2_28.xml", FileStorage::READ);
	fs5_28["weights_conv2_28"] >> weights_conv2[27];
	fs5_28.release();
	FileStorage fs5_29("Data/weights_conv2_29.xml", FileStorage::READ);
	fs5_29["weights_conv2_29"] >> weights_conv2[28];
	fs5_29.release();
	FileStorage fs5_30("Data/weights_conv2_30.xml", FileStorage::READ);
	fs5_30["weights_conv2_30"] >> weights_conv2[29];
	fs5_30.release();
	FileStorage fs5_31("Data/weights_conv2_31.xml", FileStorage::READ);
	fs5_31["weights_conv2_31"] >> weights_conv2[30];
	fs5_31.release();
	FileStorage fs5_32("Data/weights_conv2_32.xml", FileStorage::READ);
	fs5_32["weights_conv2_32"] >> weights_conv2[31];
	fs5_32.release();
	FileStorage fs6("Data/weights_conv3.xml", FileStorage::READ);
	fs6["weights_conv3"] >> weights_conv3;
	fs6.release();
#pragma endregion
	im = imread("in000662.jpg",1);
	Mat dst;
	//sr
	if (im.channels() > 1)
	{
		Mat im_ycrcb, im_y, im_cb, im_cr, hIm_cb, hIm_cr;
		vector<Mat> im_ycrcb_vec(3);
		vector<Mat> hIm_ycrcb(3);
		cvtColor(im, im_ycrcb, CV_RGB2YCrCb);
		split(im_ycrcb, im_ycrcb_vec);
		im_ycrcb_vec[0].copyTo(im_y);
		im_ycrcb_vec[1].copyTo(im_cr);
		im_ycrcb_vec[2].copyTo(im_cb);
		resize(im_y, lIm, Size(im_y.cols*up_scale, im_y.rows*up_scale), CV_INTER_CUBIC);
		resize(im_cr, hIm_cr, Size(im_y.cols*up_scale, im_y.rows*up_scale), CV_INTER_CUBIC);
		resize(im_cb, hIm_cb, Size(im_y.cols*up_scale, im_y.rows*up_scale), CV_INTER_CUBIC);
		hIm = SRCNN(lIm, biases_conv1, biases_conv2, biases_conv3, weights_conv1, weights_conv2, weights_conv3);
		hIm.convertTo(hIm, CV_8UC1);
		hIm.copyTo(hIm_ycrcb[0]);
		hIm_cr.copyTo(hIm_ycrcb[1]);
		hIm_cb.copyTo(hIm_ycrcb[2]);
		merge(hIm_ycrcb, dst);
		cvtColor(dst, dst, CV_YCrCb2RGB);
	}
	else
	{
		resize(im, lIm, Size(im.cols*up_scale, im.rows*up_scale), CV_INTER_CUBIC);
		hIm = SRCNN(lIm, biases_conv1, biases_conv2, biases_conv3, weights_conv1, weights_conv2, weights_conv3);
		hIm.convertTo(hIm, CV_8UC1);
		hIm.copyTo(dst);
	}
	imshow("hIm", dst);
	waitKey(0);
}

//**********SRCNN Function**********//
#pragma region SRCNN
Mat SRCNN(Mat lIm, Mat biases_conv1, Mat biases_conv2, Mat biases_conv3, \
	Mat weights_conv1, vector<Mat> weights_conv2, Mat weights_conv3)
{
	//Get CNN model parameters
	int conv1_patchsize, conv1_patchsize2, conv1_filters;
	conv1_patchsize2 = weights_conv1.rows;
	conv1_filters = weights_conv1.cols;
	conv1_patchsize = sqrt(conv1_patchsize2);

	int conv2_patchsize, conv2_channels, conv2_patchsize2, conv2_filters;
	conv2_channels = weights_conv2[0].rows;
	conv2_patchsize2 = weights_conv2[0].cols;
	conv2_filters = weights_conv2.size();
	conv2_patchsize = sqrt(conv2_patchsize2);

	int conv3_patchsize, conv3_channels, conv3_patchsize2;
	conv3_channels = weights_conv3.rows;
	conv3_patchsize2 = weights_conv3.cols;
	conv3_patchsize = sqrt(conv3_patchsize2);

	int hei, wid;
	hei = lIm.rows;
	wid = lIm.cols;

	Mat col;
	Mat double_lIm;
	lIm.convertTo(double_lIm, CV_32FC1);
	double_lIm = double_lIm / 255;
	//conv1
	//BORDER_TYPE=BORDER_REPLICATE
	vector<Mat> weights_conv1_reshape(conv1_filters), conv1_data(conv1_filters);
	for (int c = 0; c < conv1_filters; c++)
	{
		col = (weights_conv1.colRange(c, c + 1).clone());
		col = col.reshape(1, conv1_patchsize);
		weights_conv1_reshape[c] = col.t();

		//Point anchor1(weights_conv1_reshape[c].cols - weights_conv1_reshape[c].cols / 2 - 1, weights_conv1_reshape[c].rows - weights_conv1_reshape[c].rows / 2 - 1);
		Point anchor1(-1, -1);
		filter2D(double_lIm, conv1_data[c], double_lIm.depth(), weights_conv1_reshape[c], anchor1, NULL, BORDER_REPLICATE);
		float bc1;
		bc1 = biases_conv1.at<float>(c, 0);
		add(conv1_data[c], bc1, conv1_data[c]);
		max(conv1_data[c], 0, conv1_data[c]);
	}

	//conv2
	Mat row;
	vector<Mat> conv2_data(conv2_filters);
	Mat conv2_subfilter;
	for (int i = 0; i < conv2_filters; i++)
	{
		conv2_data[i] = Mat::zeros(hei, wid, CV_32FC1);
		for (int j = 0; j < conv2_channels; j++)
		{
			row = (weights_conv2[i].rowRange(j, j + 1).clone());
			row = row.reshape(1, conv2_patchsize);
			conv2_subfilter = row.t();
			Mat conv2_add;

			//Point anchor2(conv2_subfilter.cols - conv2_subfilter.cols / 2 - 1, conv2_subfilter.rows - conv2_subfilter.rows / 2 - 1);
			Point anchor2(-1, -1);
			filter2D(conv1_data[j], conv2_add, conv1_data[j].depth(), conv2_subfilter, anchor2, NULL, BORDER_REPLICATE);
			add(conv2_data[i], conv2_add, conv2_data[i]);
		}
		float bc2;
		bc2 = biases_conv2.at<float>(i, 0);
		add(conv2_data[i], bc2, conv2_data[i]);
		max(conv2_data[i], 0, conv2_data[i]);
	}

	//conv3
	Mat conv3_data = Mat::zeros(hei, wid, CV_32FC1);
	Mat conv3_subfilter;
	for (int i = 0; i < conv3_channels; i++)
	{
		row = (weights_conv3.rowRange(i, i + 1).clone());
		row = row.reshape(1, conv3_patchsize);
		conv3_subfilter = row.t();
		Mat conv3_add;

		//Point anchor3(conv3_subfilter.cols - conv3_subfilter.cols / 2 - 1, conv3_subfilter.rows - conv3_subfilter.rows / 2 - 1);
		Point anchor3(-1, -1);
		filter2D(conv2_data[i], conv3_add, conv2_data[i].depth(), conv3_subfilter, anchor3, NULL, BORDER_REPLICATE);
		add(conv3_data, conv3_add, conv3_data);
	}

	//SRCNN reconstruction
	Mat hIm = Mat::zeros(hei, wid, CV_32FC1);
	float bc3;
	bc3 = biases_conv3.at<float>(0, 0);
	add(conv3_data, bc3, hIm);
	hIm = hIm * 255;

	return hIm;
}
#pragma endregion 