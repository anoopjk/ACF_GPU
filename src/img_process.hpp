#ifndef __IMG_PROCESS_H__
#define __IMG_PROCESS_H__

#include <vector>
#include <algorithm>   //std::max
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>    //std::back_inserter
#include <sys/time.h>
#include <map>
#include <math.h>

#include <stdio.h>
#include <opencv2/opencv.hpp>

typedef unsigned long long uint64_mx;
typedef long long int64_mx;

using namespace std;
//using namespace cv;

struct bb_xma
{
	/// all fields use float due to multi-scale computations
	float x; /// top left col #
	float y; /// top left row #
	float ht;  /// height of bb
	float wd;  /// width of bb
	float wt;  /// weight
};

struct more_than_key
{
    inline bool operator() (const bb_xma& struct1, const bb_xma& struct2)
	{
		return (struct1.wt > struct2.wt);
	}
};



class img_process
{
public:
	img_process();
	virtual ~img_process();
	void rgb2luv(cv::Mat& in_img, cv::Mat& out_img); /// this function converts a uint8 array (3 channels to CV32F array of luv, NOTE that channels are mapped in order b->R, g->G, r->B)
	void rgb2luv(cv::Mat& in_img, cv::Mat& out_img, float nrm, bool useRGB = false); /// this function converts a CV32F (3 channels to CV32F array of luv), NOTE THAT THE CHANNELS ARE 1-1 MAPPED (R->R, G->G, B->B)

	void rgb2luv_gpu(cv::Mat& in_img, cv::Mat& out_img);
	void rgb2luv_gpu(cv::Mat& in_img, cv::Mat& out_img, float nrm, bool useRGB = false);
	static void imResample_array_int2lin_gpu(float* in_img, float* out_img, int n_channels, int org_ht, int org_wd, int dst_ht, int dst_wd, float r=1.0f);
	static void imResample_array_lin2lin_gpu(float* in_img, float* out_img, int n_channels, int org_ht, int org_wd, int dst_ht, int dst_wd, float r=1.0f);
	static void ConvTri1_gpu(float* I, float* O, int ht, int wd, int dim, float p, int s = 1);
	void free_gpu(void);

	static void imResample(cv::Mat& in_img, cv::Mat& out_img, int dheight, int dwidth, float r = 1.0f ); /// bilinear interpolation methods to resize image
	static void imResample_array_int2lin(float* in_img, float* out_img, int d, int org_ht, int org_wd, int dst_ht, int dst_wd, float r=1.0f);
	static void imResample_array_lin2lin(float* in_img, float* out_img, int d, int org_ht, int org_wd, int dst_ht, int dst_wd, float r=1.0f);
	static void ConvTri1(float* I, float* O, int ht, int wd, int dim, float p = 7.6f, int s = 1);
	/// structure test functions:
	void get_pix_all_scales_int(cv::Mat& img, const vector<cv::Size>& scales, float* pix_array);    /// save all pixel values for all scales in an float array. Color channels are interleaved.
	void get_pix_all_scales_lin(cv::Mat& img, const vector<cv::Size>& scales, float* pix_array);    /// save all pixel values for all scales in an float array. Color channels are separated

	float *dev_output_luv_img; /* pointer to output image on gpu */

private:
	/// function and variables for rgb2luv conversion
	void rgb2luv_setup(float nrm);
	void rgb2luv_setup_gpu(float nrm);
	float minu, minv, un, vn, mr[3], mg[3], mb[3];
	float lTable[1064];

	unsigned char *dev_input_img; /* pointer to input image on gpu */
	//float *dev_C_temp;

	static void resampleCoef(int ha, int hb, int &n, int *&yas, int *&ybs, float *&wts, int bd[2], int pad=0);
	static void ConvTri1X(float* I, float* O, int wd, float p = 7.6f, int s = 1);
};

#endif

