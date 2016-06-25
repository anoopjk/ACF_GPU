#ifndef __ACF_DETECT_H__
#define __ACF_DETECT_H__

#include <vector>
#include <algorithm>   //std::max, copy, sort
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>    //std::back_inserter
#include <sys/time.h>
#include <map>
#include <math.h>
#include <string.h>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include "img_process.hpp"
#include "cartToPolar.hpp"

#ifndef PI
#define PI 3.14159265f
#endif

using namespace std;
//using namespace cv;


class acf_detect
{
public:
	acf_detect(cv::Size org_sz);
	void operator()(cv::Mat& img, vector<bb_xma>& bbs, float *luv_img_gpu);
	void chnsPyramid(float* pix_array, float* chnsPyramid); /// compute the chnsPyramid for all scales, note that the channels are linearly stored (not interleaved)
	void chnsCompute(float* pix_array, float* chnsPyramid, cv::Size scale);
	void GradMag (float* pix_array, float* M, float* O, int ht, int wd, int dim, bool full = false);
	void GradHist(float*  chnsPyramid, float* M, float* O, int ht, int wd, bool full = false);
	void acfDetect(float*chns, const int height, const int width, const int nChns, vector<bb_xma>& bbs);
	void bbNms(vector<bb_xma>& bbs_det, vector<bb_xma>& bbs_res, float overlap = 0.650f, bool greedy = true, bool ovrDnm = false);
	inline unsigned get_total_pix_num()const {return total_array_size;};
	inline int get_nPerOct() const  {return nPerOct;};
	inline int get_nOctUp()  const  {return nOctUp;};
	inline int get_shrink()  const  {return shrink;};
	inline int get_nOrients()const  {return nOrients;};
	inline int get_nChannels()const {return nChannels;};
	static unsigned get_scales(const int nPerOct, const int nOctUp, const int shrink, const cv::Size org_sz, const float minDs_ht, const float minDs_wd,
							   vector<cv::Size>& scales_sz, vector<float>& scale_h, vector<float>& scale_w, vector< float>& scale_o);
	virtual ~acf_detect();

private:
	const int nPerOct;
	const int nOctUp;
	const int shrink;
	const int nOrients;
	const int nChannels;
	const cv::Size org_sz;
	const float convTri_p;
	/// the two parameters are computed based on matlab code, see acfDetect function in matlab, Piotr's toolbox
	cv::Size_<float> modelDs;    /// width 20.5, height 50; 
	cv::Size_<float> modelDsPad; /// width 32,   height 64;
	float shift_ht;
	float shift_wd;
	vector<cv::Size> scales;
	vector<float > scale_h; /// height scale factor
	vector<float > scale_w; /// width scale factor
	vector<float > scale_o; /// overall scale factor
	unsigned total_array_size;
	void  acosTable();
	float a[20020];
	float* acost;
	void grad_row( float *I, float *Gx, float *Gy, int ht, int wd, int y, int dim );
	void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1,
					  int nb, int n, float norm, bool full);
	void get_acf_cl();
	
	const float* thrs;
	const float* hs;
	const unsigned* fids;

	/// getChild is a binary tree pruning:
	/// k0 is the node # (0,1,2,3,4,5,6), k0's child is computed by 2 * k0 + k
	/// k could be 1 or 2 depending on the ftr compared to chns1 value
	/// k will be added on top of offset because all tress are concatenated to form a single array
	/// xma illustration of a 7-node binary tree, note that for each tree

	/*                   0
	//                 /   \
	//                1     2
	//               / \   / \
	//              3   4 5   6
	*/
	
	inline
	void getChild( const float *chns1, const unsigned *cids1, const unsigned *fids1,
				   const float *thrs1, const unsigned offset, unsigned &k0, unsigned &k /*, bool print = false*/ )
	{
		float ftr = chns1[cids1[fids1[k]]];
		/*
		if(print)
			cout << "k = " << k << ", fids[" << k << "] = " << fids1[k] << ", cids[" << fids1[k] << "] = " << cids1[fids1[k]]
				 << ", chns[" << cids1[fids1[k]]  << "] = " << chns1[cids1[fids1[k]]] << endl;
		*/
		k = (ftr<thrs1[k]) ? 1 : 2;
		k0=k+=k0*2; k+=offset;
	};
	
};

#endif
	
