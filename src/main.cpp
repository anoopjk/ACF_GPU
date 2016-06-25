#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iterator>
#include <sys/time.h>
#include <sys/stat.h>
#include <math.h>

#include "img_process.hpp"
#include "acf_detect.hpp"

using namespace std;
using namespace cv;



int main(int argc, char* argv[])
{
	//ofstream outfile("bbs.txt");

	VideoCapture capture(argv[1]);
	
    
	if (!capture.isOpened())
		cout << "fail to open!" << endl;

	
	
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);

	cout << "Total Frame Number:" << totalFrameNumber << endl;
	
	VideoWriter out_capture("output/output.avi", CV_FOURCC('M','J','P','G'), 10, Size(frame_width,frame_height));


	long frameToStart =  1;                  //3 * totalFrameNumber / 4.0;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	cout << "Start with frame" << frameToStart << endl;


	long frameToStop = totalFrameNumber;

	if (frameToStop < frameToStart)
	{
		cout << "Frame Number is wrong!" << endl;
		return -1;
	}
	else
	{
		cout << "End with frame" << frameToStop << endl;
	}
	
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "Frame Ratio FPS:" << rate << endl;


	bool stop = false;

	int delay = 1;          // 1000 / rate;
	long currentFrame = frameToStart;
	
	namedWindow("pedestrian detector", 1);
	

	
	unsigned img_height = frame_height;
	unsigned img_width  = frame_width;
	acf_detect acf(Size(img_width, img_height));	
	img_process im_proc;
	
	
	
	Mat img;
	
	while (!stop)
	{
		Mat img_luv;
		vector<bb_xma> bbs;   /// bb stored the detected bb
		if (!capture.read(img))
		{
			cout << "Video Reading Failure" << endl;
			return -1;
		}

		cout << endl << "Now detect frame " << currentFrame << endl;

		double t = (double)getTickCount();
		/////////////////////////////////////////////////////////////////////////////////////
		
		if (!img.data)
		  {
			  cout << "Image  is not loaded properly" << endl;  //handle failing images
			  continue;                  
		  }    
		// im_proc.rgb2luv(img, img_luv);
		im_proc.rgb2luv_gpu(img, img_luv);
		 acf(img_luv, bbs, im_proc.dev_output_luv_img);
		  
		
		//outfile << "image# " << currentFrame << "\n" ;
		
		///////////////////////////////////////////////////////////////////////////////////////
		t = (double)getTickCount() - t;
		
		printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());
		 t = t*1000. / cv::getTickFrequency();
		size_t i, j;
		for(unsigned int j = 0; j < bbs.size(); ++j )
			  {
				  ///@xma updated to include detection score (which will be used to sort the detection result)
				if(bbs[j].wt > 10)
				{
				// cout < "boundingboxes ," << bbs[j].x << "," << bbs[j].y << "," <<  bbs[j].wd << "," << bbs[j].ht << "," << bbs[j].wt << endl;
				  Rect r(bbs[j].x,bbs[j].y, bbs[j].wd ,bbs[j].ht);
				 // cout << "rectangle topleft " << r.tl() << " bottom right " << r.br() << endl;
					rectangle(img,r.tl() , r.br(),  cv::Scalar(0, 0, 255), 2);
					//outfile << "[" << bbs[j].x << ", " << bbs[j].y << ", " << bbs[j].wd << ", " << bbs[j].ht << "]" << "\n" ;
				}
			//rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
			}
		
		std::stringstream ss;
		ss << t ;
		std::string s = ss.str();
		
	    
		int fontFace = FONT_HERSHEY_SIMPLEX;
		double fontScale = 1;
		int thickness = 3;  
		cv::Point textOrg(250, 100);
		cv::putText(img, s, textOrg, fontFace, fontScale, Scalar::all(0), thickness,4);
		cv::putText(img, "Detection(ms) : ", cv::Point(0,100), fontFace, fontScale, Scalar::all(0), thickness,4);
		
		imshow("pedestrian detector", img);
        out_capture.write(img);// writing image to output video
        
		//waitKey(int delay=0)

		int c = waitKey(delay);


		if ((char)c == 27 || currentFrame > frameToStop)
		{
			stop = true;
		}

		if (c >= 0)
		{
			waitKey(0);
		}
		currentFrame++;

	}
	
	im_proc.free_gpu();
//outfile.close();
capture.release();
	waitKey(0);
	return 0;
}

