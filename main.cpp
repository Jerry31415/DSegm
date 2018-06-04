#pragma warning(disable : 4996)
#pragma warning(disable: 4800) // 'int' : forcing value to bool 'true' or 'false'...

#include <opencv2\highgui.hpp> 
#include <opencv\cv.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include "opencv2\imgcodecs\imgcodecs.hpp"

#include "Segmentation.h"

using namespace cv;

// изменять размер кадров на 320х240
//#define USE_320_RESIZE

// сглаживать Гауссом с ядром 3х3
#define USE_GAUSSIAN_BLUR_3

int main(int argc, char *argv[]){
	Mat fr0, fr1, g0, g1;
	double cost_step = -0.5;
	if (argc < 3){
		std::cout << "Error: incorrect arguments\n";
		fr0 = imread("0300.bmp");
		fr1 = imread("0300_.bmp");
	}
	else {
		fr0 = imread(argv[1]);
		fr1 = imread(argv[2]);
	}
	if (argc == 4){
		try{
			cost_step = std::stod(argv[3]);
		}	
		catch (...){
			std::cout << "Error: incorrect cost_step argument\n";
			std::cin.get();
		}
	}
	g0 = fr0;
	g1 = fr1;
#ifdef USE_320_RESIZE
	resize(fr0, fr0, Size(320, 240));
	resize(fr1, fr1, Size(320, 240));
#endif

#ifdef USE_GAUSSIAN_BLUR_3
	GaussianBlur(g0, g0, Size(3, 3), 0);
	GaussianBlur(g1, g1, Size(3, 3), 0);
#endif
	SegInfo seg_info;
	Segmentation<AffineTransformation>(g0, g1, seg_info, 100, cost_step);
	std::cout << "Quality=" << CalcQuality<AffineTransformation>(fr0, fr1, seg_info.seg) << "\n";

	if (argc<3) imwrite("seg.bmp", seg_info.img);
	else imwrite("seg_" + std::string(argv[1]), seg_info.img);
	
	std::cin.get();
}