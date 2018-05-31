#include "opencv2\imgcodecs\imgcodecs.hpp"
#include "Segmentation.h"

using namespace cv;

// изменять размер кадров на 320х240
//#define USE_320_RESIZE

// сглаживать Гауссом с ядром 3х3
//#define USE_GAUSSIAN_BLUR_3

int main(int argc, char *argv[]){
	
	if (argc < 3){
		std::cout << "Error: incorrect arguments\n";
	}
	Mat fr0 = imread(argv[1]);
	Mat fr1 = imread(argv[2]);

#ifdef USE_320_RESIZE
	resize(fr0, fr0, Size(320, 240));
	resize(fr1, fr1, Size(320, 240));
#endif

#ifdef USE_GAUSSIAN_BLUR_3
	GaussianBlur(fr0, fr0, Size(3, 3), 0);
	GaussianBlur(fr1, fr1, Size(3, 3), 0);
#endif

	SegInfo seg_info;
	Segmentation<ShiftTransformation>(fr0, fr1, seg_info, 100, -5);
	std::cout << "Quality=" << CalcQuality<ShiftTransformation>(fr0, fr1, seg_info.seg) << "\n";
	imwrite("seg_" + std::string(argv[1]), seg_info.img);
}