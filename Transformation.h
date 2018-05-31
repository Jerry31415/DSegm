#include <opencv2\core.hpp>
#include <vector>

#include <iostream>

// интерфейс для преобразований
// все классы реализуюемые данный интерфейс должны содержать методы apply и calc
// apply - возвращет объект Point2d - результат применения преобразования для аргумента
// calc - вычисляет преобразование 
// получает на входе оптический поток (матрицу типа CV_32FC2) и сегмент (вектор точек)  
class Transformation : public std::vector<double>{
	public:
		virtual ~Transformation(){
			std::vector<double>::~vector();
		}

		virtual cv::Point2d apply(const cv::Point2d&) = 0;
		virtual void calc(const cv::Mat&, const std::vector<cv::Point2d>&) = 0;
};

// Преобразование - сдвиг
class ShiftTransformation : public Transformation{
	public: 

		ShiftTransformation(){
			resize(2, 0);
		}

		ShiftTransformation(const std::vector<double>& tr){
			resize(2, 0);
			std::copy(tr.begin(), tr.begin() + 2, begin());
		}

		cv::Point2d apply(const cv::Point2d& p){
			return cv::Point2d(p.x + at(0), p.y + at(1));
		}

		void calc(const cv::Mat& flow, const std::vector<cv::Point2d>& seg){
			if (seg.empty()) return;
			cv::Vec2f v(0, 0);
			for (const auto& p : seg){
				v += flow.at<cv::Vec2f>((int)p.y, (int)p.x);
			}
			at(0) = v[0] / seg.size();
			at(1) = v[1] / seg.size();
		}

};

// Аффинное преобразование
class AffineTransformation : public Transformation{
	public:

		AffineTransformation(){
			resize(6, 0);
		}

		AffineTransformation(const std::vector<double>& tr){
			resize(6, 0);
			std::copy(tr.begin(), tr.begin()+6, begin());
		}

		cv::Point2d apply(const cv::Point2d& p){
			return cv::Point2d(at(0) * p.x + at(1) * p.y + at(2), at(3) * p.x + at(4) * p.y + at(5));
		}

		void calc(const cv::Mat& flow, const std::vector<cv::Point2d>& seg){
			if (seg.empty()) return;
			cv::Vec2f v(0, 0);
			double sx(0), sy(0), s2x(0), s2y(0), sxy(0), sxVx(0), syVy(0), sVx(0), sVy(0), syVx(0), sxVy(0);
			for (auto& e : seg){
				v = flow.at<cv::Vec2f>((int)e.y, (int)e.x);
				sx += e.x;
				sy += e.y;
				s2x += e.x*e.x;
				s2y += e.y*e.y;
				sxy += e.x*e.y;
				sxVx += e.x*v[0];
				syVy += e.y*v[1];
				sVx += v[0];
				sVy += v[1];
				syVx += e.y*v[0];
				sxVy += e.x*v[1];
			}
			cv::Mat A = cv::Mat::zeros(3, 3, CV_64FC1);
			cv::Mat b0 = cv::Mat::zeros(3, 1, CV_64FC1);
			cv::Mat b1 = cv::Mat::zeros(3, 1, CV_64FC1);

			b0.at<double>(0, 0) = s2x + sxVx;
			b0.at<double>(1, 0) = sxy + syVx;
			b0.at<double>(2, 0) = sx + sVx;

			b1.at<double>(0, 0) = sxy + sxVy;
			b1.at<double>(1, 0) = s2y + syVy;
			b1.at<double>(2, 0) = sy + sVy;

			A.at<double>(0, 0) = s2x;
			A.at<double>(0, 1) = sxy;
			A.at<double>(0, 2) = sx;
			A.at<double>(1, 0) = sxy;
			A.at<double>(1, 1) = s2y;
			A.at<double>(1, 2) = sy;
			A.at<double>(2, 0) = sx;
			A.at<double>(2, 1) = sy;
			A.at<double>(2, 2) = seg.size();

			cv::Mat inv;
			invert(A, inv);
			cv::Mat ax = inv*b0;
			cv::Mat ay = inv*b1;

			for (int i = 0; i < 3; ++i){
				at(i) = ax.at<double>(i);
				at(i + 3) = ay.at<double>(i);
			}
		}


};

// Проективное преобразование
class ProjectiveTransformation : public Transformation{
	public:

	ProjectiveTransformation(){
		resize(9, 0);
	}

	ProjectiveTransformation(const std::vector<double>& tr){
		resize(9, 0);
		std::copy(tr.begin(), tr.begin() + 9, begin());
	}

	cv::Point2d apply(const cv::Point2d& p){
		double dev = at(6) * p.x + at(7) * p.y + at(8);
		return cv::Point2d((at(0) * p.x + at(1) * p.y + at(2)) / dev, (at(3) * p.x + at(4) * p.y + at(5)) / dev);
	}

	void calc(const cv::Mat& flow, const std::vector<cv::Point2d>& seg){
		if (seg.empty()) {
			at(8) = 1;
			return;
		}
		cv::Vec2f v(0, 0);
		cv::Mat A = cv::Mat::zeros(8, 8, CV_64FC1);
		cv::Mat b = cv::Mat::zeros(8, 1, CV_64FC1);
		double sx, sy;
		for (auto& e : seg){
			v = flow.at<cv::Vec2f>((int)e.y, (int)e.x);
			sx = e.x + v[0];
			sy = e.y + v[1];
			// 1-я строка
			A.at<double>(0, 0) += e.x*e.x;
			A.at<double>(0, 1) += e.x*e.y;
			A.at<double>(0, 2) += e.x;
			A.at<double>(0, 6) += -sx*e.x*e.x;
			A.at<double>(0, 7) += -sx*e.x*e.y;
			b.at<double>(0, 0) += sx*e.x;
			// 2-я строка
			A.at<double>(1, 1) += e.y*e.y;
			A.at<double>(1, 2) += e.y;
			A.at<double>(1, 7) += -sx*e.y*e.y;
			b.at<double>(1, 0) += sx*e.y;
			// 3-я строка
			b.at<double>(2, 0) += sx;
			// 4-я строка
			A.at<double>(3, 6) += -sy*e.x*e.x;
			A.at<double>(3, 7) += -sy*e.x*e.y;
			b.at<double>(3, 0) += sy*e.x;
			// 5-я строка
			A.at<double>(4, 7) += -sy*e.y*e.y;
			b.at<double>(4, 0) += sy*e.y;
			// 6-я строка
			b.at<double>(5, 0) += sy;
			// 7-я строка
			A.at<double>(6, 0) += -sx*e.x*e.x;
			A.at<double>(6, 1) += -sx*e.x*e.y;
			A.at<double>(6, 2) += -sx*e.x;
			A.at<double>(6, 3) += -sy*e.x*e.x;
			A.at<double>(6, 4) += -sy*e.x*e.y;
			A.at<double>(6, 5) += -sy*e.x;
			A.at<double>(6, 6) += (sx*sx + sy*sy)*e.x*e.x;
			A.at<double>(6, 7) += (sx*sx + sy*sy)*e.x*e.y;
			b.at<double>(6, 0) += -(sx*sx + sy*sy)*e.x;
			// 8-я строка
			A.at<double>(7, 1) += -sx*e.y*e.y;
			A.at<double>(7, 4) += -sy*e.y*e.y;
			A.at<double>(7, 5) += -sy*e.y*e.y;
			A.at<double>(7, 7) += (sx*sx + sy*sy)*e.y*e.y;
			b.at<double>(7, 0) += -(sx*sx + sy*sy)*e.y;
		}
		A.at<double>(1, 0) = A.at<double>(0, 1);
		A.at<double>(1, 6) = A.at<double>(0, 7);
		// 3-я строка - симметрия
		A.at<double>(2, 0) = A.at<double>(0, 2);
		A.at<double>(2, 1) = A.at<double>(1, 2);
		A.at<double>(2, 2) = seg.size();
		A.at<double>(2, 6) = -b.at<double>(0, 0);
		A.at<double>(2, 7) = -b.at<double>(1, 0);
		// 4-6 я строки - симметрия
		for (int i = 0; i < 3; ++i){
			for (int j = 0; j < 3; ++j){
				A.at<double>(i + 3, j + 3) = A.at<double>(i, j);
			}
		}
		// 5-я строка - симметрия
		A.at<double>(4, 6) = A.at<double>(3, 7);
		// 6-я строка - симметрия
		A.at<double>(5, 6) = -b.at<double>(3, 0);
		A.at<double>(5, 7) = -b.at<double>(4, 0);
		// 8-я строка - симметрия
		A.at<double>(7, 0) = A.at<double>(6, 1);
		A.at<double>(7, 2) = -b.at<double>(1, 0);
		A.at<double>(7, 3) = A.at<double>(6, 4);
		A.at<double>(7, 5) = -b.at<double>(4, 0);
		A.at<double>(7, 6) = A.at<double>(6, 7);

		// Решение системы
		cv::Mat inv;
		invert(A, inv);
		cv::Mat X = inv*b;
		// Инициализация весовых коэф.
		for (int i = 0; i < 8; ++i) at(i) = X.at<double>(i);
		at(8) = 1;
	}
};

