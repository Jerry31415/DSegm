#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4996)
#pragma warning(disable: 4800) // 'int' : forcing value to bool 'true' or 'false'...

#include <stdio.h>

#include <iostream>
#include <memory>
#include <exception>

#include <chrono>
#include <ctime>

#include "opencv2\imgproc.hpp"
#include "opencv2\video\tracking.hpp"

#include "Transformation.h"
#include "utility.h"

using namespace cv;

// вывод текущей сегментации на экран
//#define SHOW_SMAT

#include "opencv2\highgui\highgui.hpp"
#include <thread>

bool active_wnd;

void winThread(){
	active_wnd = true;
	cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
	while (cv::waitKey(1) != 27 && active_wnd){}
}

struct SegInfo{
	Mat seg;
	Mat img;
	double Q;
};

// Динамический сегмент - сегмент + преобразование
template<typename TRANSFORM_TYPE>
class Segment{
	public:
		std::vector<Point2d> seg;
		std::shared_ptr<Transformation> tr;

		Segment(){
			tr = std::make_shared<TRANSFORM_TYPE>();
		}

		~Segment(){
		}

		void UpdateTransform(const Mat& flow, const Mat& fr0, const Mat& fr1){
			std::vector<double> tmpV;
			if (!seg.empty()){
				double Q0 = Q(fr0, fr1);
				tr->copyTo(tmpV);
				tr->calc(flow, seg);
				double Q1 = Q(fr0, fr1);
				//std::cout << "Q0=" << Q0 << "; Q1=" << Q1 << "\n";
				if (Q1 > Q0) {
					tr->copyFrom(tmpV);
					//std::cout << "Q0=" << Q0 << "; Q1=" << Q1 << tr->toString() <<  "\n";
				}
			}
			else {
				throw std::runtime_error("Error: empty segment");
			}
		}

		// Добавление прямоугольного сегмента
		void AddPointsRect(const Point2i& p0, const Point2i& p1){
			for (int i = p0.y; i <= p1.y; ++i){
				for (int j = p0.x; j <= p1.x; ++j){
					seg.push_back(Point2d(j, i));
				}
			}
		}

		double Q(const Mat& fr0, const Mat& fr1){
			double t, sum(0);
			int cnt = 0;
			Point2d new_p;
			Vec3d fp, sp;
			for (auto& p : seg){
				// точка на втором кадре
				new_p = tr->apply(p);
				// Если точка внутри кадра
				if (new_p.x < fr0.cols && new_p.x >= 0){
					if (new_p.y <= fr0.rows && new_p.y >= 0){
						t = 0;
						// точка на втором кадре
						fp = interpColor(fr1, new_p);
						// точка с первого кадра
						sp = interpColor(fr0, p);
						// MSE
						for (int z = 0; z < 3; ++z) t += (fp[z] - sp[z])*(fp[z] - sp[z]);
						sum += t;
						++cnt;
					}
				}
			}
			if (cnt == 0) return 9999;
			sum = sqrt(sum/(cnt * 3.));
			return sum;
		}

		double Q(const Mat& fr0, const Mat& fr1, Transformation* trans){
			double t, sum(0);
			int cnt = 0;
			Point2d new_p;
			Vec3d fp, sp;
			for (auto& p : seg){
				new_p = trans->apply(p);
				// Если точка внутри кадра
				if (new_p.x < fr0.cols && new_p.x >= 0){
					if (new_p.y <= fr0.rows && new_p.y >= 0){
						t = 0;
						// точка на втором кадре
						fp = interpColor(fr1, new_p);
						// точка с первого кадра
						sp = interpColor(fr0, p);
						// MSE
						for (int z = 0; z < 3; ++z) t += (fp[z] - sp[z])*(fp[z] - sp[z]);
						sum += t;
						++cnt;
					}
				}
			}
			if (cnt == 0) return 9999;
			sum = sqrt(sum / (cnt * 3.));
			return sum;
		}

};

#ifdef SHOW_SMAT
template<typename TRANSFORM_TYPE>
void ShowSegMatrix(Mat& fr0, std::vector<Segment<TRANSFORM_TYPE>>& SegList){
	Mat tmp;
	TransformListToImage_MidColor(fr0, SegList, tmp);
	imshow("test", tmp);
}
#endif

template<typename TRANSFORM_TYPE>
double PointQuality(const Mat& fr0, const Mat& fr1, const Point2d& p, 
	std::vector<Segment<TRANSFORM_TYPE>>& SegList, int index){
	if (p.x >= fr0.cols || p.x < 0 || p.y >= fr0.rows || p.y < 0) return -1;
	double t = 0;
	Point2d new_p = SegList[index].tr->apply(p);
	if (new_p.x >= fr0.cols || new_p.x < 0 || new_p.y >= fr0.rows || new_p.y < 0) return -1;
	Vec3d fp = interpColor(fr1, new_p);
	Vec3d sp = interpColor(fr0, p);
	for (int z = 0; z < 3; ++z) t += (fp[z] - sp[z])*(fp[z] - sp[z]);
	return t / 3.;
}

// качество точки по окресности радиуса SR
template<typename TRANSFORM_TYPE>
double QPoint(const Mat& fr0, const Mat& fr1, std::vector<Segment<TRANSFORM_TYPE>>& tr, int index, int i, int j, int SR = 1){
	double pq, t, cnt_p;
	pq = cnt_p = 0;
	for (int ii = -SR; ii <= SR; ++ii){
		if (ii + i < 0 || ii + i >= fr0.rows) continue;
		for (int jj = -SR; jj <= SR; ++jj){
			if (jj + j < 0 || jj + j >= fr0.cols) continue;
			t = PointQuality(fr0, fr1, Point2d(j + jj, i + ii), tr, index);
			if (t < 0) continue;
			pq += t;
			cnt_p++;
		}
	}
	return (cnt_p == 0) ? -1 : sqrt(pq / cnt_p);
}

template<typename TRANSFORM_TYPE>
double CalcQuality(const Mat& fr0, const Mat& fr1, Mat& seg, std::vector<Segment<TRANSFORM_TYPE>>& SegList){
	double Q(0), PQ(0), cnt(0);
	for (int i = 0; i < fr0.rows; ++i){
		for (int j = 0; j < fr0.cols; ++j){
			PQ = PointQuality(fr0, fr1, Point2d(j, i), SegList, seg.at<int>(i, j));
			if (PQ >= 0){
				Q += PQ;
				cnt++;
			}
		}
	}
	return sqrt(Q / (cnt));
}

template<typename TRANSFORM_TYPE>
void TransformListToImage_MidColor(const Mat& fr0, const std::vector<Segment<TRANSFORM_TYPE>>& SegmList, Mat& out){
	std::vector<Vec3b> colors;
	Vec3b tmp_color;

	auto mid_color = [](const Mat& fr0, const std::vector<Segment<TRANSFORM_TYPE>>& SegmList, const int& index)->Vec3b{
		Vec3d dC(0, 0, 0);
		for (auto& p : SegmList[index].seg){
			dC += fr0.at<Vec3b>(p.y, p.x);
		}
		dC *= 1. / SegmList[index].seg.size();
		return Vec3b((int)dC[0], (int)dC[1], (int)dC[2]);
	};

	for (int i = 0; i < SegmList.size(); ++i){
		colors.push_back(mid_color(fr0, SegmList, i));
	}

	out = Mat(fr0.size(), CV_8UC3);

	for (size_t i = 0; i < SegmList.size(); ++i){
		for (auto& p : SegmList[i].seg){
			out.at<Vec3b>((int)p.y, (int)p.x) = colors[i];
		}
	}
}

// Построение матрицы сегментации seg из списка SegmList
template<typename TRANSFORM_TYPE>
void BuildMatFromSegmentList(Mat& seg, Size mat_size, std::vector<Segment<TRANSFORM_TYPE>>& SegmList){
	seg = Mat::zeros(mat_size, CV_32SC1);
	for (int i = 0; i < SegmList.size(); ++i){
		for (auto& p : SegmList[i].seg){
			seg.at<int>((int)p.y, (int)p.x) = i;
		}
	}
}

// Построение списка сегментов из матрицы сегментации
template<typename TRANSFORM_TYPE>
void BuildSegmentListFromMat(Mat& seg, std::vector<Segment<TRANSFORM_TYPE>>& SegList, Mat& flow,
	const Mat& fr0, const Mat& fr1){
	// Calc new size of transform list
	int max_index(-1), min_index(99999999), tmp_val;
	for (int i = 0; i < seg.rows; ++i){
		for (int j = 0; j < seg.cols; ++j){
			tmp_val = seg.at<int>(i, j);
			if (max_index < tmp_val) max_index = tmp_val;
			if (min_index > tmp_val) min_index = tmp_val;
		}
	}
	if (min_index != 0){
		std::cerr << "Warning : Minimal index in segment matrix != 0. Alghoritm is not applicable\n";
		std::cerr << "Max index=" << max_index << " ; Min index=" << min_index << "\n";
		std::cin.get();
	}

	// Init new transform list
	SegList.clear();
	for (int i = 0; i <= max_index; ++i){
		SegList.push_back(Segment<TRANSFORM_TYPE>());
	}

	for (int i = 0; i < seg.rows; ++i){
		for (int j = 0; j < seg.cols; ++j){
			tmp_val = seg.at<int>(i, j);
			SegList[tmp_val].seg.push_back(Point2d(j, i));
		}
	}
	for (int i = 0; i < SegList.size();++i){
		SegList[i].UpdateTransform(flow, fr0, fr1);
	}
}


template<typename TRANSFORM_TYPE>
double CalcQuality(const Mat& fr0, const Mat& fr1, Mat& seg){
	std::vector<Segment<TRANSFORM_TYPE>> SegList;
	Mat flow, g0, g1;
	cvtColor(fr0, g0, COLOR_BGR2GRAY);
	cvtColor(fr1, g1, COLOR_BGR2GRAY);
	calcOpticalFlowFarneback(g0, g1, flow, 0.4, 1, 16, 25, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
	BuildSegmentListFromMat<TRANSFORM_TYPE>(seg, SegList, flow, fr0, fr1);
	double Q(0), PQ(0), cnt(0);
	for (int i = 0; i < fr0.rows; ++i){
		for (int j = 0; j < fr0.cols; ++j){
			PQ = PointQuality(fr0, fr1, Point2d(j, i), SegList, seg.at<int>(i, j));
			if (PQ >= 0){
				Q += PQ;
				cnt++;
			}
		}
	}
	return sqrt(Q / (cnt));
}

// Перенос всех точек сегмента с индексом index_from в сегмент с индексом index_to
template<typename TRANSFORM_TYPE>
void TransferPoints(Mat& seg, std::vector<Segment<TRANSFORM_TYPE>>& tr, int index_from, int index_to){
	cv::Point2d p;
	for (int z = 0; z < tr[index_from].seg.size(); ++z){
		// Перекрашиваем сегмент в матрице
		p = tr[index_from].seg[z];
		seg.at<int>((int)p.y, (int)p.x) = index_to;
		// Копируем точки в соседний сегмент
		tr[index_to].seg.push_back(p);
	}
	// Очищаем список точек присоединенного сегмента
	tr[index_from].seg.clear();
}

// Перенос всех точек сегмента с индексом index_from в сегмент с индексом index_to
template<typename TRANSFORM_TYPE>
void TransferPoints(std::vector<Segment<TRANSFORM_TYPE>>& tr, int index_from, int index_to){
	// Копируем точки в соседний сегмент
	tr[index_to].seg.insert(tr[index_to].seg.begin(), tr[index_from].seg.begin(), tr[index_from].seg.end());
	// Очищаем список точек присоединенного сегмента
	tr[index_from].seg.clear();
}

// Вычисление начальной сегментации. Разбиение на прямоугольники размера i_step x j_step
template<typename TRANSFORM_TYPE>
void Init(Mat& fr0, Mat& fr1, const Mat& flow, Mat& seg, 
	std::vector<Segment<TRANSFORM_TYPE>>& SLst, const int& i_step, const int& j_step){
	seg = Mat::zeros(fr0.size(), CV_32SC1);
	for (int i = 0; i < fr0.rows; i += i_step){
		for (int j = 0; j < fr0.cols; j += j_step){
			Segment<TRANSFORM_TYPE> tmp_segm;
			//tmp_segm.AddPointsRect(Point2i(j, i), Point2i(j + 1, i + 1));
			tmp_segm.AddPointsRect(Point2i(j, i), Point2i(j + j_step - 1, i + i_step - 1));
			SLst.push_back(tmp_segm);
		}
	}
	for (int i = 0; i < SLst.size(); ++i){
		SLst[i].UpdateTransform(flow, fr0, fr1);
	}
}

//Склейка сегментов (список)
// nIndexes - список смежности
template<typename TRANSFORM_TYPE>
int Merging(const Mat& fr0, const Mat& fr1, Mat& flow, std::vector<Segment<TRANSFORM_TYPE>>& Segm, 
	std::vector<std::set<int>>& nIndexes, int min_size, double cost = 0){
	// Пробегаем по списку всех сегментов
	// Начинаем с конца
	double Qs, Qm, best_Q;
	int min_ind;
	int cnt_merged = 0;
	loop:
	for (int i(Segm.size() - 1), cnt(1); i >= 0; --i, ++cnt){
		if (Segm[i].seg.empty()) continue;
		min_ind = -1;
		best_Q = 99999;
		// Если оценка качества присоединения < заданного порога,
		// то присоединяем к NI[i]-сегменту i-ый сегмент
		// Здесь NN - индексы соседних сегментов
		for (auto& NN : nIndexes[i]){
			if (Segm[NN].seg.empty()) {
				std::cout << "Error : segm size == 0, line - " << __LINE__ << "\n";
				std::cin.get();
			}
			// если размер присоединяемого сегмента больше исходного
			if (Segm[NN].seg.size() < Segm[i].seg.size()) continue;
			Qs = Segm[i].Q(fr0, fr1);
			Qm = Segm[i].Q(fr0, fr1, Segm[NN].tr.get());
			if ((Qs - Qm) > cost){
				if (Qm < best_Q){
					best_Q = Qm;
					min_ind = NN;
				}
			}
		}
		if (min_ind != -1){
			// Копируем точки в соседний сегмент
			Segm[min_ind].seg.insert(Segm[min_ind].seg.begin(), Segm[i].seg.begin(), Segm[i].seg.end());
			// Пересчитываем преобразование
			//!!!!!!!!!!!
			//if (Segm[i].seg.size()/(Segm[min_ind].seg.size()-Segm[i].seg.size())>0.05)
			Segm[min_ind].UpdateTransform(flow, fr0, fr1);
			// Удаляем присоединенный сегмент из списка
			Segm.erase(Segm.begin() + i);
			// Обновляем список смежности
			MergeNList(nIndexes, i, min_ind, true);
			++cnt_merged;
			goto loop;
		}
	}
	return cnt_merged;
}

template<typename TRANSFORM_TYPE>
void UpdateSegmentList(Mat& seg, std::vector<Segment<TRANSFORM_TYPE>>& tr){
	for (int i = 0; i < tr.size(); ++i) tr[i].seg.clear();
	for (int i = 0; i < seg.rows; ++i){
		for (int j = 0; j < seg.cols; ++j){
			tr[seg.at<int>(i, j)].seg.push_back(Point2d(j, i));
		}
	}
}

//-----------------------------------------------------------------------------
// сравнение матрицы со списком сегментов
template<typename TRANSFORM_TYPE>
void testList(Mat& seg, std::vector<Segment<TRANSFORM_TYPE>>& trs) {
	int cnt = 0;
	for (int i = 0; i < trs.size(); ++i) {
		int sz = trs[i].seg.size();
		cnt += sz;
		for (int j = 0; j < sz; j++) {
			Point2d p = trs[i].seg[j];
			int isg = seg.at<int>(p.y, p.x);
			if (isg != i) {
				std::cout << "Error in testList: i=" << i << ", isg="<<isg << ", p="<<p << "\n";
			}
		}
	}
}

//-----------------------------------------------------------------------------
// записать результаты в текстовый файл
template<typename TRANSFORM_TYPE>
void SaveRes(const char *fname, const char *msg, const Mat& fr0, const Mat& fr1,
	Mat& seg, std::vector<Segment<TRANSFORM_TYPE>>& trs) {
	FILE *f = fopen(fname, "a");
	double Q = CalcQuality(fr0, fr1, seg, trs);
	fprintf(f, "----------------------------\ninfo, nSegm=%d, Q=%8.2f, %s\n", (int)trs.size(), Q, msg);
	for (int i = 0; i < trs.size(); ++i) {
		int sz = trs[i].seg.size();
		Q = trs[i].Q(fr0, fr1);
		fprintf(f, "%4d, %6d, %7.2f, %s\n", i, sz, Q, trs[i].tr->toString().c_str());
	}

	fclose(f);
}


template<typename TRANSFORM_TYPE>
int BordersUpdate(const Mat& fr0, const Mat& fr1, Mat& seg, Mat& flow, 
	std::vector<Segment<TRANSFORM_TYPE>>& SegList, int min_size, int SR = 1){
	bool isChanged = false;
	double Q, QM;
	int ind, ind_src;
	std::set<int> e;
	Point2d p;
	int cnt = 0;
//	std::vector<Point2d> ve;
	int eN;
	// Для каждой точки изображения
	for (int i = 0; i < seg.rows; ++i){
		for (int j = 0; j < seg.cols; ++j){
			p = Point2d(j, i);

			// Точки не являющиеся граничными пропускаем
			if (!isEdge(seg, p, e, eN)) continue;
			//isEdgeV(seg, p, ve, eN);
			// Номер сегмента
			ind_src = seg.at<int>(i, j);
			ind = ind_src;

			// Вычсляем качество точки для исходного афф. преобразования
			if (SegList[ind_src].seg.empty()) {
				std::cout << "Logic Error\n";
				continue;
			}
			// Если размер сегмента < min_size, то
			// Искуственно зададим ему плохое качество, чтобы его точки разделились между соседями

			if (SegList[ind_src].seg.size()<min_size || eN == 8) {
				Q = 9999999;
			}
			else{
				Q = QPoint(fr0, fr1, SegList, ind_src, i, j, SR);
			}
			// Если точка вылетела за границу изображения
			if (Q < 0) {
				Q = 9999999;
			}
			// Оцениваем качество точки, при ее присоединении к соседнему (k-ому) сегменту
			for (auto& k : e){
				if (SegList[k].seg.empty()) {
					std::cout << "Logic Error\n";
					continue;
				}
				// Если размер сегмента < min_size, то
				if (SegList[k].seg.size()<min_size) QM = 9999999;
				// Вычисляем качество точки из сегмента
				else {
					QM = QPoint(fr0, fr1, SegList, k, i, j, SR);
				}
				// Если точка вылетела за границу изображения
				if (QM < 0) {
					continue;
				}

				if (QM < Q){
					Q = QM;
					ind = k;
				}
			}
			// Если качество получилось лучше, то граничную точку присоединяем к сегменту с номером ind
			if (ind != ind_src){

				for (int z = 0; z < SegList[seg.at<int>(p.y, p.x)].seg.size(); ++z){
					if (SegList[seg.at<int>(p.y, p.x)].seg[z] == p){
						SegList[seg.at<int>(p.y, p.x)].seg.erase(SegList[seg.at<int>(p.y, p.x)].seg.begin() + z);
						break;
					}
				}
				SegList[ind].seg.push_back(Point2d(p.x, p.y));

				isChanged = true;
				seg.at<int>(p.y, p.x) = ind;
				
			

		//		Point2d tmp_p = SegList[ind].tr->apply(Point2d(p.y, p.x));	
		//		flow.at<Vec2f>(p.y, p.x) = Vec2f(tmp_p.y-p.y, tmp_p.x-p.x);
				++cnt;
			}
		}
	}
	//std::cout << "Borders Update: " << cnt << " - cnt\n";
	UpdateSegmentList(seg, SegList);

	int size = SegList.size();
	for (int i = 0; i < size;){
		if (SegList[i].seg.empty()){
			SegList.erase(SegList.begin() + i);
			size = SegList.size();
		}
		else ++i;
	}

	BuildMatFromSegmentList(seg, seg.size(), SegList);
	
	return cnt;
}

template<typename TRANSFORM_TYPE>
void Segmentation(Mat& fr0, Mat& fr1, SegInfo& info, int min_seg_size, double TRH){
	#ifdef SHOW_SMAT
	std::thread winThd(winThread);
	winThd.detach();
	#endif
	int min_size = min_seg_size;
	double cost_step = TRH;
	double cost = 0;
	int i_step = fr0.rows / 20;
	int j_step = fr0.cols / 20;
	std::vector<Segment<TRANSFORM_TYPE>> SegList;
	Mat seg;
	// Вычисляем оптический поток для кадров в градациях серого
	Mat flow, top, g0, g1;
	cvtColor(fr0, g0, COLOR_BGR2GRAY);
	cvtColor(fr1, g1, COLOR_BGR2GRAY);
	calcOpticalFlowFarneback(g0, g1, flow, 0.4, 1, 16, 25, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
	// Разбиваем на прямоугольники и вычисляем для них соостветствующие преобразования
	Init(fr0, fr1, flow, seg, SegList, i_step, j_step);
	std::vector<std::set<int>> nIndexes;
	BuildMatFromSegmentList(seg, fr0.size(), SegList);
	UpdateTopology(nIndexes, seg, top);

	FILE *f = fopen("res.txt", "w"); fclose(f);

	testList(seg, SegList);
	SaveRes("res.txt", "init", fr0, fr1, seg, SegList);
	
	bool merged ;
	int cnt_merged;
	double Q0, Q1;

	do{
		merged = false;
		UpdateTopology(nIndexes, seg, top);
		cnt_merged = 0;
		std::cout << "Q0=" << (Q0=CalcQuality(fr0, fr1, seg, SegList)) << "\n";
		// Склейка сегментов с увеличением стоимости объединения
		do{
			cnt_merged = Merging(fr0, fr1, flow, SegList, nIndexes, min_seg_size, cost);
			std::cout << SegList.size() << " " << cnt_merged << " " << cost << "\n";
			if(cnt_merged!=0) cost += cost_step;
		} while (cnt_merged!=0);

		BuildMatFromSegmentList(seg, seg.size(), SegList);
		UpdateTopology(nIndexes, seg, top);

		#ifdef SHOW_SMAT
		ShowSegMatrix(fr0, SegList);
		#endif
		
		SaveRes("res.txt", "After merge", fr0, fr1, seg, SegList);

		std::cout << "Q=" << CalcQuality(fr0, fr1, seg, SegList) << " N=" << SegList.size() << "\n";

		while (BordersUpdate(fr0, fr1, seg, flow, SegList, min_seg_size, 1)>100){
			std::cout << "Q=" << CalcQuality(fr0, fr1, seg, SegList) << " N=" << SegList.size() << " \n";
		}
		SaveRes("res.txt", "After Border update", fr0, fr1, seg, SegList);
		std::cout << "Q1:" << (Q1 = CalcQuality(fr0, fr1, seg, SegList)) << "\n";
		
//		BuildSegmentListFromMat(seg, SegList, flow, fr0, fr1);
		UpdateSegmentList(seg, SegList); 
		testList(seg, SegList);


		SaveRes("res.txt", "After Border update3", fr0, fr1, seg, SegList);
		UpdateTopology(nIndexes, seg, top);

		SaveRes("res.txt", "After Border update4", fr0, fr1, seg, SegList);
		std::cout << "Q1=" << (Q1=CalcQuality(fr0, fr1, seg, SegList)) << "\n";
		std::cout << "Cost = " << cost << " cnt_merged=" << cnt_merged << "\n";

		#ifdef SHOW_SMAT
		ShowSegMatrix(fr0, SegList);
		#endif
	} while (cnt_merged>0 || abs(Q0 - Q1)>0.005);
	info.Q = Q1;
	info.seg = seg;
	TransformListToImage_MidColor(fr0, SegList, info.img);
}