#include <set>
#include <vector>
#include "opencv2\core\mat.hpp"
#include <fstream>

using namespace cv;


template<typename T>
T FVal(const Mat& m, int y, int x){
	int ix(x), iy(y);
	if (x < 0) ix = 0;
	else if (x>m.cols - 1) ix = m.cols - 1;
	if (y < 0) iy = 0;
	else if (y>m.rows - 1) iy = m.rows - 1;
	return m.at<T>(iy, ix);
}

// Интерполирует значение матрицы f в точке v = (x,y)
Vec3d interpColor(const Mat& m, const Point2d &v){
	Vec3d res;
	int ix((int)v.x), iy((int)v.y);
	double dx(v.x - (int)v.x), dy(v.y - (int)v.y), v0, v1;
	for (int k = 0; k < 3; ++k){
		v0 = ((double)FVal<Vec3b>(m, iy, ix + 1)[k] - (double)FVal<Vec3b>(m, iy, ix)[k])*dx + (double)FVal<Vec3b>(m, iy, ix)[k];
		v1 = ((double)FVal<Vec3b>(m, iy + 1, ix + 1)[k] - (double)FVal<Vec3b>(m, iy + 1, ix)[k])*dx + (double)FVal<Vec3b>(m, iy + 1, ix)[k];
		res[k] = (v1 - v0)*dy + v0;
	}
	return res;
}

void CalcTopologyMat(const Mat& seg, Mat& top){
	int seg_num(0), tmp;
	std::set<int> s;
	for (int i = 0; i < seg.rows; ++i){
		for (int j = 0; j < seg.cols; ++j){
			s.insert(seg.at<int>(i, j));
			if ((tmp = seg.at<int>(i, j))>seg_num){
				seg_num = tmp;
			}
		}
	}
	top = Mat::zeros(Size(seg_num + 1, seg_num + 1), CV_32SC1);
	int p, p0, p1;
	int ii, jj;

	for (int i = 0; i < seg.rows - 1; ++i){
		for (int j = 0; j < seg.cols - 1; ++j){
			p = seg.at<int>(i, j);
			p0 = seg.at<int>(i, j + 1);
			p1 = seg.at<int>(i + 1, j);

			if (p != p0) {
				top.at<int>(p, p0)++;
				top.at<int>(p0, p)++;
			}
			if (p != p1) {
				top.at<int>(p, p1)++;
				top.at<int>(p1, p)++;
			}
		}
	}

	for (int i = 1; i < seg.rows - 1; ++i){
		jj = seg.cols - 1;
		p = seg.at<int>(i, jj);
		p0 = seg.at<int>(i, jj - 1);
		p1 = seg.at<int>(i - 1, jj);

		if (p != p0) {
			top.at<int>(p, p0)++;
			top.at<int>(p0, p)++;
		}
		if (p != p1) {
			top.at<int>(p, p1)++;
			top.at<int>(p1, p)++;
		}
	}

	for (int j = 1; j < seg.cols; ++j){
		ii = seg.rows - 1;
		p = seg.at<int>(ii, j);
		p0 = seg.at<int>(ii, j - 1);
		p1 = seg.at<int>(ii - 1, j);

		if (p != p0) {
			top.at<int>(p, p0)++;
			top.at<int>(p0, p)++;
		}
		if (p != p1) {
			top.at<int>(p, p1)++;
			top.at<int>(p1, p)++;
		}
	}

	jj = seg.cols - 1;
	p = seg.at<int>(0, jj);
	p0 = seg.at<int>(0, jj - 1);
	p1 = seg.at<int>(1, jj);
	if (p != p0) {
		top.at<int>(p, p0)++;
		top.at<int>(p0, p)++;
	}
	if (p != p1) {
		top.at<int>(p, p1)++;
		top.at<int>(p1, p)++;
	}

	ii = seg.rows - 1;
	p = seg.at<int>(ii, 0);
	p0 = seg.at<int>(ii, 1);
	p1 = seg.at<int>(ii - 1, 0);
	if (p != p0) {
		top.at<int>(p, p0)++;
		top.at<int>(p0, p)++;
	}
	if (p != p1) {
		top.at<int>(p, p1)++;
		top.at<int>(p1, p)++;
	}
}


void UpdateTopology(std::vector<std::set<int>>& nIndexes, Mat& seg, Mat& top){
	nIndexes.clear();
	CalcTopologyMat(seg, top);
	for (int i = 0; i < top.rows; ++i){
		nIndexes.push_back(std::set<int>());
	}
	int index(0);
	for (int i = 0; i < top.rows; ++i){
		for (int j = 0; j < top.rows; ++j){
			index = top.at<int>(i, j);
			if (index != 0){
				nIndexes[i].insert(j);
			}
		}
	}
}

// возвращает в edges номера соседних сегментов
// n - число заграничных соседей
bool isEdge(Mat& seg, const Point2d& p, std::set<int>& edges, int& n){
	n = 0;
	edges.clear();
	int x((int)p.x), y((int)p.y);
	int val_p = seg.at<int>(y, x);
	for (int i(y - 1); i <= (y + 1); ++i){
		if (i < 0 || i >= seg.rows) continue;
		for (int j(x - 1); j <= (x + 1); ++j){
			if (j < 0 || j >= seg.cols) continue;
			if (seg.at<int>(i, j) != val_p) {
				edges.insert(seg.at<int>(i, j));
				n++;
			}
		}
	}
	return !edges.empty();
}

// возвращает в edges список заграничных соседей
// n - число заграничных соседей
bool isEdgeV(Mat& seg, const Point2d& p, std::vector<Point2d>& edges, int& n){
	n = 0;
	edges.clear();
	int x((int)p.x), y((int)p.y);
	int val_p = seg.at<int>(y, x);
	for (int i(y - 1); i <= (y + 1); ++i){
		if (i < 0 || i >= seg.rows) continue;
		for (int j(x - 1); j <= (x + 1); ++j){
			if (j < 0 || j >= seg.cols) continue;
			if (seg.at<int>(i, j) != val_p) {
				edges.push_back(Point2d(i, j));
				n++;
			}
		}
	}
	return !edges.empty();
}
// обновление списка смежности
void MergeNList(std::vector<std::set<int>>& nIndexes, int from, int to, bool erase_from = false){

	auto eraseIndex = [](std::set<int>& s, int erase_index)->bool{
		auto search = s.find(erase_index);
		if (search != s.end()) {
			s.erase(search);
			return true;
		}
		return false;
	};

	for (int i = 0; i < nIndexes.size(); ++i){
		if (i != from){
			if (eraseIndex(nIndexes[i], from)) {
				if (i != to) nIndexes[i].insert(to);
			}
		}
		else {
			for (auto& e : nIndexes[i]){
				if (e != to) nIndexes[to].insert(e);
			}
			nIndexes[i].clear();
		}
	}
	if (erase_from){
		nIndexes.erase(nIndexes.begin() + from);
		std::vector<int> gr_ind;
		for (int i = 0; i < nIndexes.size(); ++i){
			gr_ind.clear();
		loop:
			for (auto& e : nIndexes[i]){
				if (e > from) {
					gr_ind.push_back(e - 1);
					nIndexes[i].erase(e);
					goto loop;
				}
			}
			nIndexes[i].insert(gr_ind.begin(), gr_ind.end());
		}
	}
}

void SaveFlow(const cv::Mat& flow){
	std::ofstream f("flow.txt");
	for (int i = 0; i < flow.rows; ++i){
		for (int j = 0; j < flow.cols; ++j){
			f << i << " " << j << " " << flow.at<Vec2f>(i, j)[0] << " " << flow.at<Vec2f>(i, j)[0] << "\n";
		}
	}
	f.close();
}

