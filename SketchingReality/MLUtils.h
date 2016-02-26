#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace ml {

std::vector<std::string> splitDataset(const std::string &str, char sep);
void splitDataset(const cv::Mat_<double>& data, float ratio1, cv::Mat_<double>& data1, cv::Mat_<double>& data2);
void splitDataset(const cv::Mat_<double>& data, float ratio1, float ratio2, cv::Mat_<double>& data1, cv::Mat_<double>& data2, cv::Mat_<double>& data3);
void shuffle(cv::Mat_<double>& data);
bool loadDataset(char* filename, cv::Mat_<double>& X, bool binary = false);
void saveDataset(char* filename, const cv::Mat_<double>& mat, bool binary = false);
void normalizeDataset(cv::Mat_<double> mat, cv::Mat_<double>& normalized_mat, cv::Mat_<double>& mean, cv::Mat_<double>& stddev);
void normalizeDataset2(cv::Mat_<double> mat, cv::Mat_<double>& normalized_mat, cv::Mat_<double>& mean, cv::Mat_<double>& stddev);
void addBias(cv::Mat& data);
void quadratic_dataset(const cv::Mat_<double>& data, cv::Mat_<double>& data2);

double mat_get_value(const cv::Mat& m, int r, int c);
void mat_set_value(cv::Mat& m, int r, int c, double val);
double mat_min(const cv::Mat& m);
cv::Mat mat_min(const cv::Mat& m1, const cv::Mat& m2);
double mat_max(const cv::Mat& m);
cv::Mat mat_max(const cv::Mat& m1, const cv::Mat& m2);
double mat_sum(const cv::Mat& m);
cv::Mat mat_square(const cv::Mat& m);
double mat_squared_sum(const cv::Mat& m);
void mat_clamp(cv::Mat& m, double min_val, double max_val);
cv::Mat mat_threhold(const cv::Mat& m, double threshold);
void mat_save(char* filename, const cv::Mat& mat, bool normalize = false);
double mat_variance(const cv::Mat& mat);
cv::Mat create_mask(int rows, int cols, int type, const cv::Point& center, int radius);
cv::Mat mat_mask(const cv::Mat& m, const cv::Mat& mask, float alpha);

double correlation(const cv::Mat_<double>& m1, const cv::Mat_<double>& m2);
void meanStdDev(const cv::Mat& src, cv::Mat_<double>& mean, cv::Mat_<double>& stddev);
double rmse(const cv::Mat_<double>& trueData, const cv::Mat_<double>& predData, bool averageColumns);


template<typename T>
T sqr(T val) { return val * val; }

// random
void initRand(int seed);
float genRand();
float genRand(float a, float b);
float genRandNormal(float mean, float variance);
float genRandInt(float a, float b, int num);

template<typename T>
int sampleFromCdf(const std::vector<T> &cdf) {
	float rnd = genRand(0, cdf.back());

	for (int i = 0; i < cdf.size(); ++i) {
		if (rnd <= cdf[i]) return i;
	}

	return cdf.size() - 1;
}

template<typename T>
int sampleFromPdf(const std::vector<T> &pdf) {
	if (pdf.size() == 0) return 0;

	std::vector<T> cdf(pdf.size(), 0.0f);
	cdf[0] = pdf[0];
	for (int i = 1; i < pdf.size(); ++i) {
		if (pdf[i] >= 0) {
			cdf[i] = cdf[i - 1] + pdf[i];
		} else {
			cdf[i] = cdf[i - 1];
		}
	}

	return sampleFromCdf(cdf);
}

template<typename T>
T sum(const std::vector<T>& vec) {
	T s = 0;
	for (int i = 0; i < vec.size(); ++i) {
		s += vec[i];
	}
	return s;
}

template<typename T>
T mean(const std::vector<T>& vec) {
	return sum(vec) / vec.size();
}

template<typename T>
T variance(const std::vector<T>& vec) {
	T avg = mean(vec);
	T s = 0;
	for (int i = 0; i < vec.size(); ++i) {
		s += (vec[i] - avg) * (vec[i] - avg);
	}
	return s / vec.size();
}

double log(double val, double base);

}