#include "MLUtils.h"
#include <string>

using namespace std;

namespace ml {


vector<string> splitDataset(const string &str, char sep) {
    vector<string> v;
    stringstream ss(str);
    string buffer;
    while (getline(ss, buffer, sep)) {
        v.push_back(buffer);
    }
    return v;
}

/**
 * データを、指定された比率に従い、2つのデータに分割する。
 *
 */
void splitDataset(const cv::Mat_<double>& data, float ratio1, cv::Mat_<double>& data1, cv::Mat_<double>& data2) {
	int rows1 = data.rows * ratio1;
	int rows2 = data.rows - rows1;

	data1 = cv::Mat_<double>(rows1, data.cols);
	data2 = cv::Mat_<double>(rows2, data.cols);

	for (int r = 0; r < data.rows; ++r) {
		if (r < rows1) {
			data.row(r).copyTo(data1.row(r));
		} else {
			data.row(r).copyTo(data2.row(r - rows1));
		}
	}
}

void splitDataset(const cv::Mat_<double>& data, float ratio1, float ratio2, cv::Mat_<double>& data1, cv::Mat_<double>& data2, cv::Mat_<double>& data3) {
	int rows1 = data.rows * ratio1;
	int rows2 = data.rows * ratio2;
	int rows3 = data.rows - rows1 - rows2;

	data1 = cv::Mat_<double>(rows1, data.cols);
	data2 = cv::Mat_<double>(rows2, data.cols);
	data3 = cv::Mat_<double>(rows3, data.cols);

	for (int r = 0; r < data.rows; ++r) {
		for (int c = 0; c < data.cols; ++c) {
			if (r < rows1) {
				data1(r, c) = data(r, c);
			} else if (r < rows1 + rows2) {
				data2(r - rows1, c) = data(r, c);
			} else {
				data3(r - rows1 - rows2, c) = data(r, c);
			}
		}
	}
}

/**
 * 行列の、行方向にシャッフルする。
 * ex. 1,2,3
 *     4,5,6
 *     7,8,9
 *     can be shuffled to
 *     4,5,6
 *     1,2,3
 *     7,8,9
 *
 * @param data 行列
 */
void shuffle(cv::Mat_<double>& data) {
	cv::Mat_<double> tmp = data.clone();

	std::vector<int> seeds;
	for (int i = 0; i < tmp.rows; ++i)
		seeds.push_back(i);

	cv::randShuffle(seeds);

	for (int i = 0; i < tmp.rows; ++i)
		tmp.row(seeds[i]).copyTo(data.row(i));
}

/**
 * ファイルからデータを読み込む。
 * ファイルの各行は、[x1, x2, ...],[y1, y2, ...] のフォーマットになっていること。
 * [x1, x2, ...]は、行列Xへ格納され、[y1, y2, ...]は、行列Yへ格納される。
 * もしファイルがオープンできない場合は、falseを返却する。
 * 
 */
bool loadDataset(char* filename, cv::Mat_<double>& mat, bool binary) {
	if (binary) {
		std::ifstream ifs(filename, ios::binary);
		if (ifs.fail()) {
			return false;
		}

		int rows, cols;
		ifs.read((char*)&rows, sizeof(int));
		ifs.read((char*)&cols, sizeof(int));

		mat = cv::Mat_<double>(rows, cols);
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				ifs.read((char*)&mat(r, c), sizeof(double));
			}
		}
		ifs.close();
	} else {
		std::ifstream ifs(filename);
		if (ifs.fail()) {
			return false;
		}

		vector<vector<double> > vec;

		std::string str;
		while (getline(ifs, str)) {
			if (str == "") break;

			vector<double> rec;
			vector<string> list = ml::splitDataset(str, ' ');
			for (int i = 0; i < list.size(); ++i) {
				rec.push_back(stof(list[i]));
			}
			vec.push_back(rec);
		}

		mat = cv::Mat_<double>(vec.size(), vec[0].size());

		for (int r = 0; r < vec.size(); ++r) {
			for (int c = 0; c < vec[r].size(); ++c) {
				mat(r, c) = vec[r][c];
			}
		}
		ifs.close();
	}

	return true;
}

void saveDataset(char* filename, const cv::Mat_<double>& mat, bool binary) {
	int N = mat.rows;

	if (binary) {
		ofstream ofs(filename, ios::binary);

		int rows = mat.rows;
		int cols = mat.cols;
		ofs.write((char*)&rows, sizeof(int));
		ofs.write((char*)&cols, sizeof(int));
		for (int iter = 0; iter < mat.rows; ++iter) {
			for (int c = 0; c < mat.cols; ++c) {
				ofs.write((char*)&mat(iter, c), sizeof(double));
			}
		}
		ofs.close();
	} else {
		ofstream ofs(filename);

		for (int iter = 0; iter < N; ++iter) {
			for (int c = 0; c < mat.cols; ++c) {
				if (c > 0) ofs << " ";
				ofs << mat(iter, c);
			}
			ofs << endl;
		}
		ofs.close();
	}
}

void normalizeDataset(cv::Mat_<double> mat, cv::Mat_<double>& normalized_mat, cv::Mat_<double>& mean, cv::Mat_<double>& stddev) {
	ml::meanStdDev(mat, mean, stddev);

	normalized_mat = (mat - cv::repeat(mean, mat.rows, 1)) / cv::repeat(stddev, mat.rows, 1);
	/*
	// 中央値を計算
	cv::Mat_<double> max_mat, min_mat;
	cv::reduce(mat, max_mat, 0, CV_REDUCE_MAX);
	cv::reduce(mat, min_mat, 0, CV_REDUCE_MIN);
	mu = (max_mat + min_mat) * 0.5;

	// 中央値を0にそろえる
	normalized_mat = mat - cv::repeat(mu, mat.rows, 1);

	// [-1, 1]にする
	cv::reduce(cv::abs(normalized_mat), abs_max, 0, CV_REDUCE_MAX);
	normalized_mat /= cv::repeat(abs_max, mat.rows, 1);
	*/

	/*
	// normalization
	cv::reduce(mat, mu, 0, CV_REDUCE_AVG);
	normalized_mat = mat - cv::repeat(mu, mat.rows, 1);

	// [-1, 1]にする
	cv::reduce(cv::abs(normalized_mat), abs_max, 0, CV_REDUCE_MAX);
	normalized_mat /= cv::repeat(abs_max, mat.rows, 1);
	*/
}

void normalizeDataset2(cv::Mat_<double> mat, cv::Mat_<double>& normalized_mat, cv::Mat_<double>& minimum, cv::Mat_<double>& maximum) {
	cv::reduce(mat, minimum, 0, CV_REDUCE_MIN);
	normalized_mat = mat - cv::repeat(minimum, mat.rows, 1);

	cv::reduce(normalized_mat, maximum, 0, CV_REDUCE_MAX);
	normalized_mat /= cv::repeat(maximum, mat.rows, 1);
}


/**
 * 一番右の列に1を追加する。
 */
void addBias(cv::Mat& data) {
	cv::Mat tmp = data.clone();
	data = cv::Mat::ones(tmp.rows, tmp.cols + 1, data.type());

	cv::Mat roi(data, cv::Rect(0, 0, tmp.cols, tmp.rows));
	tmp.copyTo(roi);
}

/**
 * quadratic
 */
void quadratic_dataset(const cv::Mat_<double>& data, cv::Mat_<double>& data2) {
	int D = data.cols;
	int D2 = D + D * (D + 1) * 0.5;

	data2 = cv::Mat_<double>::zeros(data.rows, D2);
	for (int r = 0; r < data.rows; ++r) {
		int index = 0;
		for (int c1 = 0; c1 < D; ++c1) {
			for (int c2 = c1; c2 < D; ++c2) {
				data2(r, index++) = data(r, c1) * data(r, c2);
			}
		}
		for (int c = 0; c < D; ++c) {
			data2(r, index++) = data(r, c);
		}
	}
}

/**
 * 行列の指定した行、列の値を返却する。
 *
 * @param m		行列
 * @param r		行
 * @param c		列
 * @return		値
 */
double mat_get_value(const cv::Mat& m, int r, int c) {
	if (m.type() == CV_8U) {
		return m.at<uchar>(r, c);
	} else if (m.type() == CV_32F) {
		return m.at<float>(r, c);
	} else if (m.type() == CV_64F) {
		return m.at<double>(r, c);
	} else {
		return 0.0;
	}
}

/**
 * 行列の指定した行、列の値を更新する。
 *
 * @param m		行列
 * @param r		行
 * @param c		列
 * @return		値
 */
void mat_set_value(cv::Mat& m, int r, int c, double val) {
	if (m.type() == CV_8U) {
		m.at<uchar>(r, c) = val;
	} else if (m.type() == CV_32F) {
		m.at<float>(r, c) = val;
	} else if (m.type() == CV_64F) {
		m.at<double>(r, c) = val;
	}
}

double mat_min(const cv::Mat& m) {
	cv::Mat tmp;
	cv::reduce(m, tmp, 0, CV_REDUCE_MIN);
	cv::reduce(tmp, tmp, 1, CV_REDUCE_MIN);

	return mat_get_value(tmp, 0, 0);
}

/**
 * 2つの行列の、要素ごとのmaxを計算し、返却する。
 *
 * @param m1		行列1
 * @param m2		行列2
 * @return			要素ごとのmax
 */
cv::Mat mat_min(const cv::Mat& m1, const cv::Mat& m2) {
	cv::Mat ret = m1.clone();

	for (int r = 0; r < m1.rows; ++r) {
		for (int c = 0; c < m1.cols; ++c) {
			double v1 = mat_get_value(m1, r, c);
			double v2 = mat_get_value(m2, r, c);

			if (v2 < v1) {
				mat_set_value(ret, r, c, v2);
			}
		}
	}
	
	return ret;
}

double mat_max(const cv::Mat& m) {
	cv::Mat tmp;
	cv::reduce(m, tmp, 0, CV_REDUCE_MAX);
	cv::reduce(tmp, tmp, 1, CV_REDUCE_MAX);

	return mat_get_value(tmp, 0, 0);
}

/**
 * 2つの行列の、要素ごとのmaxを計算し、返却する。
 *
 * @param m1		行列1
 * @param m2		行列2
 * @return			要素ごとのmax
 */
cv::Mat mat_max(const cv::Mat& m1, const cv::Mat& m2) {
	cv::Mat ret = m1.clone();

	for (int r = 0; r < m1.rows; ++r) {
		for (int c = 0; c < m1.cols; ++c) {
			double v1 = mat_get_value(m1, r, c);
			double v2 = mat_get_value(m2, r, c);

			if (v2 > v1) {
				mat_set_value(ret, r, c, v2);
			}
		}
	}

	return ret;
}

/**
 * 行列の要素和を返却する。
 *
 * @param m		行列
 */
double mat_sum(const cv::Mat& m) {
	cv::Mat s;
	cv::reduce(m, s, 0, CV_REDUCE_SUM);
	cv::reduce(s, s, 1, CV_REDUCE_SUM);
	return mat_get_value(s, 0, 0);
}

/**
 * 行列の各要素を二乗する。
 *
 * @param m		行列
 * @return		行列の各要素を二乗した結果
 */
cv::Mat mat_square(const cv::Mat& m) {
	return m.mul(m);
}

/**
 * 行列の要素の二乗和を返却する。
 *
 * @param m		行列
 * @return		行列の各要素の二乗和
 */
double mat_squared_sum(const cv::Mat& m) {
	return mat_sum(mat_square(m));
}

/**
 * 行列の各要素について、最小値より小さい値は最小値に、最大値より大きい値は最大値になるようにする。
 *
 * @param m			行列
 * @param min_val	最小値
 * @param max_val	最大値
 */
void mat_clamp(cv::Mat& m, double min_val, double max_val) {
	for (int r = 0; r < m.rows; ++r) {
		for (int c = 0; c < m.cols; ++c) {
			double v = mat_get_value(m, r, c);
			if (v < min_val) mat_set_value(m, r, c, min_val);
			if (v > max_val) mat_set_value(m, r, c, max_val);
		}
	}
}

/**
 * 行列の各要素について、しきい値より大きい場合は1、しきい値以下の場合は0にする。
 *
 * @param m				行列
 * @param threshold		しきい値
 * @return				結果の行列
 */
cv::Mat mat_threhold(const cv::Mat& m, double threshold) {
	cv::Mat ret(m.size(), m.type());

	for (int r = 0; r < m.rows; ++r) {
		for (int c = 0; c < m.cols; ++c) {
			double v = mat_get_value(m, r, c);
			if (v > threshold) mat_set_value(ret, r, c, 1.0);
			else mat_set_value(ret, r, c, 0.0);
		}
	}

	return ret;
}

/**
 * 行列matを画像として保存する。ただし、1が白色、0が黒色となる。
 *
 * @param filename		ファイル名
 * @param mat			行列
 * @normalize			trueなら、最小値を黒、最大値を白色となるよう、スケールする
 */
void mat_save(char* filename, const cv::Mat& mat, bool normalize) {
	cv::Mat_<double> img = mat.clone();

	if (normalize) {
		double max_val = mat_max(mat);
		double min_val = mat_min(mat);

		double scale = 255.0 / (max_val - min_val);
		img.convertTo(img, CV_64F, scale, -min_val * scale);
	} else {
		img.convertTo(img, CV_64F, 255);
	}

	cv::flip(img, img, 0);
	cv::imwrite(filename, img);
}

double mat_variance(const cv::Mat& mat) {
	cv::Mat avg;
	cv::reduce(mat, avg, 0, CV_REDUCE_AVG);

	return mat_squared_sum(mat - cv::repeat(avg, mat.rows, 1)) / mat.rows;
}

/**
 * centerを中心とし、半径radiusの円の内側が1、外側が0のmask行列を返却する。
 *
 * @param rows		行数
 * @param cols		列数
 * @param center	中心
 * @param radius	半径
 * @return			mask行列
 */
cv::Mat create_mask(int rows, int cols, int type, const cv::Point& center, int radius) {
	cv::Mat mask = cv::Mat::zeros(rows, cols, type);
	cv::circle(mask, center, radius, 1, -1);
	return mask;
}

/**
 * マスク処理する。
 * alpha=1なら、マスク値=1のピクセルのみ残して、後は0にする。
 * alpha=0なら、元の画像のまま。
 *
 * @param m		元の画像
 * @param mask	マスク
 * @param alpha	アルファ値
 * @return		マスク処理した結果
 */
cv::Mat mat_mask(const cv::Mat& m, const cv::Mat& mask, float alpha) {
	cv::Mat m2;
	cv::subtract(m, cv::Mat::zeros(m.size(), m.type()), m2, mask);

	return m * (1.0 - alpha) + m2 * alpha;
}

double correlation(const cv::Mat_<double>& m1, const cv::Mat_<double>& m2) {
	// 列ベクトルにする
	cv::Mat_<double> mat1 = m1;
	if (mat1.rows == 1) mat1 = mat1.t();
	cv::Mat_<double> mat2 = m2;
	if (mat2.rows == 2) mat2 = mat2.t();

	// データ数
	int N = mat1.rows;

	// 平均を計算する
	cv::Mat_<double> avg1;
	cv::reduce(mat1, avg1, 0, CV_REDUCE_AVG);
	cv::Mat_<double> avg2;
	cv::reduce(mat2, avg2, 0, CV_REDUCE_AVG);

	cv::Mat_<double> mat1_hat = mat1 - cv::repeat(avg1, N, 1);
	cv::Mat_<double> mat2_hat = mat2 - cv::repeat(avg2, N, 1);

	return mat_sum(mat1_hat.mul(mat2_hat)) / sqrt(mat_sum(mat1_hat.mul(mat1_hat))) / sqrt(mat_sum(mat2_hat.mul(mat2_hat)));
}

/**
 * 平均と標準偏差を計算する。
 * srcの各行が各データ。
 */
void meanStdDev(const cv::Mat& src, cv::Mat_<double>& mean, cv::Mat_<double>& stddev) {
	int N = src.rows;

	mean = cv::Mat_<double>(1, src.cols);
	stddev = cv::Mat_<double>(1, src.cols);

	for (int c = 0; c < src.cols; ++c) {
		cv::Scalar m, s;
		cv::meanStdDev(src.col(c), m, s);

		mean(0, c) = m.val[0];
		stddev(0, c) = s.val[0] * sqrt((double)N / (N - 1));
	}
}

/**
 * 予測データのRMSEを計算する。
 * averageColumnsがtrueなら、D次元のデータの要素毎のRMSEを平均して計算する。
 * さもなければ、D次元のデータについて、通常のRMSEを計算する。
 *
 * @param trueData			真の値
 * @param predData			予測値
 * @param averageColumns	true/false
 */
double rmse(const cv::Mat_<double>& trueData, const cv::Mat_<double>& predData, bool averageColumns) {
	cv::Mat_<double> error;

	if (averageColumns) {
		cv::reduce(mat_square(trueData - predData), error, 1, CV_REDUCE_AVG);
		cv::reduce(error, error, 0, CV_REDUCE_AVG);
		cv::sqrt(error, error);
	} else {
		cv::reduce(mat_square(trueData - predData), error, 1, CV_REDUCE_SUM);
		cv::sqrt(error, error);
		cv::reduce(error, error, 0, CV_REDUCE_AVG);
	}

	return error(0, 0);
}

void initRand(int seed) {
	srand(seed);

	int n = seed % 13;
	for (int i = 0; i < n; ++i) {
		genRand();
	}
}

/**
 * Uniform乱数[0, 1)を生成する
 */
float genRand() {
	return rand() / (float(RAND_MAX) + 1);
}

/**
 * 指定された範囲[a, b)のUniform乱数を生成する
 */
float genRand(float a, float b) {
	return genRand() * (b - a) + a;
}

/**
 * Normal distributionを使用して乱数を生成する。
 */
float genRandNormal(float mean, float variance) {
	/*
	static std::default_random_engine generator;

	std::normal_distribution<float> distribution(mean, sqrtf(variance));
	return distribution(generator);
	*/

#if 1
	float m = mean;
	float s = sqrt(variance);

	/* mean m, standard deviation s */
	float x1, x2, w, y1;
	static float y2;
	static int use_last = 0;

	if (use_last) {	/* use value from previous call */
		y1 = y2;
		use_last = 0;
	} else {
		do {
			x1 = 2.0 * genRand(0.0f, 1.0f) - 1.0;
			x2 = 2.0 * genRand(0.0f, 1.0f) - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 );

		w = sqrt( (-2.0 * ::log( w ) ) / w );
		y1 = x1 * w;
		y2 = x2 * w;
		use_last = 1;
	}

	return m + y1 * s;
#endif
}

float genRandInt(float a, float b, int num) {
	int r = genRand(0, num);
	return a + (b - a) / (float)(num - 1) * r;
}

double log(double val, double base) {
	return ::log(val) / ::log(base);
}

}