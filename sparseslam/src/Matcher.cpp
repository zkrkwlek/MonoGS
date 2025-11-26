#include <../GaussianSparseSLAM/include/Matcher.h>
#include <Eigen/Dense>

namespace GaussianSparseSLAM {
	void Matcher::matchEigen(std::vector<cv::DMatch>& vecMatches, const cv::Mat& feats1_cv, const cv::Mat& feats2_cv, float min_cossim) {
        // OpenCV Mat -> Eigen Matrix
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            feats1(feats1_cv.ptr<float>(), feats1_cv.rows, feats1_cv.cols);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            feats2(feats2_cv.ptr<float>(), feats2_cv.rows, feats2_cv.cols);

        int N = feats1.rows();
        int M = feats2.rows();

        // 코사인 유사도 계산
        Eigen::MatrixXf cossim = feats1 * feats2.transpose();

        // Query -> Train
        std::vector<int> match12(N);
        std::vector<float> max_vals12(N);

        for (int i = 0; i < N; i++) {
            Eigen::Index max_idx;
            max_vals12[i] = cossim.row(i).maxCoeff(&max_idx);
            match12[i] = max_idx;
        }

        // Train -> Query
        std::vector<int> match21(M);
        for (int j = 0; j < M; j++) {
            Eigen::Index max_idx;
            cossim.col(j).maxCoeff(&max_idx);
            match21[j] = max_idx;
        }

        // Mutual check
        for (int i = 0; i < N; i++) {
            int j = match12[i];
            if (match21[j] == i && max_vals12[i] > min_cossim) {
                cv::DMatch m;
                m.queryIdx = i;
                m.trainIdx = j;
                vecMatches.push_back(m);
            }
        }
	}
}